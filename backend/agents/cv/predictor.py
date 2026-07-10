import logging, base64, io, json, uuid
from typing import Dict, Optional, List
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import settings

log = logging.getLogger("ml-platform.cv_predictor")


class CVPredictor:
    def __init__(self, cv_models_dir: Path, factory):
        self.cv_models_dir = cv_models_dir
        self.factory = factory

    def predict(self, model_id: str, image_path: str, session_id: str = None) -> Dict:
        try:
            from agents.cv.data_engine import CVDataEngine
            engine = CVDataEngine(settings.CV_UPLOADS_DIR)

            resolved = image_path
            if session_id:
                candidate = settings.CV_UPLOADS_DIR / f"{session_id}_extracted" / image_path
                if candidate.exists():
                    resolved = str(candidate)
                elif Path(image_path).exists():
                    resolved = image_path
            elif not Path(image_path).exists():
                meta_path = settings.CV_UPLOADS_DIR / f"{Path(image_path).stem}_meta.json"
                if meta_path.exists():
                    resolved = str(meta_path.parent / f"{Path(image_path).stem}_extracted" / image_path)

            item = engine.load_image(resolved)
            if not item:
                return {"success": False, "error": "LOAD_ERROR",
                        "message": f"Could not load image: {image_path}"}

            img = item["image"]
            orig_h, orig_w = img.shape[:2]

            model = self.factory.load_model(model_id)
            if model is None:
                return {"success": False, "error": "MODEL_NOT_FOUND",
                        "message": f"Model '{model_id}' not found"}

            meta = self.factory.get_metadata(model_id)
            task_type = meta.get("task_type", "classification")

            model_input_shape = model.input_shape[1:3]  # (H, W)
            img_resized = cv2.resize(img, (model_input_shape[1], model_input_shape[0]))
            img_norm = img_resized.astype(np.float32) / 255.0
            X = np.expand_dims(img_norm, axis=0)

            pred = model.predict(X, verbose=0)

            result = {
                "success": True,
                "model_id": model_id,
                "task_type": task_type,
                "filename": item["filename"],
                "original_size": {"width": orig_w, "height": orig_h},
            }

            if task_type == "segmentation":
                return self._process_segmentation(result, pred, img, orig_h, orig_w)
            else:
                return self._process_classification(result, pred, meta)

        except Exception as e:
            log.error(f"CV predict failed: {e}", exc_info=True)
            return {"success": False, "error": "PREDICT_ERROR", "message": str(e)}

    def _process_classification(self, result: Dict, pred: np.ndarray,
                                 meta: Dict) -> Dict:
        score = float(pred[0][0])
        confidence = max(score, 1 - score)
        is_positive = score > 0.5
        result.update({
            "prediction": "positive" if is_positive else "negative",
            "confidence": round(confidence, 4),
            "raw_score": round(score, 4),
            "findings": [],
        })
        if meta:
            result["model_info"] = {
                "val_accuracy": meta.get("val_accuracy"),
                "val_loss": meta.get("val_loss"),
            }
        return result

    def _process_segmentation(self, result: Dict, pred: np.ndarray,
                               img: np.ndarray, orig_h: int, orig_w: int) -> Dict:
        mask = pred[0, :, :, 0]
        mask_resized = cv2.resize(mask, (orig_w, orig_h))
        mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255

        confidence = float(np.mean(mask[mask > 0.5])) if mask[mask > 0.5].size > 0 else 0.0
        total_pixels = orig_h * orig_w
        anomaly_pixels = int(np.sum(mask_bin > 0) / 255)
        anomaly_pct = round(anomaly_pixels / total_pixels * 100, 2)

        overlay = img.copy()
        overlay[mask_bin > 0] = (255, 0, 0)
        blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        overlay_pil = Image.fromarray(overlay)
        blended_pil = Image.fromarray(blended)
        mask_pil = Image.fromarray(mask_bin, mode="L")

        def _img_to_b64(pil_img) -> str:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        regions = self._find_regions(mask_bin, orig_w, orig_h)
        findings = self._describe_findings(regions, anomaly_pct)

        result.update({
            "prediction": "anomaly_detected" if anomaly_pixels > 0 else "normal",
            "confidence": round(confidence, 4),
            "anomaly_percentage": anomaly_pct,
            "anomaly_pixels": anomaly_pixels,
            "total_pixels": total_pixels,
            "regions": regions,
            "findings": findings,
            "overlay_image": _img_to_b64(blended_pil),
            "mask_image": _img_to_b64(mask_pil),
            "raw_overlay_image": _img_to_b64(overlay_pil),
        })
        return result

    def _find_regions(self, mask_bin: np.ndarray, orig_w: int, orig_h: int) -> List[Dict]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_bin, connectivity=8
        )
        regions = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 50:
                continue
            x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            region_pct = round(area / (orig_w * orig_h) * 100, 2)
            regions.append({
                "id": i,
                "x": x, "y": y, "width": w, "height": h,
                "center_x": cx, "center_y": cy,
                "area_pixels": area,
                "area_percentage": region_pct,
            })
        regions.sort(key=lambda r: r["area_pixels"], reverse=True)
        return regions

    def _describe_findings(self, regions: List[Dict], total_pct: float) -> List[str]:
        findings = []
        if not regions:
            findings.append("No abnormalities detected. The image appears normal.")
            return findings

        findings.append(
            f"Detected {len(regions)} region(s) of interest covering {total_pct}% of the image."
        )
        for i, r in enumerate(regions[:5]):
            loc = self._describe_location(r)
            findings.append(
                f"Region {i+1}: {loc} — {r['area_pixels']} px ({r['area_percentage']}% of image)"
            )
        if len(regions) > 5:
            findings.append(f"Plus {len(regions) - 5} smaller region(s).")
        return findings

    @staticmethod
    def _describe_location(r: Dict) -> str:
        cx_pct = r["center_x"] / (r["center_x"] + r.get("width", 1) or 1)
        cy_pct = r["center_y"] / (r["center_y"] + r.get("height", 1) or 1)
        h_pos = "left" if cx_pct < 0.4 else "right" if cx_pct > 0.6 else "central"
        v_pos = "upper" if cy_pct < 0.4 else "lower" if cy_pct > 0.6 else "mid"
        return f"{v_pos}-{h_pos} region ({r['width']}×{r['height']} px)"

    def predict_on_folder(self, model_id: str, session_id: str) -> Dict:
        from agents.cv.data_engine import CVDataEngine
        engine = CVDataEngine(settings.CV_UPLOADS_DIR)
        X, meta = engine.get_training_data(session_id)
        model = self.factory.load_model(model_id)
        if model is None:
            return {"success": False, "error": "MODEL_NOT_FOUND", "message": "Model not found"}

        preds = model.predict(X, verbose=0)
        task_type = self.factory.get_metadata(model_id).get("task_type", "classification")
        results = []
        positive_count = 0
        for i, m in enumerate(meta["meta"]):
            if task_type == "segmentation":
                mask = preds[i, :, :, 0]
                has_anomaly = float(np.mean(mask[mask > 0.5])) > 0 if mask[mask > 0.5].size > 0 else 0
                is_pos = bool(np.sum(mask > 0.5) > 100)
                conf = float(np.mean(mask[mask > 0.5])) if mask[mask > 0.5].size > 0 else 0.0
            else:
                score = float(preds[i][0])
                is_pos = score > 0.5
                conf = max(score, 1 - score)
            if is_pos:
                positive_count += 1
            results.append({
                "image_id": m["image_id"],
                "filename": m["filename"],
                "prediction": "positive" if is_pos else "negative",
                "confidence": round(conf, 4),
            })

        return {
            "success": True,
            "total": len(results),
            "positive_count": positive_count,
            "negative_count": len(results) - positive_count,
            "results": results,
        }
