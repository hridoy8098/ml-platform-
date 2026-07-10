import os, zipfile, json, uuid, logging, io, shutil
from typing import List, Dict, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import settings

log = logging.getLogger("ml-platform.cv_data")

CV_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
CV_DICOM_EXTENSIONS = {".dcm", ".dicom"}
CV_ALLOWED_EXTENSIONS = CV_IMAGE_EXTENSIONS | CV_DICOM_EXTENSIONS


class CVDataEngine:
    def __init__(self, cv_uploads_dir: Path):
        self.cv_uploads_dir = cv_uploads_dir
        self.cv_uploads_dir.mkdir(parents=True, exist_ok=True)

    def extract_zip(self, zip_path: str, extract_to: str) -> List[Dict]:
        extracted = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
            for root, _, files in os.walk(extract_to):
                for fname in files:
                    ext = Path(fname).suffix.lower()
                    if ext in CV_ALLOWED_EXTENSIONS:
                        path = str(Path(root) / fname)
                        loaded = self.load_image(path)
                        if loaded:
                            extracted.append(loaded)
        log.info(f"Extracted {len(extracted)} images from zip")
        return extracted

    def load_image(self, path: str) -> Optional[Dict]:
        ext = Path(path).suffix.lower()
        try:
            if ext in CV_DICOM_EXTENSIONS:
                return self._load_dicom(path)
            return self._load_standard(path)
        except Exception as e:
            log.warning(f"Failed to load {path}: {e}")
            return None

    def _load_standard(self, path: str) -> Dict:
        img = cv2.imread(path)
        if img is None:
            img_pil = Image.open(path).convert("RGB")
            img = np.array(img_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        return {
            "image_id": str(uuid.uuid4())[:8],
            "path": path,
            "filename": Path(path).name,
            "image": img_rgb,
            "height": h,
            "width": w,
            "channels": img_rgb.shape[2] if len(img_rgb.shape) > 2 else 1,
            "format": "image",
            "original_shape": (h, w),
        }

    def _load_dicom(self, path: str) -> Dict:
        import pydicom
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        h, w = img_rgb.shape[:2]
        meta = {
            "PatientID": getattr(ds, "PatientID", ""),
            "PatientName": str(getattr(ds, "PatientName", "")),
            "StudyDescription": getattr(ds, "StudyDescription", ""),
            "Modality": getattr(ds, "Modality", ""),
            "BodyPart": getattr(ds, "BodyPartExamined", ""),
        }
        return {
            "image_id": str(uuid.uuid4())[:8],
            "path": path,
            "filename": Path(path).name,
            "image": img_rgb,
            "height": h,
            "width": w,
            "channels": 3,
            "format": "dicom",
            "dicom_meta": meta,
            "original_shape": (h, w),
        }

    def preprocess(self, images: List[Dict], target_size: tuple = (224, 224)) -> tuple:
        X, meta_list = [], []
        for item in images:
            img = item["image"]
            img_resized = cv2.resize(img, target_size)
            img_norm = img_resized.astype(np.float32) / 255.0
            X.append(img_norm)
            meta_list.append({
                "image_id": item["image_id"],
                "filename": item["filename"],
                "original_shape": item["original_shape"],
                "dicom_meta": item.get("dicom_meta"),
                "format": item["format"],
            })
        return np.array(X), meta_list

    def load_dataset(self, session_id: str, path: str) -> Dict:
        path_obj = Path(path)
        if not path_obj.exists():
            return {"success": False, "error": "FILE_NOT_FOUND", "message": f"Path not found: {path}"}

        images = []
        if path_obj.suffix.lower() == ".zip":
            extract_dir = str(self.cv_uploads_dir / f"{session_id}_extracted")
            images = self.extract_zip(path, extract_dir)
        elif path_obj.is_dir():
            for f in path_obj.iterdir():
                if f.suffix.lower() in CV_ALLOWED_EXTENSIONS:
                    loaded = self.load_image(str(f))
                    if loaded:
                        images.append(loaded)
        elif path_obj.suffix.lower() in CV_ALLOWED_EXTENSIONS:
            loaded = self.load_image(str(path))
            if loaded:
                images.append(loaded)

        if not images:
            return {"success": False, "error": "NO_IMAGES", "message": "No valid images found"}

        X, meta_list = self.preprocess(images)
        total = len(images)
        train_count = max(1, int(total * 0.8))
        indices = np.random.permutation(total)
        train_idx, val_idx = indices[:train_count], indices[train_count:]

        session_info = {
            "session_id": session_id,
            "path": path,
            "total_images": total,
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "image_shape": list(X.shape[1:]),
            "meta": meta_list,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        }

        np_path = self.cv_uploads_dir / f"{session_id}_X.npy"
        np.save(str(np_path), X)
        meta_path = self.cv_uploads_dir / f"{session_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(session_info, f, indent=2, default=str)

        return {
            "success": True,
            "session_id": session_id,
            "message": f"Loaded {total} images ({train_count} train, {len(val_idx)} val)",
            "total_images": total,
            "image_shape": list(X.shape[1:]),
            "preview": [{"image_id": m["image_id"], "filename": m["filename"],
                          "shape": list(m["original_shape"]), "format": m["format"]}
                        for m in meta_list[:20]],
        }

    def get_training_data(self, session_id: str) -> tuple:
        X = np.load(str(self.cv_uploads_dir / f"{session_id}_X.npy"))
        with open(self.cv_uploads_dir / f"{session_id}_meta.json") as f:
            meta = json.load(f)
        return X, meta

    def augment(self, X: np.ndarray, y: Optional[np.ndarray] = None, factor: int = 2) -> tuple:
        augmented = [X]
        labels = [y] if y is not None else None
        for _ in range(factor - 1):
            aug = []
            for img in X:
                img_uint8 = (img * 255).astype(np.uint8)
                if np.random.rand() > 0.5:
                    img_uint8 = cv2.flip(img_uint8, 1)
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    h, w = img_uint8.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    img_uint8 = cv2.warpAffine(img_uint8, M, (w, h),
                                                borderMode=cv2.BORDER_REFLECT)
                aug.append(img_uint8.astype(np.float32) / 255.0)
            augmented.append(np.array(aug))
            if labels is not None and y is not None:
                labels.append(y)
        X_aug = np.concatenate(augmented, axis=0)
        if labels is not None and y is not None:
            y_aug = np.concatenate(labels, axis=0)
            return X_aug, y_aug
        return X_aug, None

    def extract_and_process(self, zip_path: str) -> Dict:
        session_id = str(uuid.uuid4())[:12]
        return self.load_dataset(session_id, zip_path)

    def get_session_info(self, session_id: str) -> Dict:
        meta_path = self.cv_uploads_dir / f"{session_id}_meta.json"
        X_path = self.cv_uploads_dir / f"{session_id}_X.npy"
        if not meta_path.exists() or not X_path.exists():
            return {"success": False, "error": "SESSION_NOT_FOUND",
                    "message": f"No data found for session '{session_id}'"}
        with open(meta_path) as f:
            meta = json.load(f)
        X = np.load(str(X_path))
        return {
            "success": True,
            "session_id": session_id,
            "total_images": meta["total_images"],
            "image_shape": meta["image_shape"],
            "preview": [{"image_id": m["image_id"], "filename": m["filename"],
                          "shape": list(m.get("original_shape", [])),
                          "format": m.get("format", "image")}
                        for m in meta.get("meta", [])[:20]],
        }

    def cleanup_session(self, session_id: str):
        for f in self.cv_uploads_dir.glob(f"{session_id}*"):
            f.unlink(missing_ok=True)
        extract_dir = self.cv_uploads_dir / f"{session_id}_extracted"
        if extract_dir.exists():
            shutil.rmtree(str(extract_dir), ignore_errors=True)
