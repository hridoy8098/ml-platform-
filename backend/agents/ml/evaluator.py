import json, logging
from typing import Dict, Callable
from pathlib import Path
from .data_engine import MLError
from config import settings

log = logging.getLogger("ml-platform.evaluator")


def _load_artifact(models_dir: Path, active_models: dict, model_id: str) -> dict:
    if model_id in active_models:
        return active_models[model_id]
    p = models_dir / f"{model_id}.joblib"
    if not p.exists():
        raise MLError("MODEL_NOT_FOUND", f"Model '{model_id}' not found.")
    import joblib
    art = joblib.load(p)
    meta_path = models_dir / f"{model_id}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        expected_hash = meta.get("_model_hash", "")
        if expected_hash:
            import hashlib
            actual = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    actual.update(chunk)
            if actual.hexdigest() != expected_hash:
                raise MLError("MODEL_CORRUPTED", f"Model '{model_id}' integrity check failed. File may be corrupted.")
    if not isinstance(art, dict):
        raise MLError("INVALID_ARTIFACT", "Invalid model artifact format.")
    active_models[model_id] = art
    return art


class Evaluator:
    def __init__(self, models_dir: Path, active_models: dict,
                 ml_libs_loader: Callable, model_factory, data_engine):
        self.models_dir    = models_dir
        self.active_models = active_models
        self._load         = ml_libs_loader
        self.factory       = model_factory
        self.data_engine   = data_engine

    def evaluate(self, session_id: str, model_id: str) -> Dict:
        try:
            import numpy as np
            from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix, classification_report,
                mean_squared_error, mean_absolute_error, r2_score)

            artifact = _load_artifact(self.models_dir, self.active_models, model_id)
            meta     = self.factory.get_metadata(model_id)

            df       = self.data_engine.load_session_df(session_id)
            features = artifact["columns"]
            target   = meta.get("target")
            task     = meta.get("task", "classification")

            miss = [f for f in features if f not in df.columns]
            if miss:
                raise MLError("MISSING_COLS", f"Dataset missing columns: {miss}")

            X = df[features]
            y = df[target]

            X_s   = artifact["scaler"].transform(X)
            model = artifact["model"]
            preds = model.predict(X_s)

            result = {"success": True, "model_id": model_id, "task": task}

            if task == "classification":
                classes = sorted(y.unique().tolist())
                avg     = "binary" if len(classes) == 2 else "weighted"
                result.update({
                    "accuracy":  round(float(accuracy_score(y, preds)), 4),
                    "precision": round(float(precision_score(y, preds, average=avg, zero_division=0)), 4),
                    "recall":    round(float(recall_score(y, preds, average=avg, zero_division=0)), 4),
                    "f1":        round(float(f1_score(y, preds, average=avg, zero_division=0)), 4),
                    "classes":   [str(c) for c in classes],
                    "confusion_matrix": confusion_matrix(y, preds).tolist(),
                    "per_class": classification_report(y, preds, output_dict=True, zero_division=0),
                    "train_score": meta.get("train_score"),
                    "test_score":  round(float(accuracy_score(y, preds)), 4),
                })
                if hasattr(model, "predict_proba") and len(classes) == 2:
                    proba = model.predict_proba(X_s)[:, 1]
                    try:
                        result["auc_roc"] = round(float(roc_auc_score(y, proba)), 4)
                    except Exception as e:
                        log.debug(f"AUC failed: {e}")

            else:
                ss_res = np.sum((y.values - preds) ** 2)
                ss_tot = np.sum((y.values - y.mean()) ** 2)
                r2_val = float(1 - ss_res / ss_tot) if ss_tot else 0.0
                mape   = float(np.mean(np.abs((y.values - preds) / (np.abs(y.values) + 1e-8))) * 100)

                sample = list(zip(y.values[:200].tolist(), preds[:200].tolist()))
                result.update({
                    "rmse":  round(float(mean_squared_error(y, preds) ** 0.5), 4),
                    "mae":   round(float(mean_absolute_error(y, preds)), 4),
                    "r2":    round(r2_val, 4),
                    "mape":  round(mape, 2),
                    "actual_vs_pred": [{"actual": a, "predicted": p} for a, p in sample],
                    "train_score": meta.get("train_score"),
                    "test_score":  round(r2_val, 4),
                })

            result["feature_importance"] = meta.get("feature_importance")
            return result

        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Evaluation failed: {e}", exc_info=True)
            return MLError("EVAL_ERROR", str(e)).to_dict()


class Predictor:
    def __init__(self, models_dir: Path, active_models: dict,
                 ml_libs_loader: Callable, model_factory):
        self.models_dir    = models_dir
        self.active_models = active_models
        self._load         = ml_libs_loader
        self.factory       = model_factory

    def get_model_info(self, model_id: str) -> Dict:
        try:
            meta = self.factory.get_metadata(model_id)
            if not meta:
                raise MLError("NO_META", f"No metadata for '{model_id}'.",
                              "Retrain the model.")
            info = {k: v for k, v in meta.items() if not k.startswith("_")}
            return {"success": True, **info}
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Model info failed: {e}", exc_info=True)
            return MLError("INFO_ERROR", str(e)).to_dict()

    def predict(self, model_id: str, input_data: Dict) -> Dict:
        try:
            artifact = _load_artifact(self.models_dir, self.active_models, model_id)
            meta     = self.factory.get_metadata(model_id)

            model    = artifact["model"]
            scaler   = artifact["scaler"]
            features = artifact["columns"]

            missing = [f for f in features if f not in input_data]
            if missing:
                raise MLError("MISSING_FEATURES", f"Missing: {missing}",
                              f"Provide all features: {features}")

            row = {}
            for f in features:
                v    = input_data[f]
                dtype = meta.get("feature_types", {}).get(f, "")
                try:
                    row[f] = float(v) if ("float" in dtype or "int" in dtype) else v
                except (ValueError, TypeError):
                    row[f] = v

            import pandas as pd
            input_df = pd.DataFrame([row])[features]
            X_s      = scaler.transform(input_df)
            pred     = model.predict(X_s)[0]

            proba = None
            if hasattr(model, "predict_proba"):
                pa    = model.predict_proba(X_s)[0]
                cls = meta.get("classes", [str(c) for c in model.classes_])
                proba = {str(c): round(float(p), 4) for c, p in zip(cls, pa)}

            return {
                "success": True,
                "prediction": pred.item() if hasattr(pred, "item") else str(pred),
                "probabilities": proba,
                "confidence": round(float(max(proba.values())), 4) if proba else None,
                "model_id": model_id,
                "task": meta.get("task"),
            }
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Prediction failed: {e}", exc_info=True)
            return MLError("PREDICT_ERROR", str(e)).to_dict()


class Deployer:
    def __init__(self, models_dir: Path, active_models: dict, model_factory):
        self.models_dir    = models_dir
        self.active_models = active_models
        self.factory       = model_factory

    def list_models(self) -> Dict:
        try:
            import time as _time
            models = []
            for f in sorted(self.models_dir.glob("*.joblib"),
                            key=lambda x: x.stat().st_ctime, reverse=True):
                stat = f.stat()
                meta = self.factory.get_metadata(f.stem)
                models.append({
                    "model_id":      f.stem,
                    "size":          self._fmt(stat.st_size),
                    "created":       _time.ctime(stat.st_ctime),
                    "task":          meta.get("task"),
                    "algorithm":     meta.get("algorithm"),
                    "target":        meta.get("target"),
                    "features":      meta.get("features", []),
                    "feature_types": meta.get("feature_types", {}),
                    "classes":       meta.get("classes"),
                    "accuracy":      meta.get("accuracy"),
                    "rmse":          meta.get("rmse"),
                    "r2":            meta.get("r2"),
                    "trained_at":    meta.get("trained_at"),
                    "dataset_shape": meta.get("dataset_shape"),
                })
            return {"success": True, "models": models, "count": len(models)}
        except Exception as e:
            log.error(f"List models failed: {e}", exc_info=True)
            return MLError("LIST_ERROR", str(e)).to_dict()

    def delete_model(self, model_id: str) -> Dict:
        try:
            mp = self.models_dir / f"{model_id}.joblib"
            jp = self.models_dir / f"{model_id}.json"
            if not mp.exists():
                raise MLError("NOT_FOUND", f"Model '{model_id}' not found.")
            mp.unlink()
            if jp.exists(): jp.unlink()
            self.active_models.pop(model_id, None)
            log.info(f"Deleted model: {model_id}")
            return {"success": True, "message": f"Deleted '{model_id}'"}
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Delete failed: {e}", exc_info=True)
            return MLError("DELETE_ERROR", str(e)).to_dict()

    def export_model(self, model_id: str, format: str = "joblib") -> Dict:
        try:
            from fastapi.responses import FileResponse
            from fastapi import HTTPException
            p = self.models_dir / f"{model_id}.joblib"
            if not p.exists():
                raise MLError("NOT_FOUND", f"Model '{model_id}' not found.")
            return {"success": True, "export_path": str(p), "filename": f"{model_id}.{format}"}
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Export failed: {e}", exc_info=True)
            return MLError("EXPORT_ERROR", str(e)).to_dict()

    @staticmethod
    def _fmt(b):
        for u in ["B","KB","MB","GB"]:
            if abs(b) < 1024: return f"{b:.1f} {u}"
            b /= 1024
        return f"{b:.1f} TB"
