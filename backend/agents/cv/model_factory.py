import json, logging, time, uuid
from typing import Dict, Optional, Callable
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import settings

log = logging.getLogger("ml-platform.cv_model")


class CVModelFactory:
    def __init__(self, cv_models_dir: Path, cv_uploads_dir: Path):
        self.cv_models_dir = cv_models_dir
        self.cv_uploads_dir = cv_uploads_dir
        self.cv_models_dir.mkdir(parents=True, exist_ok=True)

    def build_classifier(self, input_shape=(224, 224, 3), num_classes=2) -> keras.Model:
        base = keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet", pooling="avg"
        )
        base.trainable = False
        inputs = keras.Input(shape=input_shape)
        x = base(inputs, training=False)
        x = keras.layers.Dropout(0.3)(x)
        if num_classes == 2:
            outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        else:
            outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs, name="cv_classifier")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=settings.CV_LEARNING_RATE),
            loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        return model

    def build_segmentation_unet(self, input_shape=(224, 224, 3)) -> keras.Model:
        inputs = keras.Input(shape=input_shape)
        c1 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
        c1 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
        p1 = keras.layers.MaxPooling2D()(c1)
        c2 = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
        c2 = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
        p2 = keras.layers.MaxPooling2D()(c2)
        c3 = keras.layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
        c3 = keras.layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
        p3 = keras.layers.MaxPooling2D()(c3)
        c4 = keras.layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
        c4 = keras.layers.Conv2D(256, 3, activation="relu", padding="same")(c4)
        u5 = keras.layers.UpSampling2D()(c4)
        u5 = keras.layers.concatenate([u5, c3])
        c5 = keras.layers.Conv2D(128, 3, activation="relu", padding="same")(u5)
        u6 = keras.layers.UpSampling2D()(c5)
        u6 = keras.layers.concatenate([u6, c2])
        c6 = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(u6)
        u7 = keras.layers.UpSampling2D()(c6)
        u7 = keras.layers.concatenate([u7, c1])
        c7 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(u7)
        outputs = keras.layers.Conv2D(1, 1, activation="sigmoid", padding="same")(c7)
        model = keras.Model(inputs, outputs, name="cv_segmentation_unet")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=settings.CV_LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.MeanIoU(num_classes=2, name="mean_iou")],
        )
        return model

    def _progress_callback(self, job_status: dict, job_id: str):
        class ProgressCallback(keras.callbacks.Callback):
            def __init__(self, total_epochs):
                super().__init__()
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                pct = int((epoch + 1) / self.total_epochs * 90) + 5
                step = f"Epoch {epoch+1}/{self.total_epochs} — loss={logs.get('loss',0):.4f}, acc={logs.get('accuracy',0):.4f}"
                if job_status and job_id and job_id in job_status:
                    job_status[job_id].update({"progress": pct, "step": step})

        return ProgressCallback

    def train(self, session_id: str, task_type: str = "classification",
              epochs: Optional[int] = None, batch_size: Optional[int] = None,
              learning_rate: Optional[float] = None,
              job_status=None, job_id=None) -> Dict:
        epochs = epochs or settings.CV_DEFAULT_EPOCHS
        batch_size = batch_size or settings.CV_BATCH_SIZE
        learning_rate = learning_rate if learning_rate else settings.CV_LEARNING_RATE

        def _p(pct, step):
            if job_status and job_id and job_id in job_status:
                job_status[job_id].update({"progress": pct, "step": step})

        try:
            _p(2, "Loading training data...")
            X = np.load(str(self.cv_uploads_dir / f"{session_id}_X.npy"))
            with open(self.cv_uploads_dir / f"{session_id}_meta.json") as f:
                meta = json.load(f)

            has_labels = (self.cv_uploads_dir / f"{session_id}_y.npy").exists()
            if not has_labels:
                return {"success": False, "error": "NO_LABELS",
                        "message": "No labels found. Upload labeled data first."}

            y = np.load(str(self.cv_uploads_dir / f"{session_id}_y.npy"))
            ti, vi = meta["train_indices"], meta["val_indices"]
            X_tr, X_val = X[ti], X[vi]
            y_tr, y_val = y[ti], y[vi]

            if task_type == "segmentation":
                model = self.build_segmentation_unet(input_shape=X.shape[1:])
                y_tr = y_tr[..., np.newaxis] if y_tr.ndim == 3 else y_tr
                y_val = y_val[..., np.newaxis] if y_val.ndim == 3 else y_val
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss="binary_crossentropy", metrics=["accuracy"],
                )
            else:
                num_classes = len(np.unique(y))
                model = self.build_classifier(input_shape=X.shape[1:], num_classes=num_classes)
                loss = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=loss, metrics=["accuracy"],
                )

            _p(5, f"Training {task_type} model for {epochs} epochs...")
            cb = self._progress_callback(job_status, job_id)(epochs)
            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[cb, keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, monitor="val_loss")],
                verbose=0,
            )

            _p(95, "Saving model...")
            model_id = f"cv_{task_type}_{session_id}_{int(time.time())}"
            model_path = self.cv_models_dir / f"{model_id}.keras"
            model.save(str(model_path))

            val_loss = float(history.history["val_loss"][-1])
            val_acc = float(history.history.get("val_accuracy", [0])[-1])
            train_loss = float(history.history["loss"][-1])
            train_acc = float(history.history.get("accuracy", [0])[-1])

            meta_record = {
                "model_id": model_id,
                "task_type": task_type,
                "session_id": session_id,
                "epochs": epochs,
                "batch_size": batch_size,
                "input_shape": list(X.shape[1:]),
                "train_samples": len(X_tr),
                "val_samples": len(X_val),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "history": {k: [round(float(v), 6) for v in vals] for k, vals in history.history.items()},
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            json_path = self.cv_models_dir / f"{model_id}.json"
            with open(json_path, "w") as f:
                json.dump(meta_record, f, indent=2)

            _p(100, "Done!")
            return {
                "success": True,
                "model_id": model_id,
                "task_type": task_type,
                "val_accuracy": round(val_acc, 4),
                "val_loss": round(val_loss, 4),
                "epochs_trained": epochs,
            }

        except Exception as e:
            log.error(f"CV train failed: {e}", exc_info=True)
            return {"success": False, "error": "CV_TRAIN_ERROR", "message": str(e)}

    def load_model(self, model_id: str) -> Optional[keras.Model]:
        path = self.cv_models_dir / f"{model_id}.keras"
        if not path.exists():
            return None
        return keras.models.load_model(str(path))

    def get_metadata(self, model_id: str) -> Dict:
        path = self.cv_models_dir / f"{model_id}.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def list_models(self) -> Dict:
        models = []
        for f in sorted(self.cv_models_dir.glob("*.json"),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            meta = json.loads(f.read_text())
            models.append({
                "model_id": meta.get("model_id", f.stem),
                "task_type": meta.get("task_type"),
                "val_accuracy": meta.get("val_accuracy"),
                "val_loss": meta.get("val_loss"),
                "trained_at": meta.get("trained_at"),
                "train_samples": meta.get("train_samples"),
            })
        return {"success": True, "models": models, "count": len(models)}

    def delete_model(self, model_id: str) -> Dict:
        for ext in [".keras", ".json"]:
            p = self.cv_models_dir / f"{model_id}{ext}"
            if p.exists():
                p.unlink()
        return {"success": True, "message": f"Deleted {model_id}"}
