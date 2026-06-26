import json, logging
from typing import Dict, List, Callable, Optional
from pathlib import Path

from .data_engine import MLError
from config import settings
from .evaluator import _load_artifact

log = logging.getLogger("ml-platform.visualizer")


class Visualizer:
    def __init__(self, models_dir: Path, ml_libs_loader: Callable):
        self.models_dir = models_dir
        self._load_ml_libs = ml_libs_loader

    def get_chart_data(self, session_id: str, data_engine,
                       plot_type: str = "histogram",
                       columns: Optional[List[str]] = None) -> Dict:
        try:
            libs = self._load_ml_libs()
            np = libs["np"]
            df = data_engine.load_session_df(session_id)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if plot_type == "histogram":
                cols = columns or numeric_cols[:4]
                if not cols:
                    raise MLError("NO_NUMERIC", "No numeric columns available for histogram.")
                result = {}
                for col in cols:
                    if col not in df.columns:
                        continue
                    s = df[col].dropna()
                    counts, edges = np.histogram(s, bins=20)
                    result[col] = {"bins": [round(float(e), 4) for e in edges[:-1]], "counts": counts.tolist()}
                return {"success": True, "plot_type": "histogram", "data": result}

            elif plot_type == "scatter":
                cols = columns or numeric_cols[:2]
                if len(cols) < 2:
                    raise MLError("NOT_ENOUGH_COLS", "Scatter needs 2+ numeric columns.")
                sample = df[cols[:2]].dropna().head(settings.SCATTER_MAX_POINTS)
                return {"success": True, "plot_type": "scatter", "x_label": cols[0], "y_label": cols[1],
                        "data": [{"x": float(r[cols[0]]), "y": float(r[cols[1]])} for _, r in sample.iterrows()]}

            elif plot_type == "correlation":
                cols = columns or numeric_cols[:settings.CORRELATION_MAX_COLS]
                if len(cols) < 2:
                    raise MLError("NOT_ENOUGH_COLS", "Correlation needs 2+ numeric columns.")
                corr = df[cols].corr()
                matrix = []
                for row_col in cols:
                    for col_col in cols:
                        val = corr.loc[row_col, col_col]
                        matrix.append({"row": row_col, "col": col_col, "value": round(float(val), 4) if not np.isnan(val) else 0})
                return {"success": True, "plot_type": "correlation", "columns": cols, "data": matrix}

            elif plot_type == "boxplot":
                cols = columns or numeric_cols[:6]
                if not cols:
                    raise MLError("NO_NUMERIC", "No numeric columns for boxplot.")
                result = {}
                for col in cols:
                    s = df[col].dropna()
                    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                    iqr = q3 - q1
                    result[col] = {"min": round(float(s.min()), 4), "q1": round(q1, 4), "median": round(float(s.median()), 4),
                                   "q3": round(q3, 4), "max": round(float(s.max()), 4),
                                   "outliers": [round(float(v), 4) for v in s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)].head(50)]}
                return {"success": True, "plot_type": "boxplot", "data": result}

            elif plot_type == "bar":
                cols = columns or categorical_cols[:1]
                if not cols:
                    raise MLError("NO_CATEGORICAL", "No categorical columns for bar chart.")
                vc = df[cols[0]].value_counts().head(20)
                return {"success": True, "plot_type": "bar", "label": cols[0],
                        "data": [{"name": str(k), "count": int(v)} for k, v in vc.items()]}

            else:
                raise MLError("UNKNOWN_PLOT", f"Plot type '{plot_type}' not supported.")

        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"Chart failed: {e}", exc_info=True)
            return MLError("CHART_ERROR", f"Chart failed: {str(e)}").to_dict()

    def feature_importance(self, model_id: str, active_models: dict, model_factory) -> Dict:
        try:
            artifact = active_models.get(model_id)
            if artifact is None:
                artifact = _load_artifact(self.models_dir, active_models, model_id)
            if not isinstance(artifact, dict) or "model" not in artifact:
                raise MLError("UNSUPPORTED", "Feature importance requires a tabular model.")
            model = artifact["model"]
            features = artifact["columns"]
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_.tolist()
            elif hasattr(model, "coef_"):
                import numpy as np
                coef = model.coef_
                imps = np.abs(coef).mean(axis=0).tolist() if coef.ndim > 1 else [abs(c) for c in coef.tolist()]
            else:
                raise MLError("NO_IMPORTANCE", "Model does not expose feature importances.")
            pairs = sorted(zip(features, imps), key=lambda x: x[1], reverse=True)
            return {"success": True, "model_id": model_id,
                    "data": [{"feature": f, "importance": round(imp, 6)} for f, imp in pairs]}
        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"Feature importance failed: {e}", exc_info=True)
            return MLError("IMPORTANCE_ERROR", f"Failed: {str(e)}").to_dict()

    def confusion_matrix_data(self, model_id: str, session_id: str, data_engine) -> Dict:
        try:
            from sklearn.metrics import confusion_matrix
            artifact = _load_artifact(self.models_dir, {}, model_id)
            if not isinstance(artifact, dict):
                raise MLError("UNSUPPORTED", "Confusion matrix requires a tabular model.")
            model, scaler, features = artifact["model"], artifact["scaler"], artifact["columns"]
            meta_path = self.models_dir / f"{model_id}.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            target = meta.get("target", "")
            df = data_engine.load_session_df(session_id)
            if target not in df.columns:
                raise MLError("TARGET_MISSING", f"Target '{target}' not found.")
            X_s = scaler.transform(df[features])
            y_pred = model.predict(X_s)
            cm = confusion_matrix(df[target], y_pred).tolist()
            classes = meta.get("classes", sorted(set(str(c) for c in df[target].unique())))
            return {"success": True, "matrix": cm, "labels": classes, "model_id": model_id}
        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"CM failed: {e}", exc_info=True)
            return MLError("CM_ERROR", str(e)).to_dict()

    def roc_curve_data(self, model_id: str, session_id: str, data_engine) -> Dict:
        try:
            from sklearn.metrics import roc_curve, auc
            import numpy as np
            artifact = _load_artifact(self.models_dir, {}, model_id)
            if not isinstance(artifact, dict) or not hasattr(artifact.get("model"), "predict_proba"):
                raise MLError("NO_PROBA", "Model does not support probability predictions.")
            model, scaler, features = artifact["model"], artifact["scaler"], artifact["columns"]
            meta_path = self.models_dir / f"{model_id}.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            target = meta.get("target", "")
            df = data_engine.load_session_df(session_id)
            if target not in df.columns:
                raise MLError("TARGET_MISSING", f"Target '{target}' not found.")
            X_s = scaler.transform(df[features])
            classes = sorted(df[target].unique().tolist())
            if len(classes) != 2:
                raise MLError("NOT_BINARY", "ROC curve requires binary classification. "
                              f"Found {len(classes)} classes: {classes}")
            y_score = model.predict_proba(X_s)[:, 1]
            fpr, tpr, _ = roc_curve(df[target], y_score)
            roc_auc_val = float(auc(fpr, tpr))
            return {"success": True, "fpr": [round(float(x), 4) for x in fpr],
                    "tpr": [round(float(x), 4) for x in tpr], "auc": round(roc_auc_val, 4)}
        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"ROC failed: {e}", exc_info=True)
            return MLError("ROC_ERROR", str(e)).to_dict()

    def elbow_plot_data(self, session_id: str, data_engine, max_k: int = 10) -> Dict:
        try:
            from sklearn.cluster import KMeans
            libs = self._load_ml_libs()
            df = data_engine.load_session_df(session_id)
            cols = df.select_dtypes(include=[libs["np"].number]).columns.tolist()
            if not cols:
                raise MLError("NO_NUMERIC", "No numeric columns.")
            X = df[cols].dropna()
            scaler = libs["StandardScaler"]()
            X_s = scaler.fit_transform(X)
            inertias = []
            max_k = min(max_k, settings.ELBOW_MAX_K, len(X) - 1)
            ks = list(range(1, max_k + 1))
            for k in ks:
                km = KMeans(n_clusters=k, random_state=42, n_init=settings.KMEANS_N_INIT)
                km.fit(X_s)
                inertias.append(float(km.inertia_))
            return {"success": True, "plot_type": "elbow",
                    "data": [{"k": k, "inertia": inertias[i]} for i, k in enumerate(ks)]}
        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"Elbow failed: {e}", exc_info=True)
            return MLError("ELBOW_ERROR", str(e)).to_dict()

    def export_plot_png(self, session_id: str, data_engine,
                        plot_type: str = "histogram", columns: Optional[List[str]] = None) -> Dict:
        try:
            libs = self._load_ml_libs()
            np = libs["np"]
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns; import base64; import io
            df = data_engine.load_session_df(session_id)
            cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()[:4]
            fig, ax = plt.subplots(figsize=(10, 6))
            if plot_type == "histogram":
                df[cols].hist(ax=ax, bins=20, edgecolor="white")
            elif plot_type == "scatter" and len(cols) >= 2:
                ax.scatter(df[cols[0]], df[cols[1]], alpha=0.6)
                ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1])
            elif plot_type == "correlation":
                sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
            elif plot_type == "boxplot":
                df[cols].boxplot(ax=ax)
            buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(); buf.seek(0)
            return {"success": True, "plot_type": plot_type, "plot_base64": base64.b64encode(buf.read()).decode("utf-8"), "format": "png"}
        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"Export PNG failed: {e}", exc_info=True)
            return MLError("EXPORT_ERROR", str(e)).to_dict()
