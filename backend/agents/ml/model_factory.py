import json, time, logging, hashlib
from datetime import datetime, timezone
from typing import Dict, List, Callable
from pathlib import Path
from .data_engine import MLError
from config import settings

log = logging.getLogger("ml-platform.model_factory")


def _compute_model_hash(model_path: Path) -> str:
    actual = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            actual.update(chunk)
    return actual.hexdigest()


def _verify_model_integrity(filepath: Path, expected_hash: str) -> bool:
    if not expected_hash:
        return True
    try:
        actual = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                actual.update(chunk)
        return actual.hexdigest() == expected_hash
    except Exception:
        return False


class ModelFactory:
    def __init__(self, models_dir: Path, sessions_dir: Path,
                 ml_libs_loader: Callable, active_models: dict, data_engine):
        self.models_dir   = models_dir
        self.sessions_dir = sessions_dir
        self._load        = ml_libs_loader
        self.active_models = active_models
        self.data_engine  = data_engine

    def _progress(self, job_status, job_id, pct, step):
        if job_status and job_id and job_id in job_status:
            job_status[job_id].update({"progress": pct, "step": step})

    def _save_artifact(self, model_id: str, artifact: dict, meta: dict):
        model_path = self.models_dir / f"{model_id}.joblib"
        libs = self._load()
        libs["joblib"].dump(artifact, model_path)
        meta["_model_hash"] = _compute_model_hash(model_path)
        self._save_metadata(model_id, meta)
        from threading import Lock
        self._active_lock = getattr(self, '_active_lock', Lock())
        with self._active_lock:
            self.active_models[model_id] = artifact

    def _save_metadata(self, model_id: str, meta: dict):
        tmp = self.models_dir / f"{model_id}.json.tmp"
        final = self.models_dir / f"{model_id}.json"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        tmp.rename(final)

    def get_metadata(self, model_id: str) -> dict:
        p = self.models_dir / f"{model_id}.json"
        if not p.exists():
            return {}
        meta = json.loads(p.read_text())
        model_file = self.models_dir / f"{model_id}.joblib"
        if model_file.exists():
            expected = meta.get("_model_hash", "")
            if expected and not _verify_model_integrity(model_file, expected):
                log.warning(f"Model {model_id} integrity check FAILED")
                return {}
        return meta

    def _prepare(self, session_id, target, features=None):
        libs = self._load()
        df   = self.data_engine.load_session_df(session_id)

        if target and target not in df.columns:
            raise MLError("TARGET_NOT_FOUND", f"Column '{target}' not found.",
                          f"Available: {list(df.columns)}")

        feat_cols = [c for c in (features or df.columns) if c != target]
        non_num   = [c for c in feat_cols if df[c].dtype == "object"]
        if non_num:
            raise MLError("NON_NUMERIC", f"Columns {non_num} have text values.",
                          "Encode them in the Prepare tab first.")

        miss_cols = [c for c in feat_cols if df[c].isna().any()]
        if miss_cols:
            raise MLError("MISSING_VALUES", f"Columns {miss_cols} have missing values.",
                          "Fill or drop missing values in the Prepare tab.")

        X = df[feat_cols]
        y = df[target] if target else None
        return X, y, df, libs

    def _train_and_select(self, candidates, X_tr_s, y_tr, X_te_s, y_te, task, _p):
        best_score = float("-inf")
        best_model = None
        chosen = ""
        total = len(candidates)
        for i, (name, m) in enumerate(candidates.items()):
            _p(30 + int(40 * i / total), f"Trying {name}...")
            try:
                m.fit(X_tr_s, y_tr)
                score = m.score(X_te_s, y_te)
                log.debug(f"{name} score: {score}")
                if score > best_score:
                    best_score, best_model, chosen = score, m, name
            except Exception as e:
                log.debug(f"{name} failed: {e}")
                continue
        if not best_model:
            raise MLError("ALL_FAILED", f"All {task} models failed to train.")
        return best_model, chosen, best_score

    def _extract_feature_importance(self, model, columns):
        if hasattr(model, "feature_importances_"):
            fi = sorted(zip(columns, model.feature_importances_),
                        key=lambda x: x[1], reverse=True)
            return [{"feature": f, "importance": round(float(v), 6)} for f, v in fi]
        return None

    def train_classifier(self, session_id, target, algorithm="auto",
                         test_size=0.2, cv_folds=5, random_seed=42,
                         features=None, job_status=None, job_id=None) -> Dict:
        def _p(pct, step): self._progress(job_status, job_id, pct, step)
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.naive_bayes import GaussianNB
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import classification_report

            _p(5, "Loading dataset...")
            X, y, df, libs = self._prepare(session_id, target, features)

            _p(15, "Splitting data...")
            try:
                X_tr, X_te, y_tr, y_te = libs["train_test_split"](
                    X, y, test_size=test_size, random_state=random_seed, stratify=y)
            except Exception:
                X_tr, X_te, y_tr, y_te = libs["train_test_split"](
                    X, y, test_size=test_size, random_state=random_seed)

            _p(25, "Scaling features...")
            scaler = libs["StandardScaler"]()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            candidates = {
                "random_forest":   RandomForestClassifier(n_estimators=100, random_state=random_seed),
                "gradient_boost":  GradientBoostingClassifier(random_state=random_seed),
                "logistic":        LogisticRegression(max_iter=1000, random_state=random_seed),
                "svm":             SVC(probability=True, random_state=random_seed),
                "naive_bayes":     GaussianNB(),
                "knn":             KNeighborsClassifier(),
                "decision_tree":   DecisionTreeClassifier(random_state=random_seed),
                "extra_trees":     ExtraTreesClassifier(n_estimators=100, random_state=random_seed),
                "adaboost":        AdaBoostClassifier(random_state=random_seed),
            }
            try:
                from xgboost import XGBClassifier
                candidates["xgboost"] = XGBClassifier(eval_metric="logloss", random_state=random_seed, verbosity=0)
            except ImportError:
                pass

            if algorithm != "auto":
                if algorithm not in candidates:
                    raise MLError("UNKNOWN_ALGO", f"Algorithm '{algorithm}' not found.",
                                  f"Available: {list(candidates.keys())}")
                _p(50, f"Training {algorithm}...")
                model = candidates[algorithm]
                model.fit(X_tr_s, y_tr)
                chosen = algorithm
            else:
                model, chosen, _ = self._train_and_select(candidates, X_tr_s, y_tr, X_te_s, y_te, "classification", _p)

            _p(78, "Evaluating...")
            preds    = model.predict(X_te_s)
            accuracy = float(libs["accuracy_score"](y_te, preds))
            report   = classification_report(y_te, preds, output_dict=True, zero_division=0)

            cv_score = None
            if cv_folds > 1:
                _p(85, f"Running {cv_folds}-fold CV...")
                try:
                    from sklearn.model_selection import cross_val_score
                    cvs = cross_val_score(model, X_tr_s, y_tr, cv=min(cv_folds, 10))
                    cv_score = float(cvs.mean())
                except Exception as e:
                    log.debug(f"CV failed: {e}")

            fi = self._extract_feature_importance(model, X.columns)

            _p(93, "Saving model...")
            model_id = f"classifier_{target}_{chosen}_{int(time.time())}"
            artifact = {"model": model, "scaler": scaler, "columns": list(X.columns)}

            meta = {
                "model_id": model_id, "task": "classification",
                "algorithm": chosen, "target": target,
                "features": list(X.columns),
                "feature_types": {c: str(df[c].dtype) for c in X.columns},
                "classes": sorted([str(c) for c in y.unique()]),
                "accuracy": round(accuracy, 4),
                "cv_score": round(cv_score, 4) if cv_score else None,
                "train_score": round(float(model.score(X_tr_s, y_tr)), 4),
                "per_class_report": report,
                "feature_importance": fi,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "dataset_shape": list(df.shape),
            }
            self._save_artifact(model_id, artifact, meta)
            _p(100, "Done!")

            return {"success": True, **{k: meta[k] for k in
                ["model_id","task","algorithm","target","features","classes",
                 "accuracy","cv_score","feature_importance","trained_at"]}}

        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Classifier train failed: {e}", exc_info=True)
            return MLError("TRAIN_ERROR", str(e)).to_dict()

    def train_regressor(self, session_id, target, algorithm="auto",
                        test_size=0.2, cv_folds=5, random_seed=42,
                        features=None, job_status=None, job_id=None) -> Dict:
        def _p(pct, step): self._progress(job_status, job_id, pct, step)
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor
            import numpy as np

            _p(5, "Loading dataset...")
            X, y, df, libs = self._prepare(session_id, target, features)

            _p(15, "Splitting data...")
            X_tr, X_te, y_tr, y_te = libs["train_test_split"](
                X, y, test_size=test_size, random_state=random_seed)

            _p(25, "Scaling features...")
            scaler = libs["StandardScaler"]()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            candidates = {
                "random_forest":  RandomForestRegressor(n_estimators=100, random_state=random_seed),
                "gradient_boost": GradientBoostingRegressor(random_state=random_seed),
                "linear":         LinearRegression(),
                "ridge":          Ridge(),
                "lasso":          Lasso(max_iter=5000),
                "elastic_net":    ElasticNet(max_iter=5000),
                "svr":            SVR(),
                "knn":            KNeighborsRegressor(),
                "extra_trees":    ExtraTreesRegressor(n_estimators=100, random_state=random_seed),
            }
            try:
                from xgboost import XGBRegressor
                candidates["xgboost"] = XGBRegressor(random_state=random_seed, verbosity=0)
            except ImportError:
                pass

            if algorithm not in ("auto", ""):
                if algorithm not in candidates:
                    raise MLError("UNKNOWN_ALGO", f"Algorithm '{algorithm}' not found.")
                _p(50, f"Training {algorithm}...")
                model = candidates[algorithm]
                model.fit(X_tr_s, y_tr)
                chosen = algorithm
            else:
                candidates_neg = {
                    name: m for name, m in candidates.items()
                }
                model, chosen, _ = self._train_and_select(candidates_neg, X_tr_s, y_tr, X_te_s, y_te, "regression", _p)

            _p(78, "Evaluating...")
            preds = model.predict(X_te_s)
            mse   = float(libs["mean_squared_error"](y_te, preds))
            rmse  = mse ** 0.5
            mae   = float(np.mean(np.abs(y_te.values - preds)))
            r2    = float(libs["r2_score"](y_te, preds))

            fi = self._extract_feature_importance(model, X.columns)

            _p(93, "Saving model...")
            model_id = f"regressor_{target}_{chosen}_{int(time.time())}"
            artifact = {"model": model, "scaler": scaler, "columns": list(X.columns)}

            meta = {
                "model_id": model_id, "task": "regression",
                "algorithm": chosen, "target": target,
                "features": list(X.columns),
                "feature_types": {c: str(df[c].dtype) for c in X.columns},
                "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4),
                "train_score": round(float(model.score(X_tr_s, y_tr)), 4),
                "feature_importance": fi,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "dataset_shape": list(df.shape),
            }
            self._save_artifact(model_id, artifact, meta)
            _p(100, "Done!")

            return {"success": True, **{k: meta[k] for k in
                ["model_id","task","algorithm","target","features",
                 "rmse","mae","r2","feature_importance","trained_at"]}}

        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Regressor train failed: {e}", exc_info=True)
            return MLError("TRAIN_ERROR", str(e)).to_dict()

    def auto_ml(self, session_id, target, test_size=0.2, cv_folds=5,
                random_seed=42, features=None,
                job_status=None, job_id=None) -> Dict:
        try:
            def _p(pct, step): self._progress(job_status, job_id, pct, step)
            _p(3, "Detecting task type...")
            df = self.data_engine.load_session_df(session_id)
            y  = df[target]
            task = "classification" if (y.dtype == "object" or y.nunique() <= 20) else "regression"
            _p(8, f"AutoML → {task}")
            if task == "classification":
                return self.train_classifier(session_id, target, "auto",
                    test_size, cv_folds, random_seed, features, job_status, job_id)
            else:
                return self.train_regressor(session_id, target, "auto",
                    test_size, cv_folds, random_seed, features, job_status, job_id)
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"AutoML failed: {e}", exc_info=True)
            return MLError("AUTOML_ERROR", str(e)).to_dict()

    def train_cluster(self, session_id, algorithm="kmeans", n_clusters=3,
                      eps=0.5, min_samples=5,
                      job_status=None, job_id=None) -> Dict:
        def _p(pct, step): self._progress(job_status, job_id, pct, step)
        try:
            _p(5, "Loading dataset...")
            libs = self._load()
            np = libs["np"]
            df = self.data_engine.load_session_df(session_id)
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not cols:
                raise MLError("NO_NUMERIC", "No numeric columns for clustering.")
            X = df[cols].dropna()
            _p(25, "Scaling...")
            scaler = libs["StandardScaler"]()
            X_s = scaler.fit_transform(X)
            _p(40, f"Running {algorithm}...")
            if algorithm == "kmeans":
                from sklearn.cluster import KMeans
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=settings.KMEANS_N_INIT)
            elif algorithm == "dbscan":
                from sklearn.cluster import DBSCAN
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif algorithm == "agglomerative":
                from sklearn.cluster import AgglomerativeClustering
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                raise MLError("UNKNOWN_ALGO", f"Unknown: {algorithm}")
            labels = model.fit_predict(X_s)
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            _p(70, "Computing metrics...")
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(X_s, labels)) if n_found > 1 else 0.0
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            xy = pca.fit_transform(X_s).tolist()
            result = {
                "success": True,
                "algorithm": algorithm,
                "n_clusters": n_clusters,
                "n_found": n_found,
                "silhouette": round(sil, 4),
                "labels": [int(l) for l in labels],
                "clusters": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
                "centroids_2d": [{"x": round(c[0], 4), "y": round(c[1], 4)} for c in xy[:n_clusters]] if algorithm == "kmeans" else [],
                "points_2d": [{"x": round(p[0], 4), "y": round(p[1], 4), "cluster": int(labels[i])} for i, p in enumerate(xy)],
            }
            _p(100, "Done!")
            return result
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Cluster failed: {e}", exc_info=True)
            return MLError("CLUSTER_ERROR", str(e)).to_dict()

    def reduce_dim(self, session_id, algorithm="pca", n_components=2,
                   job_status=None, job_id=None) -> Dict:
        def _p(pct, step): self._progress(job_status, job_id, pct, step)
        try:
            _p(10, "Loading...")
            libs = self._load()
            np = libs["np"]
            df = self.data_engine.load_session_df(session_id)
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not cols:
                raise MLError("NO_NUMERIC", "No numeric columns.")
            X = df[cols].dropna()
            _p(30, "Scaling...")
            scaler = libs["StandardScaler"]()
            X_s = scaler.fit_transform(X)
            _p(50, f"Reducing with {algorithm}...")
            if algorithm == "pca":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=min(n_components, min(X_s.shape)))
            elif algorithm == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=min(n_components, 3), random_state=42, perplexity=min(30, len(X_s)-1))
            elif algorithm == "svd":
                from sklearn.decomposition import TruncatedSVD
                reducer = TruncatedSVD(n_components=min(n_components, min(X_s.shape)-1))
            else:
                raise MLError("UNKNOWN_ALGO", f"Unknown: {algorithm}")
            embedded = reducer.fit_transform(X_s)
            explained = getattr(reducer, "explained_variance_ratio_", None)
            _p(90, "Formatting...")
            result = {
                "success": True,
                "algorithm": algorithm,
                "n_components": embedded.shape[1],
                "data": [{"x": round(float(r[0]), 4), "y": round(float(r[1]), 4) if embedded.shape[1] > 1 else 0} for r in embedded],
                "explained_variance": [round(float(v), 4) for v in explained] if explained is not None else None,
            }
            _p(100, "Done!")
            return result
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Dim reduction failed: {e}", exc_info=True)
            return MLError("DIMRED_ERROR", str(e)).to_dict()

    def train_mlp(self, session_id, target, task_type="classification",
                  hidden_layers="100,50", max_iter=500,
                  job_status=None, job_id=None) -> Dict:
        def _p(pct, step): self._progress(job_status, job_id, pct, step)
        try:
            _p(5, "Preparing...")
            X, y, df, libs = self._prepare(session_id, target)
            np = libs["np"]
            _p(15, "Splitting...")
            X_tr, X_te, y_tr, y_te = libs["train_test_split"](X, y, test_size=0.2, random_state=42)
            _p(25, "Scaling...")
            scaler = libs["StandardScaler"]()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            try:
                layers = [int(x.strip()) for x in hidden_layers.split(",") if x.strip()]
            except ValueError:
                raise MLError("INVALID_LAYERS", f"Invalid layer format: '{hidden_layers}'. Use comma-separated integers like '100,50'.")
            _p(40, f"Training MLP layers={layers}...")
            if task_type == "classification":
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(hidden_layer_sizes=layers, max_iter=max_iter, random_state=42, early_stopping=True)
            else:
                from sklearn.neural_network import MLPRegressor
                model = MLPRegressor(hidden_layer_sizes=layers, max_iter=max_iter, random_state=42, early_stopping=True)
            model.fit(X_tr_s, y_tr)
            _p(75, "Evaluating...")
            score = float(model.score(X_te_s, y_te))
            loss_curve = [round(float(l), 6) for l in model.loss_curve_]
            model_id = f"mlp_{target}_{int(time.time())}"
            artifact = {"model": model, "scaler": scaler, "columns": list(X.columns)}
            meta = {
                "model_id": model_id, "task": task_type,
                "algorithm": "mlp", "target": target,
                "layers": layers, "max_iter": max_iter,
                f"{'accuracy' if task_type=='classification' else 'r2'}": round(score, 4),
                "loss_curve": loss_curve,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_artifact(model_id, artifact, meta)
            _p(100, "Done!")
            return {"success": True, **meta}
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"MLP failed: {e}", exc_info=True)
            return MLError("MLP_ERROR", str(e)).to_dict()

    def tune_hyperparams(self, session_id, target, task_type="classification",
                         algorithm="random_forest", param_grid=None, cv=5, scoring="accuracy",
                         job_status=None, job_id=None) -> Dict:
        def _p(pct, step): self._progress(job_status, job_id, pct, step)
        try:
            _p(5, "Loading...")
            X, y, df, libs = self._prepare(session_id, target)
            _p(15, "Splitting...")
            X_tr, X_te, y_tr, y_te = libs["train_test_split"](X, y, test_size=0.2, random_state=42)
            _p(25, "Scaling...")
            scaler = libs["StandardScaler"]()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            _p(35, f"Tuning {algorithm}...")
            base_params = {
                "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
                "gradient_boost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2]},
                "logistic": {"C": [0.1, 1, 10], "max_iter": [500, 1000]},
                "svm": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
            }
            from sklearn.model_selection import GridSearchCV
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC

            if task_type == "classification":
                estimators = {
                    "random_forest": RandomForestClassifier(random_state=42),
                    "gradient_boost": GradientBoostingClassifier(random_state=42),
                    "logistic": LogisticRegression(max_iter=1000),
                    "svm": SVC(),
                }
                scoring = "accuracy"
            else:
                estimators = {
                    "random_forest": RandomForestRegressor(random_state=42),
                    "gradient_boost": GradientBoostingRegressor(random_state=42),
                }
                scoring = "neg_mean_squared_error"
            est = estimators.get(algorithm)
            if est is None:
                raise MLError("UNKNOWN_ALGO", f"Algorithm '{algorithm}' not supported for tuning.")
            grid = param_grid or base_params.get(algorithm, {})
            _p(50, f"Grid search ({cv}-fold)...")
            n_jobs = settings.GRIDSEARCH_N_JOBS
            if n_jobs == -1:
                import os
                n_jobs = min(os.cpu_count() or 2, 4)
            gs = GridSearchCV(est, grid, cv=min(cv, 5), scoring=scoring, n_jobs=n_jobs)
            gs.fit(X_tr_s, y_tr)
            _p(80, "Evaluating best...")
            best = gs.best_estimator_
            test_score = float(best.score(X_te_s, y_te))
            model_id = f"tuned_{algorithm}_{target}_{int(time.time())}"
            artifact = {"model": best, "scaler": scaler, "columns": list(X.columns)}
            meta = {
                "model_id": model_id, "task": task_type,
                "algorithm": f"tuned_{algorithm}", "target": target,
                "best_params": gs.best_params_,
                "best_score": round(float(gs.best_score_), 4),
                "test_score": round(test_score, 4),
                "cv_results": {k: str(v) for k, v in gs.cv_results_.items() if k in ["mean_test_score", "std_test_score", "params"]},
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_artifact(model_id, artifact, meta)
            _p(100, "Done!")
            return {"success": True, **meta}
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Tune failed: {e}", exc_info=True)
            return MLError("TUNE_ERROR", str(e)).to_dict()

    def train_kfold(self, session_id, target, task_type="classification",
                    algorithm="random_forest", n_splits=5,
                    job_status=None, job_id=None) -> Dict:
        def _p(pct, step): self._progress(job_status, job_id, pct, step)
        try:
            _p(5, "Loading...")
            X, y, df, libs = self._prepare(session_id, target)
            np = libs["np"]
            _p(20, f"{n_splits}-fold CV...")
            from sklearn.model_selection import KFold, cross_val_score
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scoring = "accuracy"
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                scoring = "r2"
            scaler = libs["StandardScaler"]()
            X_s = scaler.fit_transform(X)
            _p(40, "Performing cross-validation...")
            scores = cross_val_score(model, X_s, y, cv=kf, scoring=scoring)
            fold_results = [{"fold": i+1, "score": round(float(s), 4)} for i, s in enumerate(scores)]
            _p(70, "Training final model...")
            model.fit(X_s, y)
            model_id = f"kfold_{algorithm}_{target}_{int(time.time())}"
            artifact = {"model": model, "scaler": scaler, "columns": list(X.columns)}
            meta = {
                "model_id": model_id, "task": task_type,
                "algorithm": f"kfold_{algorithm}", "target": target,
                "n_splits": n_splits, "scoring": scoring,
                "scores": [round(float(s), 4) for s in scores],
                "mean_score": round(float(scores.mean()), 4),
                "std_score": round(float(scores.std()), 4),
                "fold_results": fold_results,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_artifact(model_id, artifact, meta)
            _p(100, "Done!")
            return {"success": True, **meta}
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"K-Fold failed: {e}", exc_info=True)
            return MLError("KFOLD_ERROR", str(e)).to_dict()

    def feature_engineering(self, session_id, operations: List[Dict]) -> Dict:
        try:
            libs = self._load()
            np = libs["np"]
            pd = libs["pd"]
            df = self.data_engine.load_session_df(session_id)
            applied = []
            for op in operations:
                t = op.get("type", "")
                if t == "add_ratio":
                    a = op.get("col_a", ""); b = op.get("col_b", "")
                    if a in df.columns and b in df.columns:
                        name = op.get("name", f"{a}_div_{b}")
                        df[name] = df[a] / (df[b].replace(0, np.nan))
                        applied.append(f"Added {name} = {a}/{b}")
                elif t == "add_product":
                    a = op.get("col_a", ""); b = op.get("col_b", "")
                    if a in df.columns and b in df.columns:
                        name = op.get("name", f"{a}_x_{b}")
                        df[name] = df[a] * df[b]
                        applied.append(f"Added {name} = {a}×{b}")
                elif t == "add_sum":
                    a = op.get("col_a", ""); b = op.get("col_b", "")
                    if a in df.columns and b in df.columns:
                        name = op.get("name", f"{a}_plus_{b}")
                        df[name] = df[a] + df[b]
                        applied.append(f"Added {name} = {a}+{b}")
                elif t == "add_poly":
                    col = op.get("column", op.get("col_a", ""))
                    if col in df.columns:
                        deg = op.get("degree", 2); name = op.get("name", f"{col}_sq")
                        df[name] = df[col] ** deg
                        applied.append(f"Added {name} = {col}^{deg}")
                elif t == "binarize":
                    col = op.get("column", op.get("col_a", ""))
                    if col in df.columns:
                        thresh = op.get("threshold", 0)
                        df[f"{col}_bin"] = (df[col] > thresh).astype(int)
                        applied.append(f"Binarized {col} @ {thresh}")
                elif t == "log_transform":
                    col = op.get("column", op.get("col_a", ""))
                    if col in df.columns:
                        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
                        applied.append(f"Log-transformed {col}")
                elif t == "rolling_mean":
                    col = op.get("column", op.get("col_a", ""))
                    if col in df.columns:
                        window = op.get("window", 3)
                        df[f"{col}_rm{window}"] = df[col].rolling(window, min_periods=1).mean()
                        applied.append(f"Rolling mean {col} w={window}")
            cleaned_id = f"{session_id}_fe"
            cleaned_path = self.sessions_dir / f"{cleaned_id}.csv"
            temp_path = cleaned_path.with_suffix(".tmp")
            df.to_csv(temp_path, index=False)
            temp_path.rename(cleaned_path)
            self.data_engine.save_session(cleaned_id, str(cleaned_path))
            return {
                "success": True,
                "message": f"Applied {len(applied)} operation(s): {', '.join(applied)}",
                "cleaned_session_id": cleaned_id,
                "columns": list(df.columns),
                "shape": list(df.shape),
                "applied": applied,
            }
        except MLError as e: return e.to_dict()
        except Exception as e:
            log.error(f"Feat eng failed: {e}", exc_info=True)
            return MLError("FEATENG_ERROR", str(e)).to_dict()
