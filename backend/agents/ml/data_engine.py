import json, os, uuid, time, logging
from typing import Dict, Callable, List, Optional
from pathlib import Path
from functools import lru_cache

from config import settings

log = logging.getLogger("ml-platform.data_engine")

class MLError(Exception):
    def __init__(self, code: str, message: str, suggestion: str = ""):
        self.code = code
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)

    def to_dict(self):
        return {"success": False, "error_code": self.code,
                "message": self.message, "suggestion": self.suggestion}

class SessionCache:
    def __init__(self, maxsize: int = 10, ttl: int = 300):
        self._cache = {}
        self._timestamps = {}
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str):
        if key in self._cache:
            if time.time() - self._timestamps[key] < self._ttl:
                return self._cache[key]
            del self._cache[key]
            del self._timestamps[key]
        return None

    def set(self, key: str, value):
        if len(self._cache) >= self._maxsize:
            oldest = min(self._timestamps, key=self._timestamps.get)
            del self._cache[oldest]
            del self._timestamps[oldest]
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def invalidate(self, key: str):
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)


class DataEngine:
    def __init__(self, ml_libs_loader: Callable, sessions_dir: Path):
        self._load = ml_libs_loader
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache = SessionCache(maxsize=settings.CACHE_MAXSIZE, ttl=settings.CACHE_TTL_SECONDS)

    def save_session(self, session_id: str, filepath: str):
        with open(self.sessions_dir / f"{session_id}.json", "w") as f:
            json.dump({"filepath": filepath}, f)

    def load_session_df(self, session_id: str):
        cached = self._cache.get(session_id)
        if cached is not None:
            return cached

        meta_path = self.sessions_dir / f"{session_id}.json"
        if not meta_path.exists():
            raise MLError("SESSION_NOT_FOUND", f"Session '{session_id}' not found.",
                          "Upload your dataset again from the Data tab.")
        with open(meta_path) as f:
            meta = json.load(f)
        filepath = meta["filepath"]
        if not os.path.exists(filepath):
            raise MLError("FILE_MISSING", "Dataset file no longer exists.",
                          "Upload your dataset again from the Data tab.")
        df = self._read_file(filepath)
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        self._cache.set(session_id, df)
        return df

    def _read_file(self, filepath: str):
        libs = self._load()
        pd   = libs["pd"]
        ext  = filepath.lower().rsplit(".", 1)[-1]
        readers = {"csv": pd.read_csv, "json": pd.read_json,
                   "xlsx": pd.read_excel, "xls": pd.read_excel,
                   "parquet": pd.read_parquet, "feather": pd.read_feather}
        reader = readers.get(ext)
        if not reader:
            raise MLError("UNSUPPORTED_FORMAT", f"Format '.{ext}' not supported.",
                          "Use CSV, Excel, JSON, or Parquet.")
        log.debug(f"Reading file: {filepath}")
        return reader(filepath)

    def load_dataset(self, filepath: str, session_id: str) -> Dict:
        try:
            libs = self._load()
            np   = libs["np"]

            if not os.path.exists(filepath):
                raise MLError("FILE_NOT_FOUND", f"File not found: {os.path.basename(filepath)}")

            df = self._read_file(filepath)
            unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
            if df.empty:
                raise MLError("EMPTY_DATASET", "The file contains no data.")

            self.save_session(session_id, filepath)
            self._cache.set(session_id, df)

            missing  = {c: int(df[c].isna().sum()) for c in df.columns}
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            preview  = json.loads(df.head(200).to_json(orient="records"))

            return {
                "success": True,
                "session_id": session_id,
                "message": f"Loaded {os.path.basename(filepath)} — {len(df):,} rows, {len(df.columns)} columns",
                "preview": preview,
                "columns": list(df.columns),
                "dtypes": {c: str(d) for c, d in df.dtypes.items()},
                "shape": [len(df), len(df.columns)],
                "filename": os.path.basename(filepath),
                "missing": missing,
                "numeric_columns": num_cols,
                "categorical_columns": cat_cols,
            }
        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"Load dataset failed: {e}", exc_info=True)
            return MLError("LOAD_ERROR", f"Failed to load: {str(e)}").to_dict()

    def load_sample(self, name: str, session_id: str, uploads_dir: Path) -> Dict:
        try:
            libs = self._load()
            pd   = libs["pd"]
            from sklearn import datasets as sk_datasets

            SAMPLES = {
                "iris":          lambda: sk_datasets.load_iris(as_frame=True).frame,
                "titanic":       lambda: pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"),
                "boston housing":lambda: sk_datasets.fetch_california_housing(as_frame=True).frame,
                "diabetes":      lambda: sk_datasets.load_diabetes(as_frame=True).frame,
                "wine quality":  lambda: pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";"),
            }

            loader = SAMPLES.get(name.lower())
            if not loader:
                raise MLError("UNKNOWN_SAMPLE", f"Sample '{name}' not found.",
                              f"Available: {list(SAMPLES.keys())}")

            df = loader()
            path = uploads_dir / f"sample_{name.replace(' ','_')}_{session_id}.csv"
            df.to_csv(path, index=False)
            log.info(f"Loaded sample dataset: {name}")
            return self.load_dataset(str(path), session_id)
        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"Sample load failed: {e}", exc_info=True)
            return MLError("SAMPLE_ERROR", f"Could not load sample: {str(e)}").to_dict()

    def clean_data(self, session_id: str, operations: List[Dict] = None) -> Dict:
        try:
            libs = self._load()
            pd   = libs["pd"]
            np   = libs["np"]

            df = self.load_session_df(session_id)
            original_shape = df.shape
            applied = []

            for op in (operations or []):
                t = op.get("type", "")

                if t == "drop_na_all":
                    before = len(df); df = df.dropna()
                    applied.append(f"Dropped {before - len(df)} rows with any missing value")

                elif t == "drop_duplicates":
                    before = len(df); df = df.drop_duplicates()
                    applied.append(f"Removed {before - len(df)} duplicate rows")

                elif t == "reset_index":
                    df = df.reset_index(drop=True)
                    applied.append("Reset index")

                elif t == "drop_na_col":
                    col = op.get("column")
                    if col and col in df.columns:
                        before = len(df); df = df.dropna(subset=[col])
                        applied.append(f"Dropped {before - len(df)} rows missing '{col}'")

                elif t == "fill_na":
                    col      = op.get("column")
                    strategy = op.get("strategy")
                    value    = op.get("value")
                    if col and col in df.columns:
                        if strategy == "mean":
                            df[col] = df[col].fillna(df[col].mean())
                        elif strategy == "median":
                            df[col] = df[col].fillna(df[col].median())
                        elif strategy == "mode":
                            df[col] = df[col].fillna(df[col].mode()[0])
                        elif value is not None:
                            df[col] = df[col].fillna(value)
                        applied.append(f"Filled missing in '{col}' with {strategy or value}")

                elif t == "drop_column":
                    col = op.get("column")
                    if col and col in df.columns:
                        df = df.drop(columns=[col])
                        applied.append(f"Dropped column '{col}'")

                elif t == "encode_label":
                    col = op.get("column")
                    if col and col in df.columns:
                        df[col] = pd.Categorical(df[col]).codes
                        applied.append(f"Label encoded '{col}'")

                elif t == "encode_onehot":
                    col = op.get("column")
                    if col and col in df.columns:
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                        applied.append(f"One-Hot encoded '{col}'")

                elif t == "encode_freq":
                    col = op.get("column")
                    if col and col in df.columns:
                        freq = df[col].value_counts() / len(df)
                        df[col] = df[col].map(freq)
                        applied.append(f"Frequency encoded '{col}'")

                elif t == "normalize":
                    col = op.get("column")
                    if col and col in df.columns:
                        mn, mx = df[col].min(), df[col].max()
                        df[col] = (df[col] - mn) / (mx - mn + 1e-8)
                        applied.append(f"Normalized '{col}' to [0,1]")

                elif t == "standardize":
                    col = op.get("column")
                    if col and col in df.columns:
                        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
                        applied.append(f"Standardized '{col}'")

                elif t == "log_transform":
                    col = op.get("column")
                    if col and col in df.columns:
                        df[col] = np.log1p(df[col].clip(lower=0))
                        applied.append(f"Log-transformed '{col}'")

                elif t == "cap_outliers":
                    col = op.get("column")
                    if col and col in df.columns:
                        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                        applied.append(f"Capped outliers in '{col}'")

            cleaned_id   = f"{session_id}_cleaned"
            cleaned_path = self.sessions_dir / f"{cleaned_id}.csv"
            temp_path = cleaned_path.with_suffix(".tmp")
            df.to_csv(temp_path, index=False)
            temp_path.rename(cleaned_path)
            self.save_session(cleaned_id, str(cleaned_path))
            self._cache.invalidate(session_id)
            self._cache.set(cleaned_id, df)

            missing = {c: int(df[c].isna().sum()) for c in df.columns}
            preview = json.loads(df.head(200).to_json(orient="records"))

            return {
                "success": True,
                "message": f"Applied {len(applied)} operation(s) — {len(df):,} rows, {len(df.columns)} cols (was {original_shape[0]:,}×{original_shape[1]})",
                "cleaned_session_id": cleaned_id,
                "preview": preview,
                "columns": list(df.columns),
                "dtypes": {c: str(d) for c, d in df.dtypes.items()},
                "shape": [len(df), len(df.columns)],
                "missing": missing,
                "operations_applied": applied,
            }
        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"Clean failed: {e}", exc_info=True)
            return MLError("CLEAN_ERROR", f"Cleaning failed: {str(e)}").to_dict()

    def analyze_data(self, session_id: str) -> Dict:
        try:
            libs = self._load()
            np   = libs["np"]

            df       = self.load_session_df(session_id)
            num_df   = df.select_dtypes(include=[np.number])
            cat_df   = df.select_dtypes(include=["object", "category"])

            basic_stats = {}
            skewness    = {}
            outliers    = {}
            correlations = {}
            histogram_data = {}

            if not num_df.empty:
                desc = num_df.describe()
                basic_stats = json.loads(desc.to_json(orient="index"))

                for col in num_df.columns:
                    s = num_df[col].dropna()
                    if len(s) < 2:
                        continue
                    skewness[col] = round(float(s.skew()), 4)
                    Q1, Q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                    IQR = Q3 - Q1
                    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    n_out = int(((s < lo) | (s > hi)).sum())
                    if n_out:
                        outliers[col] = {"count": n_out, "lower": round(lo, 4), "upper": round(hi, 4)}
                    counts, edges = np.histogram(s.dropna(), bins=20)
                    histogram_data[col] = {"counts": counts.tolist(),
                                           "bins": [round(float(e), 4) for e in edges]}

                corr_cols    = num_df.columns[:12]
                correlations = json.loads(num_df[corr_cols].corr().to_json(orient="index"))

            analysis = {
                "basic_stats": basic_stats, "skewness": skewness,
                "outliers": outliers, "correlations": correlations,
                "histogram_data": histogram_data,
            }

            if not cat_df.empty:
                analysis["value_counts"] = {
                    c: {str(k): int(v) for k, v in df[c].value_counts().head(20).items()}
                    for c in cat_df.columns
                }

            return {
                "success": True,
                "message": f"Analysis complete — {len(df):,} rows, {len(df.columns)} columns",
                "analysis": analysis,
            }
        except MLError as e:
            return e.to_dict()
        except Exception as e:
            log.error(f"Analysis failed: {e}", exc_info=True)
            return MLError("ANALYSIS_ERROR", f"Analysis failed: {str(e)}").to_dict()
