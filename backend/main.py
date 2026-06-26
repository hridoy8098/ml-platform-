import uuid, logging, hashlib, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any, Dict

from config import settings
from auth import APIKeyMiddleware

logging.basicConfig(level=settings.LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("ml-platform")

UPLOADS_DIR  = settings.UPLOADS_DIR
MODELS_DIR   = settings.MODELS_DIR
SESSIONS_DIR = settings.SESSIONS_DIR
EXPORTS_DIR  = settings.EXPORTS_DIR

for d in [UPLOADS_DIR, MODELS_DIR, SESSIONS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_ml_libs = None

def load_ml_libs():
    global _ml_libs
    if _ml_libs is None:
        import pandas as pd, numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        import joblib
        _ml_libs = {"pd": pd, "np": np, "train_test_split": train_test_split,
                    "StandardScaler": StandardScaler, "LabelEncoder": LabelEncoder,
                    "accuracy_score": accuracy_score, "mean_squared_error": mean_squared_error,
                    "r2_score": r2_score, "joblib": joblib}
    return _ml_libs

active_models: dict = {}
active_models_lock = Lock()
job_status: dict = {}
job_status_lock = Lock()
job_executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_JOBS)

from agents.ml.data_engine   import DataEngine, MLError
from agents.ml.model_factory  import ModelFactory
from agents.ml.evaluator      import Evaluator, Predictor, Deployer
from agents.ml.visualizer     import Visualizer

data_engine   = DataEngine(load_ml_libs, SESSIONS_DIR)
model_factory = ModelFactory(MODELS_DIR, SESSIONS_DIR, load_ml_libs, active_models, data_engine)
evaluator     = Evaluator(MODELS_DIR, active_models, load_ml_libs, model_factory, data_engine)
predictor     = Predictor(MODELS_DIR, active_models, load_ml_libs, model_factory)
deployer      = Deployer(MODELS_DIR, active_models, model_factory)
visualizer    = Visualizer(MODELS_DIR, load_ml_libs)

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.add_middleware(CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True)

if settings.API_KEY:
    app.add_middleware(APIKeyMiddleware)

REACT_DIR = settings.FRONTEND_REACT_DIR
VANILLA_DIR = settings.FRONTEND_VANILLA_DIR
FRONTEND_DIR = REACT_DIR if REACT_DIR.exists() else VANILLA_DIR

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def root():
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"error": "Frontend not found"}

def compute_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def error_response(msg: str, code: str = "INTERNAL_ERROR") -> Dict:
    log.warning(f"{code}: {msg}")
    return {"success": False, "error": code, "message": "An unexpected error occurred. Check logs for details."}

def start_job(fn, *args, **kwargs) -> str:
    jid = str(uuid.uuid4())[:8]
    with job_status_lock:
        job_status[jid] = {"status": "running", "progress": 0, "step": "Starting...", "result": None}

    def _run():
        try:
            result = fn(*args, job_status=job_status, job_id=jid, **kwargs)
            with job_status_lock:
                if jid in job_status:
                    job_status[jid].update({"status": "done", "progress": 100, "result": result})
        except Exception as e:
            log.error(f"Job {jid} failed: {e}", exc_info=True)
            with job_status_lock:
                if jid in job_status:
                    job_status[jid].update({"status": "error", "result": {"success": False, "message": str(e)}})

    future = job_executor.submit(_run)
    future.add_done_callback(lambda f: _cleanup_old_jobs())
    return jid

def _cleanup_old_jobs():
    cutoff = time.time() - settings.JOB_CLEANUP_AGE_HOURS * 3600
    with job_status_lock:
        stale = [k for k, v in job_status.items()
                 if v.get("status") in ("done", "error") and v.get("_ts", 0) < cutoff]
        for k in stale:
            del job_status[k]

def _safe_result(jid, result):
    with job_status_lock:
        if jid in job_status:
            job_status[jid]["_ts"] = time.time()
    return result

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".feather"}

ALLOWED_MIME_PREFIXES = ["text/", "application/json", "application/vnd.openxmlformats",
                         "application/vnd.ms-excel", "application/octet-stream"]

class BaseReq(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class CleanReq(BaseReq):
    session_id: str
    operations: Optional[List[Dict]] = []

class AnalyzeReq(BaseReq):
    session_id: str

class TrainReq(BaseReq):
    session_id: str
    task:       str = "classification"
    target:     Optional[str] = None
    algorithm:  str = "auto"
    test_size:  float = 0.2
    cv_folds:   int = 5
    random_seed:int = 42
    features:   Optional[List[str]] = None

class EvalReq(BaseReq):
    session_id: str
    model_id:   str

class PredictReq(BaseReq):
    model_id: str
    input:    Dict[str, Any]

class DeleteReq(BaseReq):
    model_id: str

class SampleReq(BaseReq):
    name: str

class ClusterReq(BaseReq):
    session_id:   str
    algorithm:    str = "kmeans"
    n_clusters:   int = 3
    eps:          float = 0.5
    min_samples:  int = 5

class DimRedReq(BaseReq):
    session_id:    str
    algorithm:     str = "pca"
    n_components:  int = 2

class NeuralReq(BaseReq):
    session_id:    str
    target:        str
    task_type:     str = "classification"
    hidden_layers: str = "100,50"
    max_iter:      int = 500

class TuneReq(BaseReq):
    session_id:  str
    target:      str
    task_type:   str = "classification"
    algorithm:   str = "random_forest"
    cv:          int = 5
    scoring:     str = "accuracy"

class KFoldReq(BaseReq):
    session_id: str
    target:     str
    task_type:  str = "classification"
    algorithm:  str = "random_forest"
    n_splits:   int = 5

class FeatEngReq(BaseReq):
    session_id: str
    operations: List[Dict] = []

class ChartReq(BaseReq):
    session_id: str
    plot_type:  str = "histogram"
    columns:    Optional[List[str]] = None

class EvalAdvancedReq(BaseReq):
    session_id: str
    model_id:   str

class ImportanceReq(BaseReq):
    model_id: str

class ExportReq(BaseReq):
    model_id: str
    format:   str = "joblib"

class ElbowReq(BaseReq):
    session_id: str
    max_k: int = 10

@app.post("/ml/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        ext = Path(file.filename).suffix.lower() if file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return {"success": False, "error": "INVALID_FORMAT",
                    "message": f"Format '{ext}' not allowed. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}

        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
            return {"success": False, "error": "FILE_TOO_LARGE",
                    "message": f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit"}

        session_id = str(uuid.uuid4())[:12]
        safe_name = Path(file.filename).name.replace("..", "").replace("/", "").replace("\\", "")
        filepath = UPLOADS_DIR / f"{session_id}{ext}"
        filepath.write_bytes(content)
        log.info(f"Uploaded {safe_name} ({len(content):,} bytes) → {filepath.name}")
        return data_engine.load_dataset(str(filepath), session_id)
    except Exception as e:
        log.error(f"Upload failed: {e}", exc_info=True)
        return error_response(str(e), "UPLOAD_ERROR")

@app.post("/ml/sample")
async def load_sample(req: SampleReq):
    session_id = str(uuid.uuid4())[:12]
    return data_engine.load_sample(req.name, session_id, UPLOADS_DIR)

@app.post("/ml/clean")
async def clean_data(req: CleanReq):
    return data_engine.clean_data(req.session_id, req.operations)

@app.post("/ml/analyze")
async def analyze_data(req: AnalyzeReq):
    return data_engine.analyze_data(req.session_id)

@app.post("/ml/train")
async def train_model(req: TrainReq):
    try:
        if req.test_size < 0.05 or req.test_size > 0.5:
            return {"success": False, "error": "INVALID_PARAM",
                    "message": "test_size must be between 0.05 and 0.5"}
        kwargs = dict(
            session_id=req.session_id, target=req.target,
            algorithm=req.algorithm, test_size=req.test_size,
            cv_folds=req.cv_folds, random_seed=req.random_seed,
            features=req.features,
        )
        if req.task == "classification":
            jid = start_job(model_factory.train_classifier, **kwargs)
        elif req.task == "regression":
            jid = start_job(model_factory.train_regressor, **kwargs)
        else:
            jid = start_job(model_factory.auto_ml, **{k: v for k, v in kwargs.items() if k != "algorithm"})
        return {"success": True, "job_id": jid, "message": f"{req.task} training started"}
    except Exception as e:
        log.error(f"Train start failed: {e}", exc_info=True)
        return error_response(str(e), "TRAIN_START_ERROR")

@app.get("/ml/job/{job_id}")
async def get_job(job_id: str):
    with job_status_lock:
        if job_id not in job_status:
            return {"success": False, "message": f"Job '{job_id}' not found"}
        return {"success": True, "job_id": job_id, **job_status[job_id]}

@app.post("/ml/evaluate")
async def evaluate_model(req: EvalReq):
    return evaluator.evaluate(req.session_id, req.model_id)

@app.post("/ml/predict")
async def predict(req: PredictReq):
    return predictor.predict(req.model_id, req.input)

@app.get("/ml/model-info/{model_id}")
async def model_info(model_id: str):
    return predictor.get_model_info(model_id)

@app.get("/ml/models")
async def list_models():
    return deployer.list_models()

@app.post("/ml/delete-model")
async def delete_model(req: DeleteReq):
    return deployer.delete_model(req.model_id)

@app.post("/ml/cluster")
async def cluster_data(req: ClusterReq):
    try:
        jid = start_job(model_factory.train_cluster,
            req.session_id, req.algorithm, req.n_clusters, req.eps, req.min_samples)
        return {"success": True, "job_id": jid, "message": "Clustering started"}
    except Exception as e:
        log.error(f"Cluster failed: {e}", exc_info=True)
        return error_response(str(e), "CLUSTER_ERROR")

@app.post("/ml/dim-reduce")
async def dim_reduce(req: DimRedReq):
    try:
        jid = start_job(model_factory.reduce_dim,
            req.session_id, req.algorithm, req.n_components)
        return {"success": True, "job_id": jid, "message": "Dim reduction started"}
    except Exception as e:
        log.error(f"Dim reduce failed: {e}", exc_info=True)
        return error_response(str(e), "DIMRED_ERROR")

@app.post("/ml/neural")
async def train_neural(req: NeuralReq):
    try:
        jid = start_job(model_factory.train_mlp,
            req.session_id, req.target, req.task_type, req.hidden_layers, req.max_iter)
        return {"success": True, "job_id": jid, "message": "Neural training started"}
    except Exception as e:
        log.error(f"Neural failed: {e}", exc_info=True)
        return error_response(str(e), "NEURAL_ERROR")

@app.post("/ml/tune")
async def tune_hyperparams(req: TuneReq):
    try:
        jid = start_job(model_factory.tune_hyperparams,
            req.session_id, req.target, req.task_type, req.algorithm, None, req.cv, req.scoring)
        return {"success": True, "job_id": jid, "message": "Hyperparameter tuning started"}
    except Exception as e:
        log.error(f"Tune failed: {e}", exc_info=True)
        return error_response(str(e), "TUNE_ERROR")

@app.post("/ml/kfold")
async def kfold_cv(req: KFoldReq):
    try:
        jid = start_job(model_factory.train_kfold,
            req.session_id, req.target, req.task_type, req.algorithm, req.n_splits)
        return {"success": True, "job_id": jid, "message": "K-Fold CV started"}
    except Exception as e:
        log.error(f"K-Fold failed: {e}", exc_info=True)
        return error_response(str(e), "KFOLD_ERROR")

@app.post("/ml/feature-engineer")
async def feature_engineer(req: FeatEngReq):
    try:
        return model_factory.feature_engineering(req.session_id, req.operations)
    except Exception as e:
        log.error(f"Feat eng failed: {e}", exc_info=True)
        return error_response(str(e), "FEATENG_ERROR")

@app.post("/ml/chart-data")
async def chart_data(req: ChartReq):
    try:
        return visualizer.get_chart_data(req.session_id, data_engine, req.plot_type, req.columns)
    except Exception as e:
        log.error(f"Chart data failed: {e}", exc_info=True)
        return error_response(str(e), "CHART_ERROR")

@app.post("/ml/confusion-matrix")
async def confusion_matrix(req: EvalAdvancedReq):
    try:
        return visualizer.confusion_matrix_data(req.model_id, req.session_id, data_engine)
    except Exception as e:
        log.error(f"CM failed: {e}", exc_info=True)
        return error_response(str(e), "CM_ERROR")

@app.post("/ml/roc-curve")
async def roc_curve(req: EvalAdvancedReq):
    try:
        return visualizer.roc_curve_data(req.model_id, req.session_id, data_engine)
    except Exception as e:
        log.error(f"ROC failed: {e}", exc_info=True)
        return error_response(str(e), "ROC_ERROR")

@app.post("/ml/elbow-plot")
async def elbow_plot(req: ElbowReq):
    try:
        max_k = min(max(req.max_k, 2), settings.ELBOW_MAX_K)
        return visualizer.elbow_plot_data(req.session_id, data_engine, max_k)
    except Exception as e:
        log.error(f"Elbow failed: {e}", exc_info=True)
        return error_response(str(e), "ELBOW_ERROR")

@app.post("/ml/feature-importance")
async def feature_importance(req: ImportanceReq):
    try:
        return visualizer.feature_importance(req.model_id, active_models, model_factory)
    except Exception as e:
        log.error(f"Feature importance failed: {e}", exc_info=True)
        return error_response(str(e), "IMPORTANCE_ERROR")

@app.post("/ml/export-model")
async def export_model(req: ExportReq):
    return deployer.export_model(req.model_id, req.format)

@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION, "auth": "enabled" if settings.API_KEY else "disabled"}
