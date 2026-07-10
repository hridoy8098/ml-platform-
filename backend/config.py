import os
from pathlib import Path
from typing import Optional

class Settings:
    APP_NAME: str = "ML Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    HOST: str = os.getenv("HOST", "0.0.0.0" if os.getenv("DEBUG", "true").lower() == "true" else "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    BASE_DIR: Path = Path(__file__).parent
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    MODELS_DIR: Path = Path.home() / "ml_platform" / "models"
    SESSIONS_DIR: Path = Path.home() / "ml_platform" / "sessions"
    EXPORTS_DIR: Path = Path.home() / "ml_platform" / "exports"

    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
    JOB_POLL_INTERVAL_MS: int = 800
    JOB_CLEANUP_AGE_HOURS: int = 24
    CACHE_MAXSIZE: int = 10
    CACHE_TTL_SECONDS: int = 300

    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://localhost:5173,http://127.0.0.1:8000").split(",")

    API_KEY: Optional[str] = os.getenv("API_KEY")
    API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")

    FRONTEND_REACT_DIR: Path = BASE_DIR.parent / "frontend-react" / "dist"
    FRONTEND_VANILLA_DIR: Path = BASE_DIR.parent / "frontend"

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    GRIDSEARCH_N_JOBS: int = int(os.getenv("GRIDSEARCH_N_JOBS", "-1"))
    KMEANS_N_INIT: int = int(os.getenv("KMEANS_N_INIT", "10"))
    ELBOW_MAX_K: int = int(os.getenv("ELBOW_MAX_K", "20"))
    SCATTER_MAX_POINTS: int = int(os.getenv("SCATTER_MAX_POINTS", "1000"))
    CORRELATION_MAX_COLS: int = int(os.getenv("CORRELATION_MAX_COLS", "15"))

    CV_UPLOADS_DIR: Path = BASE_DIR / "cv_uploads"
    CV_MODELS_DIR: Path = Path.home() / "ml_platform" / "cv_models"
    CV_MAX_IMAGE_DIM: int = int(os.getenv("CV_MAX_IMAGE_DIM", "512"))
    CV_DEFAULT_EPOCHS: int = int(os.getenv("CV_DEFAULT_EPOCHS", "10"))
    CV_BATCH_SIZE: int = int(os.getenv("CV_BATCH_SIZE", "16"))
    CV_LEARNING_RATE: float = float(os.getenv("CV_LEARNING_RATE", "0.001"))
    CV_MAX_UPLOAD_MB: int = int(os.getenv("CV_MAX_UPLOAD_MB", "500"))

    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


settings = Settings()
