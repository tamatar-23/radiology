"""
Backend Application Configuration
Owner: Shivam (Backend & Deployment)

Centralizes environment settings for FastAPI, SQLAlchemy database connections,
local file uploads, and AI execution mode.
"""

import os
from pathlib import Path
from typing import List, Set
from dotenv import load_dotenv

# Load .env file if present in repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)


class BackendSettings:
    PROJECT_NAME: str = "Multimodal Agentic Radiology Assistant API"
    VERSION: str = "0.2.0"

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Default to SQLite ONLY when DATABASE_URL is not explicitly set
    DEFAULT_SQLITE_URL: str = f"sqlite:///{(REPO_ROOT / 'radiology_dev.db').as_posix()}"

    @classmethod
    def get_database_url(cls) -> str:
        raw_url = os.getenv("DATABASE_URL", "").strip()
        if not raw_url:
            return cls.DEFAULT_SQLITE_URL
        return raw_url

    @classmethod
    def get_ai_mode(cls) -> str:
        mode = os.getenv("AI_MODE", "mock").strip().lower()
        if mode not in ("mock", "real"):
            raise ValueError(f"Invalid AI_MODE '{mode}'. Must be 'mock' or 'real'.")
        return mode

    @classmethod
    def get_upload_dir(cls) -> Path:
        custom_dir = os.getenv("UPLOAD_DIR", "").strip()
        if custom_dir:
            path = Path(custom_dir)
            if not path.is_absolute():
                path = REPO_ROOT / path
        else:
            path = REPO_ROOT / "data" / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    ALLOWED_MEDIA_TYPES: Set[str] = {
        "image/png",
        "image/jpeg",
        "image/jpg",
    }

    @classmethod
    def get_cors_origins(cls) -> List[str]:
        cors_str = os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        )
        return [origin.strip() for origin in cors_str.split(",") if origin.strip()]


settings = BackendSettings()
