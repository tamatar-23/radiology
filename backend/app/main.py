"""
FastAPI Main Application Entrypoint
Owner: Shivam (Backend & Deployment)
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.api.endpoints import router as api_router
from backend.app.db.database import init_db
from ai.vision.model import ensure_default_vision_weights
from ai.rag.vector_store import MedicalVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = settings.get_ai_mode()
    logger.info(
        "Starting %s (AI_MODE=%s)",
        settings.PROJECT_NAME,
        mode,
    )
    init_db()
    if mode == "real":
        try:
            weights_file = ensure_default_vision_weights()
            vstore = MedicalVectorStore()
            num_chunks = vstore.ensure_indexed()
            logger.info(
                "Real mode initialized: weights='%s', ChromaDB chunks=%d",
                weights_file,
                num_chunks,
            )
        except Exception as exc:
            logger.warning("Real mode startup initialization warning: %s", exc)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Research prototype backend API for Chest X-Ray image preprocessing, "
        "multi-label abnormality classification, Grad-CAM explainability, and Agentic RAG reporting."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
