"""
API Endpoint Router
Owner: Shivam (Backend & APIs)

Defines thin FastAPI route handlers that delegate file storage, validation,
and AI pipeline orchestration to the service layer.
"""

import json
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.config import settings, REPO_ROOT
from backend.app.db.database import get_db, check_db_connection
from backend.app.schemas.radiology import (
    HealthCheckResponse,
    ScanUploadResponse,
    AnalysisRequest,
    AnalysisResponse,
    ReportResponse,
)
from backend.app.services.storage_service import StorageService
from backend.app.services.analysis_service import AnalysisService
from ai.config import AIConfig, resolve_repo_path
from ai.rag.vector_store import MedicalVectorStore

router = APIRouter()
storage_service = StorageService()
analysis_service = AnalysisService()


@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def get_health():
    """
    Health check endpoint reporting backend status, database connectivity,
    current AI execution mode (mock vs real), and integration readiness.
    """
    db_ok, db_status = check_db_connection()
    try:
        ai_mode = settings.get_ai_mode()
    except ValueError:
        ai_mode = "mock"

    real_status = AIConfig.validate_real_mode_status()

    acc_badge = ""
    metrics_file = resolve_repo_path(AIConfig.EVALUATION_METRICS_PATH)
    if ai_mode == "real" and metrics_file and os.path.isfile(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            bal_acc = m_data.get("classification_metrics", {}).get("balanced_accuracy")
            if bal_acc:
                acc_badge = f", {bal_acc * 100:.1f}% benchmark acc"
        except Exception:
            acc_badge = ""

    chroma_count = 0
    if ai_mode == "real" and real_status["rag_ready"]:
        try:
            chroma_count = MedicalVectorStore().count()
        except Exception:
            chroma_count = 0

    return HealthCheckResponse(
        status="healthy" if db_ok else "degraded",
        service="radiology-backend",
        database_status=db_status,
        ai_mode="real" if ai_mode == "real" else "mock",
        modality_scope="Chest X-Ray (DX)",
        integrations={
            "vision_model": (
                "demo-cxr-densenet121-mock (simulated)"
                if ai_mode == "mock"
                else f"{AIConfig.VISION_MODEL_NAME} ({'ready' + acc_badge if real_status['vision_ready'] else 'missing weights'})"
            ),
            "explainability": (
                "disabled in mock mode (no fake heatmaps)"
                if ai_mode == "mock"
                else ("Grad-CAM spatial activation overlay (ready)" if real_status["vision_ready"] else "unavailable")
            ),
            "rag_vector_store": (
                "local open-access guideline corpus (mock)"
                if ai_mode == "mock"
                else (
                    f"ChromaDB ({chroma_count} chunks indexed)"
                    if real_status["rag_ready"]
                    else "ChromaDB (uninitialized)"
                )
            ),
            "llm_synthesizer": (
                "template report synthesizer (mock)"
                if ai_mode == "mock"
                else (
                    str(real_status.get("llm_backend", AIConfig.LLM_MODEL))
                    if real_status["llm_ready"]
                    else f"{AIConfig.LLM_MODEL} (missing API key)"
                )
            ),
        },
    )


@router.post("/api/upload", response_model=ScanUploadResponse, tags=["Radiology API"])
async def upload_scan(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Validate and store an uploaded Chest X-Ray image (PNG or JPEG) and persist scan metadata.
    """
    return await storage_service.save_upload(file=file, db=db)


@router.post("/api/analyze", response_model=AnalysisResponse, tags=["Radiology API"])
async def analyze_scan(
    request: AnalysisRequest,
    db: Session = Depends(get_db),
):
    """
    Trigger the Chest X-Ray preprocessing, multi-label classification, Grad-CAM check,
    and 5-stage Agentic RAG pipeline for an uploaded scan.
    """
    return analysis_service.run_analysis(request=request, db=db)


@router.get("/api/report/{report_id}", response_model=ReportResponse, tags=["Radiology API"])
async def get_report(
    report_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieve a persisted radiology analysis report by its report_id.
    """
    return analysis_service.get_report_by_id(report_id=report_id, db=db)


@router.get("/api/scans/{scan_id}/image", tags=["Radiology API"])
async def get_scan_image(
    scan_id: str,
    db: Session = Depends(get_db),
):
    """
    Serve the persisted Chest X-Ray image file for display in the viewer.
    """
    scan = storage_service.get_scan_or_404(scan_id=scan_id, db=db)
    if not os.path.isfile(scan.file_path):
        raise HTTPException(status_code=404, detail="Scan image file is missing on disk.")
    return FileResponse(path=scan.file_path, media_type=scan.file_type, filename=scan.filename)


@router.get("/api/scans/{scan_id}/heatmap", tags=["Radiology API"])
async def get_scan_heatmap(
    scan_id: str,
    db: Session = Depends(get_db),
):
    """
    Serve the generated Grad-CAM spatial activation heatmap PNG for a scan analyzed in real mode.
    """
    scan = storage_service.get_scan_or_404(scan_id=scan_id, db=db)
    safe_scan_id = "".join(c for c in scan.id if c.isalnum() or c in ("-", "_")) or "scan"
    upload_dir = settings.get_upload_dir()
    heatmap_path = upload_dir / f"gradcam_{safe_scan_id}.png"

    if not heatmap_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Grad-CAM heatmap is not available for scan '{scan_id}'.",
        )
    return FileResponse(
        path=str(heatmap_path),
        media_type="image/png",
        filename=f"gradcam_{safe_scan_id}.png",
    )


@router.get("/api/samples/cxr", tags=["Radiology API"])
async def get_sample_cxr():
    """
    Serve the default Chest X-Ray image (`data/samples/dataset-card.jpeg`) so the
    frontend loads it by default.
    """
    default_jpeg = REPO_ROOT / "data" / "samples" / "dataset-card.jpeg"
    if default_jpeg.is_file():
        return FileResponse(
            path=str(default_jpeg),
            media_type="image/jpeg",
            filename="dataset-card.jpeg",
        )
    sample_path = REPO_ROOT / "data" / "samples" / "sample_cxr.png"
    if not sample_path.is_file():
        from scripts.seed_sample_data import generate_synthetic_cxr_phantom

        generate_synthetic_cxr_phantom(sample_path)
    return FileResponse(
        path=str(sample_path),
        media_type="image/png",
        filename="sample_cxr.png",
    )
