"""
API Endpoint Router
Owner: Shivam (Backend & APIs)

Defines thin FastAPI route handlers that delegate file storage, validation,
and AI pipeline orchestration to the service layer.
"""

import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.config import settings
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
from ai.config import AIConfig

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
                else f"{AIConfig.VISION_MODEL_NAME} ({'ready' if real_status['vision_ready'] else 'missing weights'})"
            ),
            "explainability": (
                "disabled in mock mode (no fake heatmaps)"
                if ai_mode == "mock"
                else ("ready" if real_status["vision_ready"] else "unavailable")
            ),
            "rag_vector_store": (
                "local open-access guideline corpus (mock)"
                if ai_mode == "mock"
                else f"ChromaDB ({'ready' if real_status['rag_ready'] else 'uninitialized'})"
            ),
            "llm_synthesizer": (
                "template report synthesizer (mock)"
                if ai_mode == "mock"
                else f"{AIConfig.LLM_MODEL} ({'ready' if real_status['llm_ready'] else 'missing API key'})"
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
