"""
API Endpoint Router
Owner: Shivam (Backend & APIs)

Defines FastAPI routing endpoints for image upload, scan analysis, and radiology report retrieval.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Optional
import uuid
import datetime

from backend.app.schemas.radiology import (
    HealthCheckResponse,
    ScanUploadResponse,
    AnalysisRequest,
    AnalysisResponse,
    EvidenceItem,
    ReportResponse,
)
from ai.rag.agent import RadiologyRAGAgent

router = APIRouter()
rag_agent = RadiologyRAGAgent()


@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def get_health():
    """
    Health check endpoint to verify backend service status.
    """
    return HealthCheckResponse(
        status="healthy",
        service="radiology-backend"
    )


@router.post("/api/upload", response_model=ScanUploadResponse, tags=["Radiology API"])
async def upload_scan(file: UploadFile = File(...)):
    """
    Upload DICOM or medical image file for processing.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    scan_id = f"scan_{uuid.uuid4().hex[:8]}"
    file_type = file.content_type or "image/png"

    # TODO (Shivam / Shanks): Store uploaded file to data/raw/ or S3 storage
    return ScanUploadResponse(
        scan_id=scan_id,
        filename=file.filename,
        file_type=file_type,
        message="Scan uploaded successfully to development mock storage"
    )


@router.post("/api/analyze", response_model=AnalysisResponse, tags=["Radiology API"])
async def analyze_scan(request: Optional[AnalysisRequest] = None):
    """
    Triggers multimodal radiology AI analysis pipeline.
    Returns structured mock findings, impression, confidence, and retrieved evidence.
    """
    scan_id = request.scan_id if request and request.scan_id else f"scan_{uuid.uuid4().hex[:8]}"

    # Invoke AI RAG Agent (currently returns mock payload per requirements)
    pipeline_result = rag_agent.run_pipeline(
        vision_findings=["Mock finding for development: Subtle opacity in right lower lobe."]
    )

    return AnalysisResponse(
        status="success",
        scan_id=scan_id,
        findings=pipeline_result["findings"],
        impression=pipeline_result["impression"],
        confidence=pipeline_result["confidence"],
        evidence=[
            EvidenceItem(
                source=item.get("source", "Mock Medical Source"),
                text=item.get("text", "Mock retrieved evidence")
            )
            for item in pipeline_result.get("evidence", [])
        ]
    )


@router.get("/api/report/{id}", response_model=ReportResponse, tags=["Radiology API"])
async def get_report(id: str):
    """
    Retrieve generated radiology report by ID.
    """
    return ReportResponse(
        id=id,
        scan_id=f"scan_{id}",
        created_at=datetime.datetime.utcnow(),
        impression="Mock radiology impression: Findings suggest mild atelectasis or early infiltrates. Correlate clinically.",
        findings=[
            "Mock finding for development: Subtle right lower lobe ground-glass opacity.",
            "Cardiomediastinal silhouette is within normal limits.",
            "No pneumothorax or pleural effusion."
        ],
        confidence=0.85,
        evidence=[
            EvidenceItem(
                source="Mock Medical Reference Manual Vol 4",
                text="Pulmonary opacity in lower lung fields with sub-centimeter margin typically indicates localized inflammation or micro-atelectasis."
            )
        ]
    )
