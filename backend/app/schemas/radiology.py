from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime


class HealthCheckResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ScanUploadResponse(BaseModel):
    scan_id: str
    filename: str
    file_type: str
    message: str


class AnalysisRequest(BaseModel):
    scan_id: Optional[str] = None
    file_name: Optional[str] = None
    notes: Optional[str] = None


class EvidenceItem(BaseModel):
    source: str
    text: str


class AnalysisResponse(BaseModel):
    status: str
    scan_id: Optional[str] = "scan_mock_123"
    findings: List[str]
    impression: str
    confidence: float
    evidence: List[EvidenceItem]


class ReportResponse(BaseModel):
    id: str
    scan_id: str
    created_at: datetime
    impression: str
    findings: List[str]
    confidence: float
    evidence: List[EvidenceItem]
