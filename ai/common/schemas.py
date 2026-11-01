"""
Shared Pydantic Schemas for the AI Layer
Owner: Shared AI Team (Gourav, Shanks, Vineet)

Defines strict data structures connecting image preprocessing, vision classification,
Grad-CAM explainability, and the LangGraph Agentic RAG workflow.
"""

from typing import List, Optional, Tuple, Literal, Any
from pydantic import BaseModel, Field, ConfigDict

ExecutionMode = Literal["mock", "real"]


class ImageMetadata(BaseModel):
    width: int = Field(..., description="Original image width in pixels")
    height: int = Field(..., description="Original image height in pixels")
    channels: int = Field(..., description="Number of color channels in source image")
    format: str = Field(..., description="Detected image format, e.g., PNG or JPEG")
    color_mode: str = Field(..., description="Original PIL color mode, e.g., L, RGB")
    mean_intensity: float = Field(..., description="Normalized grayscale mean intensity [0, 1]")
    std_intensity: float = Field(..., description="Normalized grayscale standard deviation [0, 1]")
    modality: str = Field(default="DX", description="Imaging modality code (DX = Chest X-Ray)")


class PreprocessedImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scan_id: str
    file_path: str
    metadata: ImageMetadata
    target_size: Tuple[int, int]
    pixel_array: Any = Field(default=None, exclude=True)


class LabelPrediction(BaseModel):
    label: str = Field(..., description="Chest X-Ray abnormality label")
    score: float = Field(..., ge=0.0, le=1.0, description="Uncalibrated model output score [0, 1]")


class PredictionResult(BaseModel):
    predictions: List[LabelPrediction]
    model_name: str
    mode: ExecutionMode


class ExplainabilityResult(BaseModel):
    heatmap_available: bool = Field(
        default=False,
        description="True only when a genuine Grad-CAM activation map was computed",
    )
    heatmap_url: Optional[str] = Field(
        default=None,
        description="URL or path to generated Grad-CAM heatmap, or None if unavailable",
    )
    target_class: Optional[str] = Field(default=None)
    method: str = Field(default="gradcam")
    mode: ExecutionMode = Field(default="mock")
    message: str = Field(
        default="Grad-CAM explainability is unavailable in mock mode.",
        description="Human-readable status explaining heatmap availability",
    )


class EvidenceCitation(BaseModel):
    evidence_id: str = Field(..., description="Unique identifier for this citation")
    document_title: str = Field(..., description="Source reference title")
    source_url: str = Field(..., description="Verifiable URL or DOI for the reference")
    passage: str = Field(..., description="Exact retrieved passage text")
    chunk_id: str = Field(..., description="Chunk identifier within the knowledge base")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Retrieval/reranking score")


class AgentStageTrace(BaseModel):
    stage_id: str = Field(..., description="Machine identifier for pipeline stage")
    stage_name: str = Field(..., description="Display name of pipeline stage")
    status: Literal["completed", "skipped", "flagged", "failed"] = Field(...)
    mode: ExecutionMode = Field(...)
    duration_ms: float = Field(default=0.0, ge=0.0)
    details: str = Field(..., description="Concise execution summary for this stage")


class RAGWorkflowResult(BaseModel):
    planned_queries: List[str]
    findings: List[str]
    impression: str
    evidence: List[EvidenceCitation]
    evidence_status: Literal["sufficient", "insufficient", "unverified_claims"]
    verification_notes: str
    agent_trace: List[AgentStageTrace]
    mode: ExecutionMode
