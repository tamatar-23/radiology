"""
AI Pipeline Orchestration & Report Persistence Service
Owner: Shivam (Backend Integration)

Coordinates:
  1. Scan retrieval from SQLAlchemy.
  2. Chest X-Ray preprocessing (ai/vision).
  3. Multi-label abnormality classification (ai/classification).
  4. Grad-CAM explainability check (ai/classification).
  5. 5-stage LangGraph Agentic RAG workflow (ai/rag).
  6. Persistence of Report and Finding records.
"""

import os
import time
import uuid
import logging
from typing import List, Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.models import Scan, Report, Finding
from backend.app.schemas.radiology import (
    AnalysisRequest,
    ReportResponse,
    DEMO_MODE_DISCLAIMER,
    REAL_MODE_DISCLAIMER,
)
from ai.config import AIConfig
from ai.common.schemas import (
    ExecutionMode,
    LabelPrediction,
    EvidenceCitation,
    AgentStageTrace,
)
from ai.vision.preprocessing import ChestXRayPreprocessor
from ai.classification.classifier import DiseaseClassifier
from ai.classification.evaluation import ModelEvaluator
from ai.rag.agent import RadiologyRAGAgent

logger = logging.getLogger(__name__)


class AnalysisService:
    """Coordinates the Chest X-Ray AI & Agentic RAG pipeline and persists results."""

    def __init__(self):
        self.preprocessor = ChestXRayPreprocessor()

    def run_analysis(self, request: AnalysisRequest, db: Session) -> ReportResponse:
        if not request or not request.scan_id:
            raise HTTPException(
                status_code=400,
                detail="scan_id is required to run Chest X-Ray analysis.",
            )

        scan: Optional[Scan] = db.query(Scan).filter(Scan.id == request.scan_id).first()
        if not scan:
            raise HTTPException(
                status_code=404,
                detail=f"Scan with scan_id '{request.scan_id}' does not exist.",
            )

        if not os.path.isfile(scan.file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Stored image file for scan '{scan.id}' is missing on disk.",
            )

        try:
            raw_mode = settings.get_ai_mode()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        mode: ExecutionMode = "mock" if raw_mode == "mock" else "real"

        # In real mode, validate required checkpoints and credentials upfront (never silently fall back)
        if mode == "real":
            status_check = AIConfig.validate_real_mode_status()
            if not status_check["all_ready"]:
                missing_details = "; ".join(status_check["missing_integrations"])
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"AI_MODE=real is configured, but required real AI integrations are unavailable: "
                        f"{missing_details}. Configure the missing components or set AI_MODE=mock."
                    ),
                )

        pipeline_trace: List[AgentStageTrace] = []

        # ------------------------------------------------------------------
        # Stage 1: Image Preprocessing
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            preprocessed = self.preprocessor.preprocess_file(
                file_path=scan.file_path, scan_id=scan.id
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to preprocess Chest X-Ray image: {exc}",
            ) from exc

        prep_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        pipeline_trace.append(
            AgentStageTrace(
                stage_id="image_preprocessing",
                stage_name="Image Preprocessing",
                status="completed",
                mode=mode,
                duration_ms=prep_ms,
                details=(
                    f"Validated {preprocessed.metadata.format} Chest X-Ray "
                    f"({preprocessed.metadata.width}x{preprocessed.metadata.height}), "
                    f"converted to grayscale and normalized to {preprocessed.target_size[0]}x{preprocessed.target_size[1]}."
                ),
            )
        )

        # ------------------------------------------------------------------
        # Stage 2: Vision Inference, Classification & Grad-CAM Check
        # ------------------------------------------------------------------
        t1 = time.perf_counter()
        try:
            classifier = DiseaseClassifier(mode=mode)
            prediction_result = classifier.predict(preprocessed)

            evaluator = ModelEvaluator(mode=mode)
            top_label = (
                prediction_result.predictions[0].label
                if prediction_result.predictions
                else "Atelectasis"
            )
            explainability = evaluator.explain(preprocessed, target_class=top_label)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        vision_ms = round((time.perf_counter() - t1) * 1000.0, 2)
        top_three = ", ".join(
            f"{p.label} ({p.score:.2f})" for p in prediction_result.predictions[:3]
        )
        pipeline_trace.append(
            AgentStageTrace(
                stage_id="vision_inference",
                stage_name="Vision Inference",
                status="completed",
                mode=mode,
                duration_ms=vision_ms,
                details=(
                    f"Executed {prediction_result.model_name} ({mode} mode). "
                    f"Top candidate scores: {top_three}. "
                    f"Grad-CAM: {'available' if explainability.heatmap_available else 'unavailable in mock mode'}."
                ),
            )
        )

        # ------------------------------------------------------------------
        # Stages 3–7: Agentic RAG Workflow (Query Plan -> Retrieve -> Rerank -> Verify -> Generate)
        # ------------------------------------------------------------------
        try:
            rag_agent = RadiologyRAGAgent(mode=mode)
            rag_result = rag_agent.execute_workflow(
                predictions=prediction_result.predictions,
                clinical_notes=request.notes,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        pipeline_trace.extend(rag_result.agent_trace)

        # ------------------------------------------------------------------
        # Persist Report & Findings in SQLAlchemy
        # ------------------------------------------------------------------
        report_id = str(uuid.uuid4())
        top_score = (
            float(prediction_result.predictions[0].score)
            if prediction_result.predictions
            else 0.0
        )

        report_record = Report(
            id=report_id,
            scan_id=scan.id,
            status="completed",
            execution_mode=mode,
            model_name=prediction_result.model_name,
            predictions_json=[p.model_dump() for p in prediction_result.predictions],
            findings_json=rag_result.findings,
            impression=rag_result.impression,
            confidence=top_score,
            heatmap_available=1 if explainability.heatmap_available else 0,
            heatmap_url=explainability.heatmap_url,
            explainability_message=explainability.message,
            evidence_status=rag_result.evidence_status,
            evidence_json=[e.model_dump() for e in rag_result.evidence],
            agent_trace_json=[t.model_dump() for t in pipeline_trace],
            model_metadata_json={
                "planned_queries": rag_result.planned_queries,
                "verification_notes": rag_result.verification_notes,
                "image_metadata": preprocessed.metadata.model_dump(),
            },
        )
        db.add(report_record)

        for finding_text in rag_result.findings:
            db.add(
                Finding(
                    id=str(uuid.uuid4()),
                    report_id=report_id,
                    description=finding_text,
                    confidence=top_score,
                )
            )

        scan.status = "analyzed"
        db.commit()
        db.refresh(report_record)

        return self._serialize_report(report_record, scan)

    def get_report_by_id(self, report_id: str, db: Session) -> ReportResponse:
        report: Optional[Report] = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"Report with report_id '{report_id}' was not found.",
            )
        scan = db.query(Scan).filter(Scan.id == report.scan_id).first()
        return self._serialize_report(report, scan)

    @staticmethod
    def _serialize_report(report: Report, scan: Optional[Scan] = None) -> ReportResponse:
        mode: ExecutionMode = "real" if report.execution_mode == "real" else "mock"
        disclaimer = DEMO_MODE_DISCLAIMER if mode == "mock" else REAL_MODE_DISCLAIMER

        predictions = [
            LabelPrediction(**item) for item in (report.predictions_json or [])
        ]
        evidence = [
            EvidenceCitation(**item) for item in (report.evidence_json or [])
        ]
        agent_trace = [
            AgentStageTrace(**item) for item in (report.agent_trace_json or [])
        ]
        findings: List[str] = list(report.findings_json or [])
        if not findings and report.findings:
            findings = [f.description for f in report.findings]

        image_meta = None
        if scan and scan.image_metadata_json:
            image_meta = scan.image_metadata_json
        elif report.model_metadata_json and "image_metadata" in report.model_metadata_json:
            image_meta = report.model_metadata_json["image_metadata"]

        return ReportResponse(
            report_id=report.id,
            id=report.id,
            scan_id=report.scan_id,
            status=report.status,
            mode=mode,
            disclaimer=disclaimer,
            model_name=report.model_name,
            predictions=predictions,
            heatmap_available=bool(report.heatmap_available),
            heatmap_url=report.heatmap_url,
            explainability_message=report.explainability_message
            or "Grad-CAM heatmap is unavailable in mock mode.",
            findings=findings,
            impression=report.impression,
            evidence_status=report.evidence_status or "sufficient",
            evidence=evidence,
            agent_trace=agent_trace,
            image_metadata=image_meta,
            confidence=report.confidence,
            created_at=report.created_at,
        )
