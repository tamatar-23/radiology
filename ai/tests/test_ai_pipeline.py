"""
Automated Unit Tests for the AI Layer (ai/vision, ai/classification, ai/rag)
Tests both mock mode and real mode (multiscale radiographic classifier, Grad-CAM
spatial heatmap generation, ChromaDB vector store indexing, and LangGraph RAG).
"""

import io
import os
import pytest
import numpy as np
from PIL import Image

from ai.common.schemas import (
    LabelPrediction,
    PredictionResult,
    EvidenceCitation,
)
from ai.vision.preprocessing import ChestXRayPreprocessor
from ai.vision.model import ensure_default_vision_weights
from ai.classification.classifier import (
    DiseaseClassifier,
    DemoPredictionProvider,
    RealPredictionProvider,
)
from ai.classification.evaluation import (
    ModelEvaluator,
    DemoExplainabilityProvider,
    RealGradCAMProvider,
)
from ai.rag.ingestion import MedicalDocumentIngestor
from ai.rag.vector_store import MedicalVectorStore
from ai.rag.providers import DemoRetrievalProvider, RealRetrievalProvider, RealReportGenerator
from ai.rag.agent import RadiologyRAGAgent


def _make_test_png_bytes(width: int = 128, height: int = 128, intensity: int = 120) -> bytes:
    arr = np.full((height, width), intensity, dtype=np.uint8)
    arr[32:96, 32:96] = min(255, intensity + 50)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_loading_and_preprocessing():
    preprocessor = ChestXRayPreprocessor(target_size=(224, 224))
    raw_png = _make_test_png_bytes(256, 256, intensity=110)

    result = preprocessor.preprocess_bytes(raw_png, scan_id="test_scan_01")

    assert result.scan_id == "test_scan_01"
    assert result.metadata.width == 256
    assert result.metadata.height == 256
    assert result.metadata.format == "PNG"
    assert result.metadata.modality == "DX"
    assert 0.0 <= result.metadata.mean_intensity <= 1.0
    assert result.pixel_array.shape == (224, 224)
    assert result.pixel_array.dtype == np.float32
    assert float(np.min(result.pixel_array)) >= 0.0
    assert float(np.max(result.pixel_array)) <= 1.0


def test_invalid_image_bytes_rejected():
    preprocessor = ChestXRayPreprocessor()
    with pytest.raises(ValueError, match="not a valid image"):
        preprocessor.preprocess_bytes(b"not an image payload at all")

    with pytest.raises(ValueError, match="empty"):
        preprocessor.preprocess_bytes(b"")


def test_demo_prediction_provider_and_schema():
    preprocessor = ChestXRayPreprocessor()
    preprocessed = preprocessor.preprocess_bytes(_make_test_png_bytes())

    provider = DemoPredictionProvider()
    pred_result: PredictionResult = provider.predict(preprocessed)

    assert pred_result.mode == "mock"
    assert pred_result.model_name == "demo-cxr-densenet121-mock"
    assert len(pred_result.predictions) == 14
    scores = [p.score for p in pred_result.predictions]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_real_prediction_provider_fails_loudly_without_weights(monkeypatch):
    monkeypatch.setenv("VISION_WEIGHTS_PATH", "")
    preprocessor = ChestXRayPreprocessor()
    preprocessed = preprocessor.preprocess_bytes(_make_test_png_bytes())

    real_provider = RealPredictionProvider(weights_path="")
    with pytest.raises(RuntimeError, match="VISION_WEIGHTS_PATH"):
        real_provider.predict(preprocessed)


def test_real_prediction_and_gradcam_providers(tmp_path, monkeypatch):
    weights_file = str(tmp_path / "test_cxr_weights.npz")
    ensure_default_vision_weights(weights_file)
    monkeypatch.setenv("VISION_WEIGHTS_PATH", weights_file)
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

    preprocessor = ChestXRayPreprocessor()
    preprocessed = preprocessor.preprocess_bytes(
        _make_test_png_bytes(256, 256, intensity=115),
        scan_id="real_scan_test_01",
    )

    real_pred_provider = RealPredictionProvider(weights_path=weights_file)
    pred_res = real_pred_provider.predict(preprocessed)
    assert pred_res.mode == "real"
    assert len(pred_res.predictions) == 14
    scores = [p.score for p in pred_res.predictions]
    assert scores == sorted(scores, reverse=True)
    assert all(0.01 <= s <= 0.98 for s in scores)

    gradcam_provider = RealGradCAMProvider(weights_path=weights_file)
    cam_res = gradcam_provider.generate_heatmap(
        preprocessed, target_class=pred_res.predictions[0].label
    )
    assert cam_res.mode == "real"
    assert cam_res.heatmap_available is True
    assert cam_res.heatmap_url == "/api/scans/real_scan_test_01/heatmap"
    assert os.path.isfile(tmp_path / "gradcam_real_scan_test_01.png")


def test_demo_explainability_never_fakes_heatmap():
    preprocessor = ChestXRayPreprocessor()
    preprocessed = preprocessor.preprocess_bytes(_make_test_png_bytes())

    evaluator = ModelEvaluator(mode="mock")
    explanation = evaluator.explain(preprocessed, target_class="Atelectasis")

    assert explanation.mode == "mock"
    assert explanation.heatmap_available is False
    assert explanation.heatmap_url is None
    assert "unavailable" in explanation.message.lower()


def test_model_evaluator_metrics():
    evaluator = ModelEvaluator(mode="mock")
    metrics = evaluator.compute_metrics(
        y_true=[1, 0, 1, 1, 0],
        y_pred=[0.85, 0.20, 0.75, 0.40, 0.10],
        threshold=0.5,
    )
    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 1.0
    assert round(metrics["recall"], 4) == 0.6667


def test_document_ingestion_and_chunking():
    ingestor = MedicalDocumentIngestor(chunk_size=120, chunk_overlap=20)
    sample_docs = [
        {
            "document_id": "test-doc-1",
            "document_title": "Thoracic Reference Test",
            "source_url": "https://example.org/cxr",
            "keywords": ["atelectasis"],
            "content": (
                "Atelectasis is characterized by reduced lung volume. "
                "Displacement of interlobar fissures and crowding of vessels are classic signs. "
                "Linear basal opacities are frequently observed on upright chest radiographs."
            ),
        }
    ]
    chunks = ingestor.ingest_documents(sample_docs)
    assert len(chunks) >= 2
    assert all(c.document_id == "test-doc-1" for c in chunks)
    assert all(len(c.text) <= 120 for c in chunks)


def test_rag_workflow_state_transitions_and_evidence_schema():
    agent = RadiologyRAGAgent(mode="mock")
    predictions = [
        LabelPrediction(label="Atelectasis", score=0.81),
        LabelPrediction(label="Effusion", score=0.66),
    ]

    result = agent.execute_workflow(
        predictions=predictions,
        clinical_notes="Right lower zone opacity and costophrenic blunting",
    )

    assert result.mode == "mock"
    assert result.evidence_status == "sufficient"
    assert len(result.evidence) > 0
    for ev in result.evidence:
        validated = EvidenceCitation(**ev.model_dump())
        assert validated.document_title
        assert validated.source_url.startswith("http")
        assert validated.chunk_id
        assert 0.0 <= validated.relevance_score <= 1.0

    stage_ids = [s.stage_id for s in result.agent_trace]
    assert stage_ids == [
        "query_planning",
        "evidence_retrieval",
        "reranking",
        "evidence_verification",
        "report_generation",
    ]
    assert "DEMO MODE" in result.impression


def test_real_chromadb_vector_store_and_rag_agent(tmp_path, monkeypatch):
    chroma_dir = str(tmp_path / "chroma_test")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
    monkeypatch.setenv("LLM_PROVIDER", "local_grounded")

    vstore = MedicalVectorStore(persist_directory=chroma_dir)
    num_chunks = vstore.ensure_indexed()
    assert num_chunks >= 15

    agent = RadiologyRAGAgent(
        mode="real",
        retrieval_provider=RealRetrievalProvider(vector_store=vstore),
        report_generator=RealReportGenerator(provider="local_grounded"),
    )
    predictions = [
        LabelPrediction(label="Pneumothorax", score=0.84),
        LabelPrediction(label="Effusion", score=0.62),
    ]
    result = agent.execute_workflow(
        predictions=predictions,
        clinical_notes="Sudden pleuritic chest pain; evaluate apical pleural line",
    )

    assert result.mode == "real"
    assert result.evidence_status == "sufficient"
    assert len(result.evidence) >= 1
    assert "DEMO MODE" not in result.impression
    assert "GROUNDED RADIOLOGY SYNTHESIS" in result.impression
    assert any("pneumothorax" in ev.passage.lower() for ev in result.evidence)


def test_rag_conditional_routing_on_insufficient_evidence():
    agent = RadiologyRAGAgent(mode="mock")
    result = agent.execute_workflow(
        predictions=[LabelPrediction(label="__no_evidence_query__", score=0.90)],
        clinical_notes="__no_evidence_query__",
    )

    assert result.evidence == []
    assert result.evidence_status == "insufficient"
    assert "INSUFFICIENT LITERATURE EVIDENCE" in result.impression
    trace_map = {s.stage_id: s.status for s in result.agent_trace}
    assert trace_map["evidence_retrieval"] == "flagged"
    assert trace_map["reranking"] == "skipped"
    assert trace_map["evidence_verification"] == "flagged"
