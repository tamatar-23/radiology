"""
Automated Integration Tests for the FastAPI Backend
Tests /health, /api/upload, /api/analyze, /api/report/{report_id},
input validation, database persistence, and mock vs real mode behavior.
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.db.models import Base
from backend.app.db.database import get_db


# Use an isolated in-memory SQLite database for automated tests
TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)
Base.metadata.create_all(bind=TEST_ENGINE)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def _create_sample_png_bytes(width: int = 160, height: int = 160) -> bytes:
    arr = np.linspace(20, 220, num=width * height, dtype=np.uint8).reshape((height, width))
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(monkeypatch):
    monkeypatch.setenv("AI_MODE", "mock")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "radiology-backend"
    assert data["ai_mode"] == "mock"
    assert "integrations" in data


def test_upload_validation_rejects_non_image():
    files = {"file": ("fake_scan.png", b"this is plain text not an image", "image/png")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "not a valid image" in response.json()["detail"].lower()


def test_upload_validation_rejects_empty_file():
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_successful_upload_and_analyze_and_report_retrieval(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_MODE", "mock")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

    # 1. Upload valid PNG Chest X-Ray
    png_bytes = _create_sample_png_bytes()
    upload_res = client.post(
        "/api/upload",
        files={"file": ("patient_cxr_test.png", png_bytes, "image/png")},
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["status"] == "uploaded"
    assert upload_data["media_type"] == "image/png"
    assert upload_data["filename"] == "patient_cxr_test.png"
    scan_id = upload_data["scan_id"]
    assert scan_id

    # 2. Trigger Analysis
    analyze_res = client.post(
        "/api/analyze",
        json={"scan_id": scan_id, "notes": "Evaluate for basal atelectasis"},
    )
    assert analyze_res.status_code == 200
    report_data = analyze_res.json()

    assert report_data["scan_id"] == scan_id
    assert report_data["report_id"]
    assert report_data["status"] == "completed"
    assert report_data["mode"] == "mock"
    assert "DEMO MODE" in report_data["disclaimer"]
    assert len(report_data["predictions"]) == 14
    assert report_data["heatmap_available"] is False
    assert report_data["heatmap_url"] is None
    assert len(report_data["findings"]) >= 2
    assert "DEMO MODE" in report_data["impression"]
    assert len(report_data["evidence"]) >= 1

    # Verify full 7-stage execution trace (2 vision + 5 RAG nodes)
    trace_stages = [s["stage_id"] for s in report_data["agent_trace"]]
    assert trace_stages == [
        "image_preprocessing",
        "vision_inference",
        "query_planning",
        "evidence_retrieval",
        "reranking",
        "evidence_verification",
        "report_generation",
    ]

    # 3. Retrieve persisted report via GET /api/report/{report_id}
    report_id = report_data["report_id"]
    get_res = client.get(f"/api/report/{report_id}")
    assert get_res.status_code == 200
    persisted = get_res.json()
    assert persisted["report_id"] == report_id
    assert persisted["scan_id"] == scan_id
    assert persisted["impression"] == report_data["impression"]
    assert len(persisted["predictions"]) == 14


def test_analyze_invalid_scan_id_returns_404(monkeypatch):
    monkeypatch.setenv("AI_MODE", "mock")
    response = client.post(
        "/api/analyze",
        json={"scan_id": "nonexistent-scan-uuid-999", "notes": None},
    )
    assert response.status_code == 404
    assert "nonexistent-scan-uuid-999" in response.json()["detail"]


def test_get_missing_report_returns_404():
    response = client.get("/api/report/nonexistent-report-uuid-999")
    assert response.status_code == 404
    assert "nonexistent-report-uuid-999" in response.json()["detail"]


def test_real_mode_returns_503_when_weights_and_keys_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("AI_MODE", "mock")

    png_bytes = _create_sample_png_bytes()
    upload_res = client.post(
        "/api/upload",
        files={"file": ("cxr_real_test.png", png_bytes, "image/png")},
    )
    scan_id = upload_res.json()["scan_id"]

    # Switch to real mode without configuring VISION_WEIGHTS_PATH or OPENAI_API_KEY
    monkeypatch.setenv("AI_MODE", "real")
    monkeypatch.setenv("VISION_WEIGHTS_PATH", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    analyze_res = client.post("/api/analyze", json={"scan_id": scan_id})
    assert analyze_res.status_code == 503
    assert "AI_MODE=real" in analyze_res.json()["detail"]
