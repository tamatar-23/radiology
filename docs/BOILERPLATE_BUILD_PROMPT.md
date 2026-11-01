# Repository-Specific Boilerplate Implementation Prompt (`docs/BOILERPLATE_BUILD_PROMPT.md`)

> **Purpose**: Actionable, file-level engineering specification derived from the Stage Zero repository audit (`docs/REPOSITORY_AUDIT.md`) to upgrade the `radiology-ai` skeleton into a functional, modular **Chest X-Ray Analysis & Agentic RAG Research Prototype**.

---

## 1. Architectural Mandate & Scope

1. **Modality Scope**: The working prototype focuses exclusively on **Chest X-Ray (CXR) Analysis** (`PNG`, `JPEG/JPG` initial upload formats, with DICOM interfaces structured for future extension). Do not claim CT or MRI execution in the initial prototype.
2. **Research Prototype Disclaimer**: The system is an academic research prototype and **never** claims to replace radiologists or provide clinical diagnostic confirmation. Avoid calling uncalibrated model scores "clinical confidence."
3. **Dual Execution Mode (`AI_MODE=mock` vs `AI_MODE=real`)**:
   - Default to `AI_MODE=mock`.
   - `mock` mode must run with zero external API keys, zero GPU hardware, zero running ChromaDB daemon, and zero multi-gigabyte model downloads.
   - `real` mode must validate model paths, vector store availability, and LLM credentials explicitly, raising clear errors when unavailable and never silently falling back to mock mode.
4. **Module Independence**:
   - `ai/` must be importable and testable without starting FastAPI (`from ai...`).
   - `backend/` API route handlers (`backend/app/api/endpoints.py`) must remain thin and delegate file handling and pipeline orchestration to `backend/app/services/`.
   - `frontend/` must communicate with the backend exclusively through a central typed client (`frontend/lib/api.ts`) configured via `NEXT_PUBLIC_API_BASE_URL`.

---

## 2. File-by-File Implementation Instructions

### Phase A: Shared AI Schemas, Config & Interfaces (`ai/`)

#### 1. `ai/config.py`
Update `AIConfig` to read environment variables dynamically while keeping class attributes backward-compatible:
- `AI_MODE`: `os.getenv("AI_MODE", "mock").lower()` (`"mock"` or `"real"`)
- `MODALITY`: `"DX"` (Chest X-Ray)
- `VISION_MODEL_NAME`: `os.getenv("VISION_MODEL_NAME", "densenet121-cxr")`
- `VISION_WEIGHTS_PATH`: `os.getenv("VISION_WEIGHTS_PATH", "")`
- `IMAGE_SIZE`: `(224, 224)` (configurable, with `(512, 512)` display normalization)
- `CHEST_XRAY_LABELS`: Default NIH-14 Chest X-Ray pathology labels (`Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia`).
- `EMBEDDING_MODEL_NAME`: `os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")`
- `VECTOR_DB_TYPE`: `"chroma"`
- `CHROMA_PERSIST_DIR`: `os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma_db")`
- `SAMPLE_GUIDELINES_PATH`: `"data/samples/chest_xray_guidelines.json"`
- `LLM_MODEL`: `os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")`
- `OPENAI_API_KEY`: `os.getenv("OPENAI_API_KEY", "")`
- Add `get_ai_mode()` and `validate_real_mode_config()` methods.

#### 2. `ai/common/schemas.py` & `ai/common/interfaces.py`
Define Pydantic v2 models shared across AI modules:
- `ExecutionMode`: `Literal["mock", "real"]`
- `ImageMetadata`: `width`, `height`, `channels`, `format`, `mode`, `mean_intensity`, `std_intensity`, `modality`
- `PreprocessedImage`: `scan_id`, `file_path`, `metadata: ImageMetadata`, `target_size: Tuple[int, int]`, `pixel_array: Any` (excluded from JSON serialization)
- `LabelPrediction`: `label: str`, `score: float` (`0.0 <= score <= 1.0`)
- `PredictionResult`: `predictions: List[LabelPrediction]`, `model_name: str`, `mode: ExecutionMode`
- `ExplainabilityResult`: `heatmap_available: bool`, `heatmap_url: Optional[str] = None`, `target_class: Optional[str] = None`, `method: str = "gradcam"`, `mode: ExecutionMode`, `message: str`
- `EvidenceCitation`: `evidence_id: str`, `document_title: str`, `source_url: str`, `passage: str`, `chunk_id: str`, `relevance_score: float`
- `AgentStageTrace`: `stage_id: str`, `stage_name: str`, `status: Literal["completed", "skipped", "flagged", "failed"]`, `mode: ExecutionMode`, `duration_ms: float`, `details: str`
- `RAGWorkflowResult`: `findings: List[str]`, `impression: str`, `evidence: List[EvidenceCitation]`, `evidence_status: Literal["sufficient", "insufficient", "unverified_claims"]`, `verification_notes: str`, `agent_trace: List[AgentStageTrace]`, `mode: ExecutionMode`

Define abstract base classes in `ai/common/interfaces.py`:
- `BasePredictionProvider`
- `BaseExplainabilityProvider`
- `BaseRetrievalProvider`
- `BaseReranker`
- `BaseReportGenerator`

---

### Phase B: Vision & Classification Pipeline (`ai/vision/`, `ai/classification/`)

#### 1. `ai/vision/preprocessing.py` (Owner: Shanks)
- Preserve `DICOMPreprocessor` for future DICOM header/windowing work.
- Implement `ChestXRayPreprocessor`:
  - `validate_and_load(file_path: str, scan_id: str = "") -> PreprocessedImage`:
    - Opens file with `PIL.Image.open()`, calls `.verify()` on a copy to detect corrupted/non-image payloads, converts to 8-bit grayscale (`"L"`), computes `mean_intensity` and `std_intensity` via `numpy`, resizes to `target_size` using bilinear/bicubic resampling, and normalizes to `[0.0, 1.0]` float32 `np.ndarray`.

#### 2. `ai/vision/model.py` & `ai/vision/embeddings.py` (Owner: Shanks)
- Implement `VisionModelAdapter` interface so DenseNet121, Rad-DINO, or BiomedCLIP backbones can be plugged in without changing downstream callers.
- Preserve `MedicalVisionModel` and `VisionEmbedder` with deterministic feature summary extraction in mock mode and explicit validation in real mode.

#### 3. `ai/classification/classifier.py` (Owner: Vineet)
- Implement `DemoPredictionProvider(BasePredictionProvider)`:
  - Computes a deterministic seed from the preprocessed image's intensity distribution so different Chest X-Rays produce reproducible, realistic multi-label score distributions across configurable Chest X-Ray labels.
  - Explicitly sets `mode="mock"` and `model_name="demo-cxr-densenet121-mock"`.
- Implement `RealPredictionProvider(BasePredictionProvider)`:
  - Checks `AIConfig.VISION_WEIGHTS_PATH`; if missing or PyTorch/MONAI weights are not loaded, raises a clear `RuntimeError` explaining how to configure `VISION_WEIGHTS_PATH`.
- Update `DiseaseClassifier` to support both providers and retain backward-compatible `.classify()` alongside `.predict() -> PredictionResult`.

#### 4. `ai/classification/evaluation.py` (Owner: Vineet)
- Implement `DemoExplainabilityProvider(BaseExplainabilityProvider)`:
  - Never fabricates fake heatmaps with arbitrary image filters.
  - Returns `ExplainabilityResult(heatmap_available=False, heatmap_url=None, target_class=target_class, mode="mock", message="Grad-CAM heatmap generation is unavailable in mock mode (requires trained convolutional/attention weights in real mode).")`.
- Implement `RealGradCAMProvider(BaseExplainabilityProvider)`:
  - Validates real model layer hooks and raises a clear error if model weights are not loaded.
- Update `ModelEvaluator.compute_metrics(y_true, y_pred, threshold=0.5)` to compute real accuracy, precision, recall, and F1-score from input arrays instead of returning hardcoded numbers.

---

### Phase C: Agentic RAG Module (`ai/rag/` & `data/samples/`)

#### 1. `data/samples/chest_xray_guidelines.json`
Create a curated local reference dataset containing real, publicly accessible thoracic radiology glossary definitions and imaging criteria (Fleischner Society Glossary of Terms for Thoracic Imaging, ACR Appropriateness Criteria for Acute Respiratory Illness / Routine Chest Radiography, and standard radiographic signs for Atelectasis, Pleural Effusion, Consolidation, Cardiomegaly, Pneumothorax, and Pulmonary Edema) with authentic DOIs/URLs.

#### 2. `ai/rag/ingestion.py` (Owner: Gourav)
- Implement `DocumentChunk` schema and `MedicalDocumentIngestor`:
  - `extract_metadata(raw_doc: Dict[str, Any]) -> Dict[str, Any]`
  - `chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]`
  - `ingest_json_corpus(file_path: str) -> List[DocumentChunk]`

#### 3. `ai/rag/vector_store.py` (Owner: Gourav)
- Implement `MedicalVectorStore` wrapping ChromaDB with lazy `import chromadb` so `mock` mode does not require a running ChromaDB server, while supporting persistent local ChromaDB collections when invoked.

#### 4. `ai/rag/reranker.py` (Owner: Gourav)
- Implement `EvidenceReranker(BaseReranker)` that scores and sorts candidate `EvidenceCitation` items against planned clinical queries and top predicted labels.

#### 5. `ai/rag/providers.py` & `ai/rag/retriever.py` (Owner: Gourav)
- Implement `DemoRetrievalProvider(BaseRetrievalProvider)`:
  - Loads chunks from `data/samples/chest_xray_guidelines.json` via `MedicalDocumentIngestor`, matches against query terms (e.g., `Atelectasis`, `Pleural Effusion`, `Consolidation`, `Cardiomegaly`), and returns real passages from the local reference corpus as `EvidenceCitation` objects.
- Implement `DemoReportGenerator(BaseReportGenerator)`:
  - Synthesizes structured Chest X-Ray `findings` and `impression` clearly distinguishing:
    1. Simulated vision classification scores (`DEMO MODE`).
    2. Retrieved reference literature passages with `[chunk_id]` citations.
    3. Synthesized research summary.
- Implement `RealRetrievalProvider` and `RealReportGenerator` with strict validation for ChromaDB collections and `OPENAI_API_KEY`.

#### 6. `ai/rag/agent.py` (Owner: Gourav)
Implement the 5-stage stateful LangGraph workflow (`RadiologyRAGState` TypedDict):
1. **`query_planner`**: Formulates targeted thoracic literature queries from top predicted Chest X-Ray labels (`score >= threshold`) and optional clinical notes.
2. **`evidence_retriever`**: Retrieves candidate passages via the configured `RetrievalProvider`.
3. **`reranker`**: Reranks retrieved passages and filters low-relevance chunks.
4. **`evidence_checker`**: Evaluates whether retrieved passages cover the predicted labels; sets `evidence_status` (`"sufficient"`, `"insufficient"`, or `"unverified_claims"`) and determines conditional routing.
5. **`report_generator`**: Generates the structured findings and impression grounded in verified citations (or explicitly flags insufficient evidence).
- Build the graph with `langgraph.graph.StateGraph` when installed (with a deterministic sequential fallback if `langgraph` is absent in minimal environments) and record `AgentStageTrace` entries for every node.

---

### Phase D: Backend Service Layer, Database & APIs (`backend/`)

#### 1. `backend/app/config.py` (Owner: Shivam)
Centralize backend configuration:
- `AI_MODE`: `os.getenv("AI_MODE", "mock").lower()`
- `DATABASE_URL`: `os.getenv("DATABASE_URL", "sqlite:///./radiology_dev.db")`
- `UPLOAD_DIR`: `os.getenv("UPLOAD_DIR", "data/uploads")`
- `MAX_UPLOAD_SIZE_BYTES`: `int(os.getenv("MAX_UPLOAD_SIZE_MB", "15")) * 1024 * 1024`
- `ALLOWED_MEDIA_TYPES`: `{"image/png", "image/jpeg", "image/jpg"}`

#### 2. `backend/app/db/database.py` & `backend/app/db/models.py` (Owner: Shivam)
- In `database.py`:
  - Default to `sqlite:///./radiology_dev.db` only when `DATABASE_URL` is unset or empty.
  - Never silently fall back to SQLite if a configured PostgreSQL URL fails to connect.
  - Provide `check_db_connection() -> Tuple[bool, str]` for `/health`.
- In `models.py`:
  - Expand `Scan` with `stored_path`, `media_type`, `file_size_bytes`, `status`, `image_metadata_json`, while keeping `file_path` and `file_type` properties/columns for backward compatibility.
  - Expand `Report` with `status`, `execution_mode`, `model_name`, `predictions_json`, `findings_json`, `impression`, `confidence`, `heatmap_url`, `evidence_json`, `agent_trace_json`, `model_metadata_json`, and `created_at`.

#### 3. `backend/app/schemas/radiology.py` (Owner: Shivam)
Define strict Pydantic v2 models used across `/health`, `/api/upload`, `/api/analyze`, and `/api/report/{report_id}`:
- `ScanUploadResponse`: `scan_id`, `filename`, `media_type`, `status`, plus backward-compatible `file_type` and `message`.
- `AnalysisRequest`: `scan_id: str`, `notes: Optional[str] = None`.
- `ReportResponse` / `AnalysisResponse`:
  - `report_id: str` (and `id: str` alias)
  - `scan_id: str`
  - `status: str`
  - `mode: Literal["mock", "real"]`
  - `disclaimer: str`
  - `model_name: str`
  - `predictions: List[LabelPrediction]`
  - `heatmap_available: bool`
  - `heatmap_url: Optional[str] = None`
  - `explainability_message: str`
  - `findings: List[str]`
  - `impression: str`
  - `evidence_status: str`
  - `evidence: List[EvidenceCitation]`
  - `agent_trace: List[AgentStageTrace]`
  - `image_metadata: Optional[Dict[str, Any]] = None`
  - `created_at: datetime`

#### 4. `backend/app/services/storage_service.py` & `backend/app/services/analysis_service.py` (Owner: Shivam)
- `StorageService`:
  - Reads uploaded bytes, enforces `MAX_UPLOAD_SIZE_BYTES`, verifies actual image format via `Pillow` (`PNG` or `JPEG`) rather than trusting filename or MIME header alone, sanitizes filename, writes to `data/uploads/{scan_id}_{safe_name}`, and saves `Scan` to DB.
- `AnalysisService`:
  - Looks up `Scan` in DB (raising HTTP `404` if not found).
  - If `AI_MODE=real`, validates real-mode readiness (raising HTTP `503` with descriptive error if weights/keys are missing).
  - Executes Stage 1 (`Image Preprocessing`) and Stage 2 (`Vision Inference & Classification` + `Grad-CAM Explainability Check`), recording `AgentStageTrace` entries.
  - Executes Stages 3–7 via `RadiologyRAGAgent` (`Query Planning`, `Evidence Retrieval`, `Reranking`, `Evidence Verification`, `Report Generation`).
  - Persists `Report` and `Finding` rows in SQLAlchemy and returns the structured response.

#### 5. `backend/app/api/endpoints.py` (Owner: Shivam)
- `GET /health`: Returns backend health, DB connectivity, `ai_mode`, and integration readiness.
- `POST /api/upload`: Delegates to `StorageService.save_upload()`.
- `POST /api/analyze`: Delegates to `AnalysisService.run_analysis()`.
- `GET /api/report/{report_id}`: Delegates to `AnalysisService.get_report_by_id()`.
- `GET /api/scans/{scan_id}/image`: Streams the stored scan image file for the frontend viewer.

---

### Phase E: Frontend Dashboard (`frontend/`)

1. **`frontend/types/radiology.ts`**: TypeScript interfaces matching all backend Pydantic schemas.
2. **`frontend/lib/api.ts`**: Central typed client (`getHealth`, `uploadScan`, `analyzeScan`, `getReport`, `getScanImageUrl`).
3. **`frontend/components/SystemStatusBar.tsx`**: Displays Backend status, Database status, AI Mode badge (`MOCK DEMO` vs `REAL AI`), and configured vision/RAG models.
4. **`frontend/components/UploadPanel.tsx`**: Drag-and-drop Chest X-Ray uploader (`PNG`, `JPEG`), client-side file validation, upload status indicator, clinical notes input, and "Run Analysis" trigger.
5. **`frontend/components/ImageViewer.tsx`**: Displays uploaded Chest X-Ray image, `scan_id`, filename, dimensions/grayscale stats, and an Explainability (Grad-CAM) toggle that clearly states when heatmap generation is unavailable in mock mode.
6. **`frontend/components/PredictionPanel.tsx`**: Displays sorted multi-label Chest X-Ray abnormality scores (`0.00–1.00`), horizontal score bars, model identifier, execution mode badge, and terminology note explaining that scores are uncalibrated model outputs.
7. **`frontend/components/ReportPanel.tsx`**: Displays prominent Demo Mode banner when `mode === "mock"`, structured `Findings`, `Impression`, `evidence_status` badge, and retrieved `EvidenceCitation` cards with clickable `source_url`, `document_title`, `chunk_id`, `relevance_score`, and `passage`.
8. **`frontend/components/AgentTracePanel.tsx`**: Displays the 7 pipeline stages returned by `result.agent_trace` with dynamic status badges (`completed`, `flagged`, `skipped`, `failed`), execution latency (`duration_ms`), stage description, and simulated-trace badge in mock mode.
9. **`frontend/app/page.tsx`**: Composes all panels into a responsive, professional medical research dashboard.

---

### Phase F: Docker, Testing & Documentation

1. **Docker Fixes**:
   - Update `docker-compose.yml` so `backend` uses `context: .` and `dockerfile: backend/Dockerfile`, mounts `./data/uploads:/app/data/uploads`, and waits for `db` healthcheck (`pg_isready`).
   - Update `backend/Dockerfile` to copy `backend/` into `/app/backend/`, `ai/` into `/app/ai/`, `data/samples/` into `/app/data/samples/`, and start `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`.
   - Create `.dockerignore` to exclude `node_modules`, `.next`, `.git`, `*.db`, `data/raw/*`, `data/processed/*`, `data/uploads/*`, and model weight files.
2. **Automated Tests**:
   - `ai/tests/test_ai_pipeline.py`: Image loading/preprocessing, schema validation, mock classifier, honest mock Grad-CAM (`heatmap_url is None`), RAG 5-node transitions, empty-evidence conditional routing, and evidence citation schema validation.
   - `backend/tests/test_api.py`: `/health`, upload validation (reject non-image/oversized files), successful PNG upload, `/api/analyze` persistence & trace verification, `/api/report/{report_id}` retrieval, `404` on invalid `scan_id`, `404` on nonexistent `report_id`, and `503` when `AI_MODE=real` lacks weights.
3. **Documentation**:
   - Update `README.md` and `PROJECT.md`.
   - Create `docs/architecture.md`, `docs/api-contract.md`, and `docs/evaluation-plan.md`.
