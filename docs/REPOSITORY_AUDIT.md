# Repository Audit Report (`docs/REPOSITORY_AUDIT.md`)

**Project**: Multimodal Agentic Radiology Assistant (Chest X-Ray Prototype)  
**Audit Date**: 2026-11-01  
**Branch**: `main` (Clean working tree at audit start)

---

## A. Actual Repository Structure (Pre-Boilerplate Upgrade)

```text
radiology/
├── .env.example                            (444 bytes)
├── .gitignore                              (648 bytes)
├── docker-compose.yml                      (862 bytes)
├── PROJECT.md                              (13,776 bytes)
├── README.md                               (5,847 bytes)
├── ai/
│   ├── __init__.py                         (35 bytes)
│   ├── config.py                           (788 bytes)
│   ├── classification/
│   │   ├── __init__.py                     (43 bytes)
│   │   ├── classifier.py                   (811 bytes)
│   │   └── evaluation.py                   (1,129 bytes)
│   ├── embeddings/
│   │   └── __init__.py                     (39 bytes)
│   ├── rag/
│   │   ├── __init__.py                     (32 bytes)
│   │   ├── agent.py                        (1,336 bytes)
│   │   ├── retriever.py                    (951 bytes)
│   │   └── vector_store.py                 (896 bytes)
│   └── vision/
│       ├── __init__.py                     (35 bytes)
│       ├── embeddings.py                   (620 bytes)
│       ├── model.py                        (965 bytes)
│       └── preprocessing.py                (1,024 bytes)
├── backend/
│   ├── Dockerfile                          (290 bytes)
│   ├── requirements.txt                    (113 bytes)
│   └── app/
│       ├── __init__.py                     (37 bytes)
│       ├── main.py                         (1,234 bytes)
│       ├── api/
│       │   ├── __init__.py                 (29 bytes)
│       │   └── endpoints.py                (3,638 bytes)
│       ├── db/
│       │   ├── __init__.py                 (34 bytes)
│       │   ├── database.py                 (1,007 bytes)
│       │   └── models.py                   (1,746 bytes)
│       └── schemas/
│           ├── __init__.py                 (33 bytes)
│           └── radiology.py                (941 bytes)
├── data/
│   ├── processed/
│   │   └── .gitkeep                        (72 bytes)
│   ├── raw/
│   │   └── .gitkeep                        (74 bytes)
│   └── samples/
│       └── .gitkeep                        (65 bytes)
├── docs/
│   └── .gitkeep                            (93 bytes)
├── frontend/
│   ├── Dockerfile                          (123 bytes)
│   ├── next-env.d.ts                       (206 bytes)
│   ├── next.config.js                      (120 bytes)
│   ├── package-lock.json                   (55,440 bytes)
│   ├── package.json                        (569 bytes)
│   ├── postcss.config.js                   (83 bytes)
│   ├── tailwind.config.js                  (254 bytes)
│   ├── tsconfig.json                       (595 bytes)
│   └── app/
│       ├── globals.css                     (330 bytes)
│       ├── layout.tsx                      (501 bytes)
│       └── page.tsx                        (15,420 bytes)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb       (748 bytes)
└── scripts/
    └── seed_sample_data.py                 (415 bytes)
```

---

## B. Existing Implementation Inventory

### 1. Backend (`backend/`)
- **`backend/app/main.py`**: Initializes `FastAPI`, configures `CORSMiddleware` from `CORS_ORIGINS`, registers `api_router`, and calls `init_db()` on startup.
- **`backend/app/db/database.py`**: Creates a SQLAlchemy engine from `DATABASE_URL` (defaulting to `sqlite:///./radiology_dev.db`), defines `SessionLocal`, `init_db()`, and `get_db()`.
- **`backend/app/db/models.py`**: Defines three basic SQLAlchemy models: `Scan` (`id`, `filename`, `file_path`, `file_type`, `uploaded_at`), `Report` (`id`, `scan_id`, `impression`, `confidence`, `evidence_json`, `created_at`), and `Finding` (`id`, `report_id`, `description`, `confidence`).
- **`backend/app/schemas/radiology.py`**: Defines basic Pydantic v2 models (`HealthCheckResponse`, `ScanUploadResponse`, `AnalysisRequest`, `EvidenceItem`, `AnalysisResponse`, `ReportResponse`).
- **`backend/app/api/endpoints.py`**:
  - `GET /health`: Returns `HealthCheckResponse(status="healthy", service="radiology-backend")`.
  - `POST /api/upload`: Generates a random `scan_id` (`scan_<8-hex>`) and returns `ScanUploadResponse` without validating image content, saving the file to disk, or writing a `Scan` row to the database.
  - `POST /api/analyze`: Instantiates `RadiologyRAGAgent()`, passes a hardcoded string `["Mock finding for development: Subtle opacity in right lower lobe."]`, and returns `AnalysisResponse` without reading the uploaded scan, running image preprocessing, running classification, or persisting a `Report`.
  - `GET /api/report/{id}`: Returns a hardcoded `ReportResponse` regardless of `id` and never queries the database.

### 2. AI Layer (`ai/`)
- **`ai/config.py`**: Contains `AIConfig` class constants (`VISION_MODEL_NAME`, `IMAGE_SIZE`, `VECTOR_DB_TYPE`, `TOP_K_RETRIEVAL`, `CHUNKING_SIZE`, `CHUNKING_OVERLAP`, `LLM_MODEL`, `TEMPERATURE`, `CONFIDENCE_THRESHOLD`, `NUM_CLASSES`). Does not read environment variables or define `AI_MODE`.
- **`ai/vision/preprocessing.py`**: Placeholder `DICOMPreprocessor` returning hardcoded `{"modality": "CT", "patient_age": "45", "image_shape": (512, 512), "status": "parsed"}`. Does not open or validate image files.
- **`ai/vision/model.py`**: Placeholder `MedicalVisionModel.predict_anomalies()` returning a single hardcoded dict.
- **`ai/vision/embeddings.py`**: Placeholder `VisionEmbedder.extract_features()` returning `[0.0] * 512`.
- **`ai/classification/classifier.py`**: Placeholder `DiseaseClassifier.classify()` returning 4 hardcoded float probabilities.
- **`ai/classification/evaluation.py`**: Placeholder `ModelEvaluator.generate_gradcam()` returning `{"target_class": target_class, "heatmap_generated": True, "status": "placeholder_heatmap"}` (claims a heatmap was generated when none exists) and `compute_metrics()` returning hardcoded numbers.
- **`ai/rag/vector_store.py`**: Placeholder `MedicalVectorStore` with no-op `add_documents()` and `search()` returning `[]`.
- **`ai/rag/retriever.py`**: Placeholder `RadiologyRetriever.retrieve()` returning a single mock citation dict.
- **`ai/rag/agent.py`**: Placeholder `RadiologyRAGAgent.run_pipeline()` calling `self.retriever.retrieve(query)` and returning a hardcoded dict.

### 3. Frontend (`frontend/`)
- **`frontend/package.json`**: Next.js `14.1.0`, React `^18.2.0`, `lucide-react` `^0.344.0`, Tailwind CSS `^3.4.1`, TypeScript `^5.3.3`.
- **`frontend/app/page.tsx`**: Single 382-line component implementing a basic file input, image `URL.createObjectURL` preview, health check badge, and results card.
  - Hardcodes `fetch` calls directly inside the component.
  - Ignores the `scan_id` returned by `POST /api/upload` when calling `POST /api/analyze`.
  - On backend failure, catches the exception and overwrites `result` with an inline hardcoded mock fallback.
  - Displays "AI Diagnostic Confidence: 85%".

### 4. Infrastructure & Scripts
- **`docker-compose.yml` & `backend/Dockerfile`**: Defines `backend`, `frontend`, and `db` (`postgres:15-alpine`) services.
- **`scripts/seed_sample_data.py`**: Prints two strings and performs no database or file operations.
- **`notebooks/01_exploratory_analysis.ipynb`**: Valid two-cell Jupyter notebook stub.

---

## C. Missing Functionality

1. **No `AI_MODE` Configuration (`mock` vs `real`)**: No separation between simulated demo providers (`DemoPredictionProvider`, `DemoExplainabilityProvider`, `DemoRetrievalProvider`, `DemoReportGenerator`) and real model/RAG providers.
2. **No File Validation or Storage**: Uploaded images are discarded; no file size checks, magic-byte/Pillow verification, safe filename sanitization, or storage in `data/uploads/`.
3. **No Database Persistence in API Workflow**: `Scan`, `Report`, and `Finding` tables are never read from or written to during `/api/upload`, `/api/analyze`, or `/api/report/{report_id}`.
4. **No Service Layer in Backend**: `endpoints.py` directly instantiates `RadiologyRAGAgent` inside the router module.
5. **No Shared AI Schemas or Interfaces (`ai/common/`)**: No typed contracts connecting preprocessing, vision inference, classification, explainability, and RAG.
6. **No Real Image Loading or Preprocessing**: `ai/vision/preprocessing.py` does not load PNG/JPEG Chest X-Rays, convert to grayscale, resize, or normalize pixel arrays.
7. **No Multi-Node Agentic RAG Workflow**: Missing LangGraph state definition and the 5-stage pipeline (`Query Planner`, `Medical Evidence Retriever`, `Reranker`, `Evidence Checker`, `Report Generator`) with conditional routing and stage execution traces.
8. **No Document Ingestion or Chunking Module**: Missing document loader, metadata extractor, and text chunker (`ai/rag/ingestion.py`) and reranker (`ai/rag/reranker.py`).
9. **No Curated Local Evidence Corpus**: Missing real open-access radiology guideline excerpts in `data/samples/` for honest demo citations.
10. **Missing Frontend Panels & Architecture**:
    - No centralized typed API client (`frontend/lib/api.ts`) or shared types (`frontend/types/radiology.ts`).
    - No `PredictionPanel` for multi-label Chest X-Ray abnormality scores.
    - No `ImageViewer` with scan metadata and honest Grad-CAM availability display.
    - No `AgentTracePanel` showing actual execution stages from the backend.
    - No `SystemStatusBar` showing DB status, AI mode (`mock` vs `real`), and model readiness.
11. **Zero Automated Tests**: Neither `backend/tests/`, `ai/tests/`, nor frontend tests exist.
12. **Missing Technical Documentation**: `docs/architecture.md`, `docs/api-contract.md`, and `docs/evaluation-plan.md` do not exist.

---

## D. Problems and Architectural Inconsistencies

1. **Broken Docker Build & Import Path (`docker-compose.yml` + `backend/Dockerfile`)**:
   - `docker-compose.yml` sets `context: ./backend` for the `backend` service.
   - `backend/Dockerfile` runs `COPY backend/ /app/` and `COPY ai/ /app/ai/`, which fails immediately when the Docker build context is `./backend` because `./ai` is outside the build context and `backend/` is already the root of `./backend`.
   - Furthermore, `backend/Dockerfile` copies `backend/` directly into `/app/` and executes `CMD ["uvicorn", "app.main:app", ...]`, whereas `backend/app/main.py` and `backend/app/api/endpoints.py` use `from backend.app...` imports, causing `ModuleNotFoundError: No module named 'backend'` inside the container.
2. **Database Connection Pitfall in `.env.example`**:
   - `.env.example` sets `DATABASE_URL=postgresql://user:password@localhost:5432/radiology_db`. When a developer runs `cp .env.example .env` for local non-Docker development (as instructed in `README.md`), `backend` will attempt to connect to PostgreSQL and fail if `python-dotenv` loads `.env`.
3. **Misleading Clinical Terminology & Fake Grad-CAM**:
   - `frontend/app/page.tsx` labels an uncalibrated float as `"AI Diagnostic Confidence"`.
   - `ai/classification/evaluation.py` returns `"heatmap_generated": True` with no actual heatmap.
   - `ai/rag/retriever.py` fabricates `"Radiology Reference Manual (Mock)"` and `endpoints.py` fabricates `"Mock Medical Reference Manual Vol 4"`.
4. **Modality Drift**:
   - The existing placeholders mix CT (`"chest_ct_scan.dcm"`, `"modality": "CT"`, Hounsfield Unit windowing) with Chest X-Ray findings, whereas the MVP scope is strictly focused on **Chest X-Ray Analysis** (with CT/MRI reserved as future extensions).
5. **Upload → Analyze State Disconnect**:
   - In `frontend/app/page.tsx`, `handleAnalyze` uploads the file to `/api/upload`, ignores the response, and then posts `{ file_name: ... }` without `scan_id` to `/api/analyze`.

---

## E. Files That Should Be Retained

All existing files in the repository will be preserved (either untouched or enhanced in-place without breaking team ownership boundaries):
- `notebooks/01_exploratory_analysis.ipynb` (Retained untouched for Shreyansh)
- `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/samples/.gitkeep`, `docs/.gitkeep` (Retained)
- `frontend/next.config.js`, `frontend/postcss.config.js`, `frontend/tailwind.config.js`, `frontend/tsconfig.json`, `frontend/next-env.d.ts`, `frontend/app/globals.css`, `frontend/app/layout.tsx`, `frontend/package.json`, `frontend/package-lock.json`, `frontend/Dockerfile` (Retained)
- All package `__init__.py` files (Retained/enhanced with exports)

---

## F. Files That Need Modification

1. `ai/config.py`: Add environment-driven `AI_MODE` (`mock` default, `real`), Chest X-Ray pathology label configuration, embedding/Chroma settings, and real-mode configuration validator.
2. `ai/vision/preprocessing.py`: Implement real `ChestXRayPreprocessor` using `Pillow` and `NumPy` alongside the preserved `DICOMPreprocessor` interface.
3. `ai/vision/model.py`: Implement configurable `VisionModelAdapter` interface and `MedicalVisionModel` supporting mock and real modes.
4. `ai/vision/embeddings.py`: Update `VisionEmbedder` with deterministic feature summary in mock mode and real model hook.
5. `ai/classification/classifier.py`: Implement `DemoPredictionProvider`, `RealPredictionProvider`, and `DiseaseClassifier` returning typed `PredictionResult`.
6. `ai/classification/evaluation.py`: Implement `DemoExplainabilityProvider` (returning `heatmap_url=None`, `heatmap_available=False` honestly), `RealGradCAMProvider`, and real metric calculations in `ModelEvaluator.compute_metrics()`.
7. `ai/rag/vector_store.py`: Implement lazy ChromaDB client wrapper and in-memory/local fallback support.
8. `ai/rag/retriever.py`: Implement `RadiologyRetriever` delegating to `DemoRetrievalProvider` or `RealRetrievalProvider`.
9. `ai/rag/agent.py`: Implement the 5-node LangGraph / stateful RAG workflow (`Query Planner`, `Evidence Retriever`, `Reranker`, `Evidence Checker`, `Report Generator`) with conditional routing and execution trace output.
10. `backend/requirements.txt`: Keep existing dependencies and add `pillow`, `numpy`, `httpx`, `pytest`, and `psycopg2-binary`.
11. `backend/Dockerfile` & `docker-compose.yml`: Fix build context to repo root, preserve `/app/backend` and `/app/ai` layout, add PostgreSQL healthcheck, and configure persistent volumes.
12. `backend/app/db/database.py` & `backend/app/db/models.py`: Load `.env`, default to SQLite only when `DATABASE_URL` is unset, never silently fall back if PostgreSQL is explicitly set, and expand `Scan` and `Report` columns for structured predictions, findings, evidence, execution mode, and agent trace.
13. `backend/app/schemas/radiology.py`: Define unified Pydantic v2 schemas for upload, analysis, report retrieval, predictions, evidence citations, and agent trace stages.
14. `backend/app/api/endpoints.py` & `backend/app/main.py`: Delegate route handling to `StorageService` and `AnalysisService`, inject DB sessions, and add proper error handling.
15. `frontend/app/page.tsx`: Refactor to compose modular dashboard components using `frontend/lib/api.ts`.
16. `scripts/seed_sample_data.py`: Implement working database and sample Chest X-Ray seeding.
17. `.env.example` & `.gitignore`: Default `AI_MODE=mock`, comment PostgreSQL `DATABASE_URL` so local SQLite works out-of-the-box, and ignore `data/uploads/*` and `*.db`.
18. `README.md` & `PROJECT.md`: Update documentation to match the implemented Chest X-Ray prototype architecture and workflows.

---

## G. Proposed New Files

- **AI Layer**:
  - `ai/common/__init__.py`
  - `ai/common/schemas.py`
  - `ai/common/interfaces.py`
  - `ai/rag/ingestion.py`
  - `ai/rag/reranker.py`
  - `ai/rag/providers.py`
  - `ai/tests/__init__.py`
  - `ai/tests/test_ai_pipeline.py`
- **Backend Layer**:
  - `backend/app/config.py`
  - `backend/app/services/__init__.py`
  - `backend/app/services/storage_service.py`
  - `backend/app/services/analysis_service.py`
  - `backend/tests/__init__.py`
  - `backend/tests/test_api.py`
- **Frontend Layer**:
  - `frontend/types/radiology.ts`
  - `frontend/lib/api.ts`
  - `frontend/components/SystemStatusBar.tsx`
  - `frontend/components/UploadPanel.tsx`
  - `frontend/components/ImageViewer.tsx`
  - `frontend/components/PredictionPanel.tsx`
  - `frontend/components/ReportPanel.tsx`
  - `frontend/components/AgentTracePanel.tsx`
- **Data & Infrastructure**:
  - `data/samples/chest_xray_guidelines.json`
  - `data/samples/sample_cxr.png`
  - `data/uploads/.gitkeep`
  - `.dockerignore`
- **Documentation**:
  - `docs/REPOSITORY_AUDIT.md`
  - `docs/BOILERPLATE_BUILD_PROMPT.md`
  - `docs/architecture.md`
  - `docs/api-contract.md`
  - `docs/evaluation-plan.md`
