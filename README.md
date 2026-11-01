# Multimodal Agentic Radiology Assistant (Chest X-Ray Research Prototype)

> **Research Prototype Disclaimer**: This repository is an academic **4th-year B.Tech Computer Science capstone research prototype** focused on **Chest X-Ray (DX) analysis**. It is **NOT** a clinical diagnostic system and must **NEVER** be used to replace a radiologist or guide clinical decision-making.

---

## 1. Project Introduction

The **Multimodal Agentic Radiology Assistant** combines medical image preprocessing, multi-label Chest X-Ray abnormality classification, Grad-CAM explainability interfaces, and a 5-stage **LangGraph Agentic Retrieval-Augmented Generation (RAG)** workflow to synthesize structured radiology reports grounded in verifiable thoracic imaging literature citations.

- **Initial Prototype Focus**: **Chest X-Ray (CXR) Analysis** (`PNG`, `JPEG/JPG` image uploads, with DICOM interfaces structured for future extension). CT and MRI modalities are reserved for future extensions.
- **Zero-Dependency Mock Mode (`AI_MODE=mock`)**: Runs out-of-the-box with zero external API keys, zero GPU hardware, zero model weight downloads, and local SQLite persistence.
- **Pluggable Real Mode (`AI_MODE=real`)**: Exposes clean provider interfaces for pretrained DenseNet121 / Rad-DINO vision checkpoints, ChromaDB vector retrieval, and LLM report synthesis, failing explicitly (`HTTP 503`) if required checkpoints or credentials are unconfigured.

---

## 2. Architecture Overview

1. **Frontend (`frontend/`)**: Next.js 14 (App Router), TypeScript, Tailwind CSS, and Lucide React dashboard composed of modular panels (`SystemStatusBar`, `UploadPanel`, `ImageViewer`, `PredictionPanel`, `ReportPanel`, `AgentTracePanel`) communicating through a central typed API client (`frontend/lib/api.ts`).
2. **Backend (`backend/`)**: FastAPI service layer separating configuration (`config.py`), SQLAlchemy ORM persistence (`db/models.py`), Pydantic v2 schemas (`schemas/radiology.py`), business services (`services/storage_service.py`, `services/analysis_service.py`), and thin REST route handlers (`api/endpoints.py`).
3. **AI Package (`ai/`)**: Framework-independent Python package (`ai/common`, `ai/vision`, `ai/classification`, `ai/rag`, `ai/embeddings`, `ai/tests`) importable and testable without starting FastAPI.

See [`docs/architecture.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs/architecture.md) and [`docs/api-contract.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs/api-contract.md) for full specifications.

---

## 3. Actual Directory Structure

```text
radiology/
├── frontend/                           # Next.js 14 Web Dashboard (Owner: Shreyash)
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── AgentTracePanel.tsx
│   │   ├── ImageViewer.tsx
│   │   ├── PredictionPanel.tsx
│   │   ├── ReportPanel.tsx
│   │   ├── SystemStatusBar.tsx
│   │   └── UploadPanel.tsx
│   ├── lib/
│   │   └── api.ts
│   ├── types/
│   │   └── radiology.ts
│   ├── __tests__/
│   │   └── dashboard.test.mjs
│   ├── Dockerfile
│   ├── next.config.js
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   └── tsconfig.json
│
├── backend/                            # FastAPI Backend & Persistence (Owner: Shivam)
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py
│   │   ├── db/
│   │   │   ├── database.py
│   │   │   └── models.py
│   │   ├── schemas/
│   │   │   └── radiology.py
│   │   ├── services/
│   │   │   ├── analysis_service.py
│   │   │   └── storage_service.py
│   │   ├── config.py
│   │   └── main.py
│   ├── tests/
│   │   └── test_api.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── ai/                                 # Decoupled AI & RAG Package
│   ├── common/                         # Shared Pydantic schemas & provider ABCs
│   │   ├── interfaces.py
│   │   └── schemas.py
│   ├── vision/                         # Chest X-Ray preprocessing & vision adapters (Owner: Shanks)
│   │   ├── embeddings.py
│   │   ├── model.py
│   │   └── preprocessing.py
│   ├── classification/                 # Multi-label classifier & Grad-CAM (Owner: Vineet)
│   │   ├── classifier.py
│   │   └── evaluation.py
│   ├── rag/                            # 5-node LangGraph Agentic RAG (Owner: Gourav)
│   │   ├── agent.py
│   │   ├── ingestion.py
│   │   ├── providers.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── embeddings/                     # Biomedical text embedder interface
│   ├── tests/
│   │   └── test_ai_pipeline.py
│   └── config.py
│
├── data/                               # Dataset & Local Storage (Owner: Shreyansh)
│   ├── raw/                            # Raw datasets (git-ignored)
│   ├── processed/                      # Processed tensors & ChromaDB store (git-ignored)
│   ├── samples/                        # Curated thoracic guidelines & synthetic CXR phantom
│   │   ├── chest_xray_guidelines.json
│   │   └── sample_cxr.png
│   └── uploads/                        # Uploaded scan files (git-ignored)
│
├── notebooks/                          # Exploratory Data Analysis (Owner: Shreyansh)
│   └── 01_exploratory_analysis.ipynb
├── docs/                               # Technical Specifications & Audit Reports
│   ├── REPOSITORY_AUDIT.md
│   ├── BOILERPLATE_BUILD_PROMPT.md
│   ├── architecture.md
│   ├── api-contract.md
│   └── evaluation-plan.md
├── scripts/
│   └── seed_sample_data.py             # Generates synthetic CXR & seeds sample report
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── PROJECT.md
└── README.md
```

---

## 4. Prerequisites

- **Python**: `3.11+` (tested on Python 3.11 and 3.12)
- **Node.js**: `18+` and `npm`
- **Docker & Docker Compose**: Optional (for containerized PostgreSQL + Backend + Frontend execution)

---

## 5. Local Setup & Environment Configuration

### Step 1: Copy Environment Template

```bash
cp .env.example .env
```

By default, `.env.example` configures:
- `AI_MODE=mock` (runs immediately with zero external API keys or model weights)
- `DATABASE_URL=sqlite:///./radiology_dev.db` (zero-config local SQLite persistence)
- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`

### Step 2: Backend Execution (FastAPI)

From the **repository root**:

```bash
# Install backend & core AI dependencies
pip install -r backend/requirements.txt

# (Optional) Generate synthetic Chest X-Ray sample and seed local SQLite DB
python scripts/seed_sample_data.py

# Start FastAPI server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Backend API Base**: `http://localhost:8000`
- **Swagger OpenAPI UI**: `http://localhost:8000/docs`
- **Health Endpoint**: `http://localhost:8000/health`

### Step 3: Frontend Execution (Next.js 14)

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser. You can upload `data/samples/sample_cxr.png` (or any PNG/JPEG Chest X-Ray) to run the full pipeline.

---

## 6. Docker Execution

To run the complete stack (**Next.js Frontend + FastAPI Backend + PostgreSQL 15**) in containers:

```bash
docker compose up --build
```

- **Frontend Dashboard**: `http://localhost:3000`
- **FastAPI Backend**: `http://localhost:8000`
- **PostgreSQL 15**: `localhost:5432` (database `radiology_db`, health-checked with persistent volume `postgres_data`)

---

## 7. API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns backend health, DB status, active `AI_MODE`, and model integration readiness |
| `POST` | `/api/upload` | Validates and stores a `PNG` or `JPEG` Chest X-Ray (`multipart/form-data`), persisting a `Scan` row |
| `POST` | `/api/analyze` | Runs preprocessing, classification, Grad-CAM check, and 5-node Agentic RAG for `scan_id`, persisting a `Report` |
| `GET` | `/api/report/{report_id}` | Retrieves a persisted `ReportResponse` by its UUID |
| `GET` | `/api/scans/{scan_id}/image` | Streams the stored Chest X-Ray image file |

See [`docs/api-contract.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs/api-contract.md) for full request/response payloads.

---

## 8. Mock Mode vs. Real Mode

- **`AI_MODE=mock` (Default)**:
  - Uses `DemoPredictionProvider`, `DemoExplainabilityProvider`, `DemoRetrievalProvider`, and `DemoReportGenerator`.
  - Clearly tags all API responses and UI panels with: `"DEMO MODE: Results are simulated and are not generated by a trained medical model."`
  - Never fabricates fake Grad-CAM overlays (`heatmap_available: false`, `heatmap_url: null`).
  - Retrieves real open-access thoracic radiology glossary passages from `data/samples/chest_xray_guidelines.json`.
- **`AI_MODE=real`**:
  - Requires `VISION_WEIGHTS_PATH` (pretrained DenseNet121/Rad-DINO checkpoint), initialized ChromaDB directory (`CHROMA_PERSIST_DIR`), and `OPENAI_API_KEY`.
  - Validates readiness upfront and returns `HTTP 503` with a clear diagnostic message if any required real integration is missing (never silently falls back to mock mode).

---

## 9. Testing Instructions

Run all automated tests from the repository root:

```bash
# 1. Run AI & Backend pytest suites (16 tests, zero GPU/API keys required)
python -m pytest ai/tests backend/tests -v

# 2. Run Python syntax and import verification
python -m compileall backend ai scripts

# 3. Run Frontend unit tests, TypeScript check, and production build
cd frontend
node --test __tests__/dashboard.test.mjs
npx tsc --noEmit
npm run build
```

---

## 10. Known Limitations

1. **Modality Scope**: The prototype supports **Chest X-Ray (`PNG`, `JPEG/JPG`)** only. Native DICOM pixel decoding (`pydicom`/`MONAI`) and CT/MRI volumetric pipelines are structured as future extensions.
2. **Real Model Checkpoints Excluded from Git**: Pretrained PyTorch checkpoints (`.pt`/`.pth`) and large datasets (NIH ChestX-ray14, MIMIC-CXR) are excluded from version control and must be configured via `VISION_WEIGHTS_PATH` when running in `AI_MODE=real`.
3. **Uncalibrated Classification Scores**: Model scores represent uncalibrated outputs in `[0.00, 1.00]` and must not be interpreted as clinical probabilities without post-hoc calibration.
