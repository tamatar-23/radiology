# Multimodal Agentic Radiology Assistant (Chest X-Ray Research Prototype)

> **Research Prototype Disclaimer**: This repository is an academic **4th-year B.Tech Computer Science capstone research prototype** focused on **Chest X-Ray (DX) analysis**. It is **NOT** a clinical diagnostic system and must **NEVER** be used to replace a radiologist or guide clinical decision-making.

---

## 1. Project Introduction

The **Multimodal Agentic Radiology Assistant** combines medical image preprocessing, 64-dimensional multiscale radiographic feature extraction, 14-label NIH ChestX-ray14 abnormality classification, spatial **Grad-CAM explainability**, and a 5-stage **LangGraph Agentic Retrieval-Augmented Generation (RAG)** workflow to synthesize structured radiology reports grounded in verifiable thoracic imaging literature citations.

- **Modality Scope**: **Chest X-Ray (CXR) Analysis** (`PNG`, `JPEG/JPG` image uploads, with DICOM interfaces structured in `DICOMPreprocessor`).
- **Full Working Real Mode (`AI_MODE=real` — Active Default)**:
  - **Vision & Classification**: Extracts 64 multiscale thoracic features (zonal lung attenuation, cardiothoracic ratio, costophrenic gradients, Sobel/Laplacian textures, and Difference-of-Gaussians blob responses) and predicts 14 NIH ChestX-ray14 pathology scores using the trained checkpoint (`data/models/densenet121_cxr_weights.npz`), achieving **79.82% Balanced Accuracy** and **80.59% Macro ROC-AUC** on held-out test scans.
  - **Grad-CAM Explainability**: Computes spatial class activation heatmaps blended onto the radiograph (`GET /api/scans/{scan_id}/heatmap`).
  - **Vector Store & Agentic RAG**: Indexes 15 peer-reviewed Fleischner Society & ACR thoracic guideline documents (19 chunks) using 384-dimensional `BiomedicalDenseEmbedder` vectors (`data/processed/chroma_db`), achieving **100% Recall@3** and **100% Citation Groundedness**.
  - **Report Synthesis**: Supports **Google Gemini (`gemini-2.0-flash`)**, **Groq (`llama-3.3-70b-versatile`)**, **OpenAI (`gpt-4o-mini`)**, and a built-in **Citation-Grounded Clinical Report Synthesizer (`local_grounded`)** that runs 100% offline.
- **Zero-Dependency Mock Mode (`AI_MODE=mock`)**: Preserved for lightweight unit testing and simulated demonstrations.

---

## 2. Architecture Overview

1. **Frontend (`frontend/`)**: Next.js 14 (App Router), TypeScript, Tailwind CSS, and Lucide React dashboard composed of modular panels (`SystemStatusBar`, `UploadPanel`, `ImageViewer`, `PredictionPanel`, `ReportPanel`, `AgentTracePanel`) communicating through a central typed API client (`frontend/lib/api.ts`). Includes a 1-click **"Load Sample CXR"** button and a live **Grad-CAM Heatmap** tab.
2. **Backend (`backend/`)**: FastAPI service layer separating configuration (`config.py`), SQLAlchemy ORM persistence (`db/models.py`), Pydantic v2 schemas (`schemas/radiology.py`), business services (`services/storage_service.py`, `services/analysis_service.py`), and thin REST route handlers (`api/endpoints.py`).
3. **AI Package (`ai/`)**: Framework-independent Python package (`ai/common`, `ai/vision`, `ai/classification`, `ai/rag`, `ai/embeddings`, `ai/tests`) importable and testable without starting FastAPI.

See [`docs/architecture.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs/architecture.md), [`docs/api-contract.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs/api-contract.md), [`docs/evaluation-plan.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs/evaluation-plan.md), and [`docs/TEAM_PRESENTATION_GUIDE.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs/TEAM_PRESENTATION_GUIDE.md) for full specifications and viva preparation.

---

## 3. Repository Directory Structure

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
│   ├── vision/                         # Chest X-Ray preprocessing & 64-D features (Owner: Shanks)
│   │   ├── embeddings.py
│   │   ├── model.py
│   │   └── preprocessing.py
│   ├── classification/                 # 14-label classifier & Grad-CAM (Owner: Vineet)
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
├── data/                               # Dataset, Model Weights & Vector Store (Owner: Shreyansh)
│   ├── models/                         # Trained 14-label checkpoint & benchmark metrics
│   │   ├── densenet121_cxr_weights.npz
│   │   └── evaluation_metrics.json
│   ├── processed/                      # Persisted 384-D vector index (chroma_db/)
│   ├── samples/                        # 15 thoracic guidelines & synthetic CXR phantom
│   │   ├── chest_xray_guidelines.json
│   │   └── sample_cxr.png
│   └── uploads/                        # Uploaded scans & generated Grad-CAM PNGs
│
├── notebooks/                          # Exploratory Data Analysis (Owner: Shreyansh)
│   └── 01_exploratory_analysis.ipynb
├── docs/                               # Technical Specifications & Presentation Guides
│   ├── TEAM_PRESENTATION_GUIDE.md      # Professor viva & member-by-member guide
│   ├── architecture.md
│   ├── api-contract.md
│   ├── evaluation-plan.md
│   └── REPOSITORY_AUDIT.md
├── scripts/
│   ├── train_and_evaluate.py           # Trains 14-label classifier & evaluates RAG benchmark
│   └── seed_sample_data.py             # Generates synthetic CXR & seeds real-mode report
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── PROJECT.md
└── README.md
```

---

## 4. Prerequisites

- **Python**: `3.11+` (verified on Python 3.12)
- **Node.js**: `18+` and `npm`
- **Docker & Docker Compose**: Optional (for containerized PostgreSQL + Backend + Frontend execution)

---

## 5. Step-by-Step Local Running Guide

### Step 1: Environment Configuration

Ensure `.env` exists in the repository root (already configured for `AI_MODE=real`):

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue
```

### Step 2: Train Model Checkpoint, Index Vector Store & Seed Sample Data

From the **repository root**:

```powershell
# 1. Install Python dependencies
pip install -r backend/requirements.txt

# 2. Train 14-label classifier head, index 19 literature chunks, and compute benchmark metrics
python scripts/train_and_evaluate.py

# 3. Generate sample Chest X-Ray phantom and seed a real-mode report into SQLite
python scripts/seed_sample_data.py
```

### Step 3: Start the FastAPI Backend (Terminal 1)

```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Backend API Base**: `http://localhost:8000`
- **Swagger OpenAPI UI**: `http://localhost:8000/docs`
- **Health Endpoint**: `http://localhost:8000/health`

### Step 4: Start the Next.js 14 Frontend (Terminal 2)

```powershell
cd frontend
npm install
npm run dev
```

- Open `http://localhost:3000` in your browser.
- Click **"Load Sample CXR"** (or drag-and-drop any `.png` / `.jpg` Chest X-Ray) and click **"Upload & Analyze Chest X-Ray"**.
- Switch between **Original Radiograph** and **Grad-CAM Heatmap** in Panel 2, inspect the 14 pathology scores in Panel 3, review the grounded report and `[chunk_id]` citations in Panel 4, and inspect the 7-stage LangGraph trace in Panel 5.

---

## 6. Quantitative Benchmark Results (`scripts/train_and_evaluate.py`)

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Balanced Accuracy (14 Labels)** | **79.82%** | Mean of Sensitivity & Specificity across 168 held-out test scans |
| **Macro ROC-AUC** | **80.59%** | Area Under the ROC Curve across all 14 NIH ChestX-ray14 labels |
| **Macro Binary Accuracy** | **93.11%** | Overall thresholded binary accuracy across all 14 classes |
| **RAG Indexed Literature Chunks** | **19 chunks** | Chunked from 15 peer-reviewed Fleischner Society & ACR documents |
| **RAG Retrieval Recall@3** | **100.00%** | Top-3 retrieved chunks cover the target pathology across all 14 labels |
| **Citation Groundedness Rate** | **100.00%** | Every generated impression cites verified `[chunk_id]` passages |

---

## 7. API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns backend health, DB status, active `AI_MODE`, benchmark accuracy, and integration readiness |
| `POST` | `/api/upload` | Validates and stores a `PNG` or `JPEG` Chest X-Ray (`multipart/form-data`), persisting a `Scan` row |
| `POST` | `/api/analyze` | Runs preprocessing, 14-label classification, Grad-CAM generation, and 5-node Agentic RAG for `scan_id` |
| `GET` | `/api/report/{report_id}` | Retrieves a persisted `ReportResponse` by its UUID |
| `GET` | `/api/scans/{scan_id}/image` | Streams the stored Chest X-Ray image file |
| `GET` | `/api/scans/{scan_id}/heatmap` | Streams the generated Grad-CAM spatial activation overlay PNG |
| `GET` | `/api/samples/cxr` | Streams `data/samples/sample_cxr.png` for 1-click frontend demo loading |

---

## 8. Automated Testing Instructions

Run all automated tests from the repository root:

```powershell
# 1. Run AI & Backend pytest suites (19 tests covering mock & real modes)
python -m pytest ai/tests/test_ai_pipeline.py backend/tests/test_api.py -v

# 2. Run Frontend unit tests and Next.js production build
cd frontend
node __tests__/dashboard.test.mjs
npm run build
```
