# PROJECT.md — Multimodal Agentic Radiology Assistant

> **Single Source of Truth** for our 4th-Year B.Tech Computer Science Capstone Project: Objective, MVP Scope, Architecture, Module Responsibilities, Dataset/AI/RAG/Evaluation Strategies, Benchmark Results, and Team Ownership.

---

## 1. Final Project Objective

Develop a modular, research-oriented **Chest X-Ray Analysis and Agentic RAG Assistant** that integrates:
1. Medical image validation and Chest X-Ray grayscale preprocessing (`Pillow` / `NumPy` / `SciPy`).
2. 64-dimensional multiscale radiographic feature extraction and calibrated 14-label thoracic abnormality classification (`densenet121-cxr-calibrated`).
3. Genuine **Grad-CAM spatial explainability** generating blended thermal activation overlay maps (`/api/scans/{scan_id}/heatmap`).
4. A 5-node stateful **LangGraph Agentic RAG** pipeline (`Query Planner` → `Evidence Retriever` → `Reranker` → `Evidence Checker` → `Report Generator`).
5. Verifiable medical literature citation tracking over all 14 NIH ChestX-ray14 pathologies (15 peer-reviewed Fleischner Society & ACR reference documents).
6. Multi-provider LLM report synthesis (`Gemini` / `Groq` / `OpenAI` / `Citation-Grounded Synthesizer`).
7. An interactive Next.js 14 research dashboard with 1-click sample loading, Grad-CAM heatmap toggle, and full 7-stage execution trace visibility.

> **Non-Clinical Mandate**: This system is an academic research prototype, **NOT** a certified clinical diagnostic system. It must never claim to replace radiologists or independently confirm diagnoses.

---

## 2. Problem Statement

Standalone deep learning classifiers for chest radiographs output raw abnormality scores without linking predictions to established thoracic imaging criteria, explaining spatial model attention, or verifying whether generated text summaries are grounded in peer-reviewed radiology literature. Conversely, general-purpose LLMs are prone to hallucinating clinical findings or fabricating references when prompted without structured retrieval and verification gates.

---

## 3. Research Question

*How effectively can a multi-label Chest X-Ray vision classifier coupled with a multi-stage, verification-gated LangGraph Agentic RAG pipeline synthesize structured radiology findings and impressions while maintaining strict citation traceability and surfacing insufficient evidence?*

---

## 4. Implemented Scope & Benchmark Achievements

- **Completed in Working Prototype (`AI_MODE=real`)**:
  - Chest X-Ray (`PNG`, `JPEG/JPG`) binary header validation, grayscale conversion, $224 \times 224$ resizing, intensity normalization $[0, 1]$, and SQLite/PostgreSQL metadata persistence.
  - Multi-label abnormality prediction across all **14 standard NIH ChestX-ray14 pathology labels** (`Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia`), achieving **79.82% Balanced Accuracy** and **80.59% Macro ROC-AUC** on the 168-scan held-out test split.
  - Real **Grad-CAM spatial activation heatmap** computation and PNG overlay serving (`GET /api/scans/{scan_id}/heatmap`), while honestly reporting unavailability in `AI_MODE=mock`.
  - 5-node LangGraph Agentic RAG workflow backed by 384-D `BiomedicalDenseEmbedder` vectors (`data/processed/chroma_db`) across 19 chunks from 15 thoracic guideline documents (**100% Recall@3**, **100% Citation Groundedness**).
  - Multi-provider LLM integration (`GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`, plus built-in offline `local_grounded` synthesis).
  - Interactive Next.js 14 dashboard preloaded with `data/samples/dataset-card.jpeg` (`GET /api/samples/cxr`), `Original` vs. `Grad-CAM` viewer tabs, vertical abnormality score bars, structured findings/impression with inline citations, and a bottom vertical step-flow 7-stage execution trace.

---

## 5. Technical Architecture

```text
Chest X-Ray Upload (PNG / JPEG) or 1-Click Sample CXR
        │
        ▼
[Stage 1] ChestXRayPreprocessor (ai/vision/preprocessing.py)
        │  Validates binary format, converts to grayscale, normalizes [0, 1] (224x224)
        ▼
[Stage 2] DiseaseClassifier & RealGradCAMProvider (ai/vision/model.py, ai/classification/)
        │  Extracts 64-D multiscale features -> 14-label sigmoid scores + Grad-CAM overlay PNG
        ▼
[Stage 3] Query Planner Node (ai/rag/agent.py)
        │  Formulates targeted thoracic literature queries from elevated labels & clinical notes
        ▼
[Stage 4] Medical Evidence Retriever Node (ai/rag/retriever.py, ai/rag/vector_store.py)
        │  Executes 384-D dense cosine similarity search over 19 indexed guideline chunks
        ▼
[Stage 5] Evidence Reranker Node (ai/rag/reranker.py)
        │  Reranks candidates via hybrid dense + TF-IDF bi-gram cosine similarity & lexical boost
        ▼
[Stage 6] Evidence Checker Node (ai/rag/agent.py)
        │  Verifies label coverage -> routes as "sufficient", "insufficient", or "unverified_claims"
        ▼
[Stage 7] Report Generator Node (ai/rag/providers.py)
        │  Synthesizes structured Findings & Impression via Gemini / Citation-Grounded Synthesizer
        ▼
Persisted in SQLAlchemy (Scan, Report, Finding) & Rendered in Next.js 14 Dashboard
```

---

## 6. Module Responsibilities & Team Ownership

| Team Member | Role & Ownership Scope | Primary Modules & Files |
| :--- | :--- | :--- |
| **Gourav** | Agentic RAG, Vector DB, 384-D Biomedical Embeddings, Retrieval, Hybrid Reranking, Evidence Verification, LangGraph Orchestration, Gemini/Grounded LLM Integration | [`ai/rag/agent.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/agent.py), [`ai/rag/providers.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/providers.py), [`ai/rag/retriever.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/retriever.py), [`ai/rag/reranker.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/reranker.py), [`ai/rag/ingestion.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/ingestion.py), [`ai/rag/vector_store.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/vector_store.py) |
| **Shanks** | Medical Image Ingestion, Binary Validation, CXR Preprocessing, 64-D Multiscale Radiographic Feature Extractor, Vision Model Adapters | [`ai/vision/preprocessing.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/vision/preprocessing.py), [`ai/vision/model.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/vision/model.py), [`ai/vision/embeddings.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/vision/embeddings.py) |
| **Shreyansh** | Dataset Collection & Synthesis, 15-Document Thoracic Guideline Corpus, Stratified Train/Test Splits, Exploratory Data Analysis | [`data/samples/chest_xray_guidelines.json`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/data/samples/chest_xray_guidelines.json), [`notebooks/01_exploratory_analysis.ipynb`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/notebooks/01_exploratory_analysis.ipynb), [`scripts/train_and_evaluate.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/scripts/train_and_evaluate.py) |
| **Vineet** | 14-Label Disease Classification, Regularized Logistic Head Training, Grad-CAM Spatial Heatmap Generation, Quantitative Evaluation Metrics | [`ai/classification/classifier.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/classification/classifier.py), [`ai/classification/evaluation.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/classification/evaluation.py), [`scripts/train_and_evaluate.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/scripts/train_and_evaluate.py) |
| **Shivam** | FastAPI Backend, SQLite/PostgreSQL Persistence, SQLAlchemy Models, REST APIs (`/upload`, `/analyze`, `/heatmap`, `/samples/cxr`), Docker Integration | [`backend/app/`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/backend/app), [`docker-compose.yml`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docker-compose.yml), [`backend/Dockerfile`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/backend/Dockerfile), [`scripts/seed_sample_data.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/scripts/seed_sample_data.py) |
| **Shreyash** | Next.js 14 Frontend Dashboard, Typed API Client, Grad-CAM Viewer Toggle, 1-Click Sample Loader, Documentation, UI Testing | [`frontend/`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/frontend), [`README.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/README.md), [`PROJECT.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/PROJECT.md), [`docs/`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs) |

---

## 7. Quantitative Benchmark Summary (`data/models/evaluation_metrics.json`)

- **Dataset Split**: 308 training radiographs / 168 held-out test radiographs across all 14 NIH ChestX-ray14 labels.
- **Balanced Accuracy**: **79.82%**
- **Macro ROC-AUC**: **80.59%**
- **Macro Binary Accuracy**: **93.11%**
- **RAG Retrieval Recall@3**: **100.00%** (19 chunks indexed across 15 thoracic guideline documents)
- **Citation Groundedness Rate**: **100.00%**

See [`docs/TEAM_PRESENTATION_GUIDE.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs/TEAM_PRESENTATION_GUIDE.md) for the complete viva defense guide.
