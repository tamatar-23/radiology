# PROJECT.md — Multimodal Agentic Radiology Assistant

> **Single Source of Truth** for our 4th-Year B.Tech Computer Science Capstone Project: Objective, MVP Scope, Architecture, Module Responsibilities, Dataset/AI/RAG/Evaluation Strategies, and Team Ownership.

---

## 1. Final Project Objective

Develop a modular, research-oriented **Chest X-Ray Analysis and Agentic RAG Assistant** that integrates:
1. Medical image validation and Chest X-Ray grayscale preprocessing (`Pillow` / `NumPy` / `MONAI`).
2. Pretrained medical vision model adapters (`DenseNet121`, `Rad-DINO`, `BiomedCLIP`) for multi-label thoracic abnormality classification.
3. Honest Grad-CAM spatial explainability interfaces.
4. A 5-node stateful **LangGraph Agentic RAG** pipeline (`Query Planner` → `Evidence Retriever` → `Reranker` → `Evidence Checker` → `Report Generator`).
5. Verifiable medical literature citation tracking.
6. An interactive Next.js 14 research dashboard with full execution trace visibility.

> **Non-Clinical Mandate**: This system is an academic research prototype, **NOT** a certified clinical diagnostic system. It must never claim to replace radiologists or independently confirm diagnoses.

---

## 2. Problem Statement

Standalone deep learning classifiers for chest radiographs output raw abnormality scores without linking predictions to established thoracic imaging criteria, explaining spatial model attention, or verifying whether generated text summaries are grounded in peer-reviewed radiology literature. Conversely, general-purpose LLMs are prone to hallucinating clinical findings or fabricating references when prompted without structured retrieval and verification gates.

---

## 3. Research Question

*How effectively can a multi-label Chest X-Ray vision classifier coupled with a multi-stage, verification-gated LangGraph Agentic RAG pipeline synthesize structured radiology findings and impressions while maintaining strict citation traceability and surfacing insufficient evidence?*

---

## 4. MVP Scope (Chest X-Ray Focus)

- **Included in MVP**:
  - Chest X-Ray (`PNG`, `JPEG/JPG`) upload validation, grayscale conversion, resizing, intensity normalization, and SQLite/PostgreSQL metadata persistence.
  - Multi-label abnormality prediction across the 14 standard NIH ChestX-ray14 pathology labels (`Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia`).
  - Grad-CAM explainability interface (honestly reporting unavailability in mock mode rather than applying fake image filters).
  - 5-node LangGraph Agentic RAG workflow with conditional routing for insufficient or unverified evidence.
  - Dual execution mode (`AI_MODE=mock` default vs. `AI_MODE=real`).
  - Interactive Next.js 14 dashboard displaying image metadata, abnormality scores, structured reports, literature citations, and the 7-stage pipeline execution trace.
- **Deferred to Future Extensions**:
  - Volumetric 3D CT and MRI modalities.
  - Mandatory raw DICOM pixel windowing in the upload endpoint (interface provided in `DICOMPreprocessor`).

---

## 5. Technical Architecture

```text
Chest X-Ray Upload (PNG / JPEG)
        │
        ▼
[Stage 1] ChestXRayPreprocessor (ai/vision/preprocessing.py)
        │  Validates binary format, converts to grayscale, normalizes [0, 1]
        ▼
[Stage 2] DiseaseClassifier & ModelEvaluator (ai/classification/)
        │  Produces 14-label abnormality scores + Grad-CAM availability check
        ▼
[Stage 3] Query Planner Node (ai/rag/agent.py)
        │  Formulates thoracic literature queries from top predicted labels & notes
        ▼
[Stage 4] Medical Evidence Retriever Node (ai/rag/retriever.py)
        │  Retrieves candidate passages from guideline corpus / ChromaDB
        ▼
[Stage 5] Evidence Reranker Node (ai/rag/reranker.py)
        │  Scores and filters candidate passages by clinical query relevance
        ▼
[Stage 6] Evidence Checker Node (ai/rag/agent.py)
        │  Verifies label coverage -> routes as "sufficient", "insufficient", or "unverified_claims"
        ▼
[Stage 7] Report Generator Node (ai/rag/providers.py)
        │  Synthesizes structured Findings & Impression with inline [chunk_id] citations
        ▼
Persisted in SQLAlchemy (Scan, Report, Finding) & Rendered in Next.js 14 Dashboard
```

---

## 6. Module Responsibilities & Team Ownership

| Team Member | Role & Ownership Scope | Primary Modules & Files |
| :--- | :--- | :--- |
| **Gourav** | Agentic RAG, Vector DB, Retrieval, Reranking, Evidence Verification, LangGraph Orchestration, LLM Integration | [`ai/rag/agent.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/agent.py), [`ai/rag/providers.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/providers.py), [`ai/rag/retriever.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/retriever.py), [`ai/rag/reranker.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/reranker.py), [`ai/rag/ingestion.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/ingestion.py), [`ai/rag/vector_store.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag/vector_store.py) |
| **Shanks** | Medical Image Ingestion, DICOM & CXR Preprocessing, Image Transformations, Vision Model Adapters | [`ai/vision/preprocessing.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/vision/preprocessing.py), [`ai/vision/model.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/vision/model.py), [`ai/vision/embeddings.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/vision/embeddings.py) |
| **Shreyansh** | Dataset Collection, Cleaning, Data Management, Sample Corpus, Exploratory Data Analysis | [`data/`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/data), [`notebooks/01_exploratory_analysis.ipynb`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/notebooks/01_exploratory_analysis.ipynb), [`data/samples/chest_xray_guidelines.json`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/data/samples/chest_xray_guidelines.json) |
| **Vineet** | Multi-Label Disease Classification, Model Training & Evaluation, Grad-CAM Explainability, Metrics | [`ai/classification/classifier.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/classification/classifier.py), [`ai/classification/evaluation.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/classification/evaluation.py) |
| **Shivam** | FastAPI Backend, PostgreSQL/SQLite Persistence, SQLAlchemy Models, REST APIs, Docker Integration | [`backend/app/`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/backend/app), [`docker-compose.yml`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docker-compose.yml), [`backend/Dockerfile`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/backend/Dockerfile), [`scripts/seed_sample_data.py`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/scripts/seed_sample_data.py) |
| **Shreyash** | Next.js 14 Frontend Dashboard, Typed API Client, UI Components, Documentation, UI Testing | [`frontend/`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/frontend), [`README.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/README.md), [`PROJECT.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/PROJECT.md), [`docs/`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs) |

---

## 7. Dataset Strategy (Owner: Shreyansh)

- **Primary Target Datasets**:
  - **NIH ChestX-ray14**: 112,120 frontal-view chest radiographs with 14 pathology labels.
  - **CheXpert / MIMIC-CXR-JPG**: Frontal and lateral chest radiographs with structured label extractors and reference reports.
- **Data Hygiene Rules**:
  - Raw datasets (`data/raw/`) and processed tensors (`data/processed/`) are strictly git-ignored.
  - Demo fixtures (`data/samples/sample_cxr.png`) use synthetic non-patient phantoms only.
  - Patient-level stratified train/validation/test splits (`70% / 15% / 15%`) prevent patient overlap across splits.

---

## 8. AI/ML Vision Strategy (Owners: Shanks & Vineet)

- **Transfer Learning Backbone Candidates**:
  - **DenseNet121** (`torchvision` / `MONAI` / `TorchXRayVision` weights pretrained on chest radiographs).
  - **Rad-DINO** (`microsoft/rad-dino` self-supervised chest X-ray Vision Transformer).
  - **BiomedCLIP** (contrastive biomedical vision-language encoder).
- **Adapter Pattern**:
  - `VisionModelAdapter` (`ai/vision/model.py`) and `BasePredictionProvider` (`ai/common/interfaces.py`) decouple model architectures from the backend API and RAG workflow.
- **Grad-CAM Explainability**:
  - `BaseExplainabilityProvider` (`ai/classification/evaluation.py`) separates spatial activation map computation from classification scores and never fabricates synthetic heatmaps in mock mode.

---

## 9. Agentic RAG Strategy (Owner: Gourav)

- **Knowledge Corpus**: Open-access thoracic radiology glossary definitions and appropriateness guidelines (Fleischner Society Glossary of Terms for Thoracic Imaging, ACR Appropriateness Criteria).
- **Chunking**: Overlapping character chunking (`chunk_size=500`, `chunk_overlap=50`) via `MedicalDocumentIngestor` (`ai/rag/ingestion.py`).
- **Embeddings & Vector Store**: Configurable biomedical embedding model (`BAAI/bge-large-en-v1.5` or `MedCPT`) indexed into **ChromaDB** (`MedicalVectorStore` in `ai/rag/vector_store.py`).
- **Verification Gate**: `verify_evidence_node` inspects whether retrieved passages cover the predicted abnormalities and flags reports when evidence is insufficient or claims are unverified.

---

## 10. Evaluation Strategy (Owners: Vineet & Gourav)

See [`docs/evaluation-plan.md`](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/docs/evaluation-plan.md) for the complete evaluation protocol:
- **Classification**: Per-label ROC-AUC, PR-AUC, thresholded Precision, Recall, F1-Score, and Expected Calibration Error (ECE).
- **RAG & Reporting**: Retrieval Precision@3, Recall@3, citation groundedness verification, and stage-level execution latency (`duration_ms`).

---

## 11. Development Phases

- **Phase 1 (Completed in Boilerplate)**: Repository audit, finalized monorepo architecture, typed Pydantic/TypeScript contracts, SQLite/PostgreSQL persistence, `AI_MODE=mock` vs `real` separation, 5-node LangGraph RAG workflow, modular Next.js 14 dashboard, and automated tests.
- **Phase 2 (Next — Shanks & Vineet)**: Integrate pretrained DenseNet121 / Rad-DINO Chest X-Ray checkpoint in `RealPredictionProvider` and convolutional hook Grad-CAM in `RealGradCAMProvider`.
- **Phase 3 (Next — Gourav & Shreyansh)**: Expand thoracic literature corpus, index embeddings into persistent ChromaDB via `MedicalVectorStore`, and connect `RealReportGenerator` to MedGemma / OpenAI / Ollama.
- **Phase 4 (Next — All Team Members)**: Execute quantitative benchmark evaluation across NIH ChestX-ray14 test split and RAG faithfulness test cases.
