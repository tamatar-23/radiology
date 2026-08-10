# PROJECT.md - Multimodal Agentic Radiology Assistant

> **Single Source of Truth** for 4th-Year Capstone Architecture, Technical Strategy, Team Ownership, and Module Specifications.

---

## 1. Project Overview

The **Multimodal Agentic Radiology Assistant** is an AI-augmented decision support framework designed to process multimodal medical imaging data (CT scans, MRI scans, and X-rays). The system integrates state-of-the-art vision models for visual feature extraction, multi-label disease classification, an **Agentic Retrieval-Augmented Generation (RAG)** pipeline backed by a vector database of medical literature, and a Large Language Model (LLM) to synthesize clinical-grade, evidence-grounded radiology reports with auditability citations.

---

## 2. Problem Statement

Radiologists face unprecedented diagnostic workloads, leading to fatigue, diagnostic delays, and potential oversights. While deep learning models offer high performance in narrow image classification tasks, standalone vision classifiers lack:
1. Contextual integration with established clinical guidelines and medical literature.
2. Natural language explanation capability acceptable in clinical workflows.
3. Verification and citation mechanisms to prevent model hallucinations.

This project addresses these gaps by coupling computer vision backbones with an **agentic RAG workflow** that retrieves verified medical evidence before drafting radiology report impressions.

---

## 3. Objectives

- **Multimodal Ingestion**: Process DICOM and standard medical image formats (CT, MRI, X-Ray) with windowing and HU adjustment.
- **Visual Feature Extraction & Disease Classification**: Extract localized region-of-interest (ROI) features and predict multi-label pathology probabilities.
- **Agentic Knowledge Retrieval**: Query a vector database (ChromaDB/Qdrant) containing medical journals and diagnostic guidelines using dense vector embeddings.
- **Verifiable Report Generation**: Generate structured radiology reports (Findings, Impression, Confidence, Evidence Citations) using an instruction-tuned medical LLM.
- **Interactive Dashboard**: Provide a responsive Next.js web application for image upload, visual state preview, and report review.
- **Modular Monorepo Architecture**: Maintain decoupled modules so all six team members can work in parallel without code conflict.

---

## 4. Proposed Architecture

```text
       Medical Image (CT / MRI / X-Ray)
                      ↓
           DICOM / Preprocessing
                      ↓
             Vision Backbone Model
                      ↓
            Feature Vector Extraction
                      ↓
           Multi-Label Disease Classifier
                      ↓
             Agentic RAG Workflow
                      ↓
               Vector Database
                      ↓
          Medical Knowledge Retrieval
                      ↓
            Medical LLM Synthesizer
                      ↓
        Structured Radiology Report & Evidence
                      ↓
            Next.js Web Dashboard
```

---

## 5. Team Responsibilities

| Team Member | Core Focus Area | Primary Modules & Code Locations |
| :--- | :--- | :--- |
| **Gourav** | Agentic RAG, Vector Database, LLM Integration, AI Pipeline | [ai/rag/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag), `retriever.py`, `vector_store.py`, `agent.py` |
| **Shanks** | Medical Vision Model, Image Processing, DICOM Preprocessing | [ai/vision/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/vision), `preprocessing.py`, `model.py`, `embeddings.py` |
| **Shreyansh** | Dataset Collection, Data Cleaning, Annotation, Dataset Management | [data/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/data), [notebooks/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/notebooks), `01_exploratory_analysis.ipynb` |
| **Vineet** | Disease Classification, Model Training, Explainability, Evaluation | [ai/classification/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/classification), `classifier.py`, `evaluation.py` |
| **Shivam** | Backend Architecture, Database Schemas, REST APIs, Docker | [backend/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/backend), `docker-compose.yml`, `main.py`, `database.py` |
| **Shreyash** | Frontend Dashboard, Documentation, Literature Review, Testing | [frontend/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/frontend), `PROJECT.md`, `README.md`, UI components |

---

## 6. Technology Decisions

- **Next.js & TypeScript**: Chosen for server-rendered performance, typed safety, and modular component lifecycle.
- **FastAPI (Python)**: High-performance asynchronous REST API framework with automatic OpenAPI documentation and Pydantic validation.
- **PyTorch & MONAI**: De-facto industry standard deep learning frameworks for medical image processing, DICOM transformations, and model execution.
- **LangGraph / LangChain**: Framework for constructing stateful, multi-step agentic workflows and tool-calling loops.
- **ChromaDB / Qdrant**: Lightweight vector databases optimized for similarity search over high-dimensional text and vision embeddings.
- **PostgreSQL & SQLAlchemy**: Relational persistence layer for scan metadata, user sessions, and generated reports.
- **Docker & Docker Compose**: Unified containerization ensuring identical environments across all developer machines.

---

## 7. AI/ML Strategy

We will **NOT** train foundation models from scratch due to compute requirements. Instead, we adopt a transfer learning and adaptation strategy:
- Fine-tune domain-specific vision backbones on target radiology datasets.
- Utilize pretrained multimodal medical models:
  - **MedGemma**: Multimodal medical language-vision model family.
  - **BiomedCLIP**: Contrastive vision-language model trained on biomedical image-text pairs.
  - **Rad-DINO**: Self-supervised Vision Transformer for chest X-ray representation.
  - **LLaVA-Med**: Instruction-tuned biomedical vision-language model.

---

## 8. Dataset Strategy

### Candidate Datasets
- **TCIA (The Cancer Imaging Archive)**: Multi-modality cancer imaging.
- **MIMIC-CXR**: Large dataset of chest radiographs with free-text reports.
- **RSNA Datasets**: Pneumonia Detection and Pulmonary Embolism challenges.
- **BraTS**: Brain Tumor Segmentation MRIs.
- **MRNet**: Knee MRI scan dataset.
- **NIH ChestX-ray14**: 112,120 chest X-ray images with 14 disease labels.

### Dataset Tracking Table (To be populated by Shreyansh)

| Field | Selected Option / Record |
| :--- | :--- |
| **Dataset Name** | TBD (e.g. NIH ChestX-ray14 / MIMIC-CXR) |
| **Version** | v1.0 |
| **License** | Open Access / PhysioNet Credentialed |
| **Number of Images** | TBD |
| **Number of Patients** | TBD |
| **Target Labels** | Atelectasis, Cardiomegaly, Effusion, Pneumonia, etc. |
| **Train/Val/Test Split** | 70% / 15% / 15% |
| **Preprocessing Operations** | Resizing (512x512), Histogram Equalization, Normalization |

---

## 9. RAG Strategy (Agentic Retrieval-Augmented Generation)

- **Corpus Data**: Peer-reviewed radiology literature, Fleischner Society guidelines, ACR Appropriateness Criteria, and anonymized reference report templates.
- **Embedding Model**: `BAAI/bge-large-en-v1.5` or `MedCPT` biomedical embeddings.
- **Chunking Strategy**: Recursive character splitting with 500-character chunks and 50-character overlap.
- **Vector Database**: ChromaDB (local/dev) or Qdrant.
- **Retrieval Strategy**: Dense similarity search with Top-K = 3.
- **Reranking**: Cross-encoder reranking to prioritize clinically relevant passages.
- **Agent Workflow**: LangGraph agent evaluates visual classifier outputs, formulates retrieval queries, inspects retrieved passages, and prompts LLM synthesizer.
- **LLM Engine**: `MedGemma` / `GPT-4o-mini` / `Ollama-Llama3`.
- **Evidence Strategy**: Every report impression must contain explicit inline citations referencing the source passage.

---

## 10. Evaluation Strategy

### Classification & Computer Vision Metrics
- **Accuracy, Precision, Recall, F1-Score**: Evaluated per pathology label.
- **ROC-AUC & PR-AUC**: Area under receiver operating characteristic curve across all 14 disease labels.

### RAG & Report Generation Metrics
- **Retrieval Precision & Recall @ K**: Measuring context retrieval accuracy.
- **ROUGE-L & BLEU-4**: Measuring n-gram overlap between generated and ground-truth radiology reports.
- **Medical Hallucination Rate**: Expert audit checking for ungrounded claims.

---

## 11. Explainability

- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Visualizing spatial attention overlays for disease classification predictions.
- **Attention Maps**: Displaying cross-attention weights between image regions and report tokens.
- **Retrieved Evidence Cards**: Showing exact medical literature excerpts used to justify the report impression.
- **Confidence Calibration Score**: Calibrated probability outputs reflecting system certainty.

---

## 12. API Design

### Base URL: `http://localhost:8000`

#### 1. Health Status Check
- **`GET /health`**
- **Response**:
  ```json
  {
    "status": "healthy",
    "service": "radiology-backend",
    "timestamp": "2026-08-10T19:46:52Z"
  }
  ```

#### 2. Scan Upload Endpoint
- **`POST /api/upload`**
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (DICOM, PNG, JPEG)
- **Response**:
  ```json
  {
    "scan_id": "scan_a1b2c3d4",
    "filename": "chest_ct_scan.dcm",
    "file_type": "application/dicom",
    "message": "Scan uploaded successfully"
  }
  ```

#### 3. Scan Analysis Endpoint
- **`POST /api/analyze`**
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "scan_id": "scan_a1b2c3d4",
    "file_name": "chest_ct_scan.dcm",
    "notes": "Patient presents with shortness of breath"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "scan_id": "scan_a1b2c3d4",
    "findings": [
      "Subtle ground-glass opacity in right lower lobe."
    ],
    "impression": "Mock radiology impression: Findings suggest mild atelectasis or early infiltrates. Correlate clinically.",
    "confidence": 0.85,
    "evidence": [
      {
        "source": "Mock Medical Source - Pulmonary Criteria",
        "text": "Ground glass opacity in lower pulmonary zone typically indicates focal infiltration."
      }
    ]
  }
  ```

#### 4. Report Retrieval Endpoint
- **`GET /api/report/{id}`**
- **Response**: `ReportResponse` JSON object with findings, impression, confidence, and timestamp.

---

## 13. Project Structure

```text
radiology-ai/
├── frontend/                 # Next.js 14 Web Application (Shreyash)
│   ├── app/                  # App Router pages and layout
│   ├── public/               # Static assets
│   ├── Dockerfile            # Frontend container file
│   └── package.json          # Node dependencies
│
├── backend/                  # FastAPI Python Application (Shivam)
│   ├── app/
│   │   ├── api/              # REST API endpoints
│   │   ├── db/               # SQLAlchemy models and DB session logic
│   │   ├── schemas/          # Pydantic data schemas
│   │   └── main.py           # FastAPI entrypoint
│   ├── Dockerfile            # Backend container file
│   └── requirements.txt      # Python dependencies
│
├── ai/                       # AI Pipeline Layer
│   ├── rag/                  # Agentic RAG & Vector Store (Gourav)
│   │   ├── retriever.py
│   │   ├── vector_store.py
│   │   └── agent.py
│   ├── vision/               # Medical Vision & DICOM Parser (Shanks)
│   │   ├── preprocessing.py
│   │   ├── model.py
│   │   └── embeddings.py
│   ├── classification/       # Disease Classifier & Grad-CAM (Vineet)
│   │   ├── classifier.py
│   │   └── evaluation.py
│   ├── embeddings/           # Shared embedding helpers
│   └── config.py             # Central AI hyperparameters
│
├── data/                     # Data Storage (Shreyansh)
│   ├── raw/                  # Raw DICOM/image datasets (git-ignored)
│   ├── processed/            # Preprocessed tensors (git-ignored)
│   └── samples/              # Sample images for local testing
│
├── notebooks/                # Jupyter Notebooks for EDA & Experiments
├── docs/                     # Additional Research & Papers
├── scripts/                  # Utility scripts (seed data, model test)
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusion rules
├── docker-compose.yml        # Multi-container orchestration
├── README.md                 # Developer quick-start guide
└── PROJECT.md                # Single source of truth documentation
```

---

## 14. Development Decisions Log

```text
Decision: Adopt Monorepo Architecture with decoupled AI subpackages
Date: 2026-08-10
Reason: Enables 6 team members to write code in parallel without blocking each other.
Alternatives considered: Polyrepo (separate git repos per module).
Final choice: Monorepo with top-level ai/, backend/, frontend/ folders.
```

---

## 15. Open Questions / TODO

- [ ] Select exact target dataset (NIH ChestX-ray14 vs MIMIC-CXR vs RSNA).
- [ ] Determine local LLM vs API model selection (Ollama MedGemma vs OpenAI GPT-4o-mini).
- [ ] Implement DICOM header parser using `pydicom` / `monai`.
- [ ] Set up vector database index (ChromaDB local instance vs Qdrant container).
- [ ] Implement Grad-CAM heatmap generation module in `ai/classification/evaluation.py`.
