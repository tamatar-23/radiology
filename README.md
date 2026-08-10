# Multimodal Agentic Radiology Assistant MVP

> **Disclaimer**: This repository is a **research and development prototype** designed for architectural validation and engineering development. It is **NOT** a medical diagnostic system and should **NOT** be used for clinical diagnostic purposes.

---

## Project Overview

The **Multimodal Agentic Radiology Assistant** is a 4th-year capstone project that aims to assist radiologists by combining **Medical Vision Models**, **Agentic RAG (Retrieval-Augmented Generation)**, **Vector Database Retrieval**, and **Medical LLM Summarization**.

This initial version provides a **super simple barebones MVP / project skeleton** that defines the system architecture, REST API interfaces, database models, and module boundaries so that six team members can collaborate independently without code collisions.

---

## Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, Lucide Icons
- **Backend**: Python 3.11, FastAPI, Pydantic v2, SQLAlchemy
- **AI & RAG Layer**: PyTorch, MONAI, LangGraph / LangChain, ChromaDB / Qdrant, MedGemma / BiomedCLIP
- **Database**: PostgreSQL (with fallback SQLite for zero-dependency local runs)
- **Containerization**: Docker & Docker Compose

---

## Repository Structure & Team Ownership

```text
radiology-ai/
│
├── frontend/             # Next.js 14 Web Application (Owner: Shreyash)
├── backend/              # FastAPI Application (Owner: Shivam)
├── ai/                   # AI Module Layer
│   ├── rag/              # Agentic RAG & Vector Store (Owner: Gourav)
│   ├── vision/           # Medical Vision & DICOM Preprocessing (Owner: Shanks)
│   ├── classification/   # Disease Classifier & Explainability (Owner: Vineet)
│   └── embeddings/       # Embedding utilities
├── data/                 # Dataset Storage & Samples (Owner: Shreyansh)
│   ├── raw/
│   ├── processed/
│   └── samples/
├── notebooks/            # Exploratory Data Analysis & Model Experiments
├── docs/                 # Research Papers & Architecture Diagrams
├── scripts/              # Data Seeding & Deployment Scripts
├── .env.example          # Environment Variables Template
├── docker-compose.yml    # Docker Multi-Container Configuration
├── README.md             # Developer Quick-Start Guide
└── PROJECT.md            # Comprehensive Architecture Specification
```

---

## Team Responsibilities Matrix

| Team Member | Module / Responsibility | Primary Location |
| :--- | :--- | :--- |
| **Gourav** | Agentic RAG, Vector Database, LLM Integration | [ai/rag/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/rag) |
| **Shanks** | Medical Vision Model, DICOM Preprocessing | [ai/vision/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/vision) |
| **Shreyansh** | Dataset Collection, Data Cleaning & Management | [data/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/data), [notebooks/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/notebooks) |
| **Vineet** | Disease Classification, Training & Evaluation | [ai/classification/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/ai/classification) |
| **Shivam** | Backend, Database Schemas, APIs, Docker | [backend/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/backend), `docker-compose.yml` |
| **Shreyash** | Frontend Dashboard, Documentation, Testing | [frontend/](file:///c:/Users/KIIT0001/Downloads/Projects/radiology/frontend), `PROJECT.md` |

---

## Quick-Start Guide (Local Running)

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional for containerized runs)

---

### 1. Environment Setup

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

---

### 2. Running Backend (FastAPI)

Navigate to backend directory or set `PYTHONPATH`:

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Start FastAPI dev server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at:
- **API Base**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

---

### 3. Running Frontend (Next.js)

In a new terminal window:

```bash
cd frontend

# Install Node dependencies
npm install

# Start Next.js development server
npm run dev
```

Open your browser at `http://localhost:3000`.

---

### 4. Running with Docker Compose

To launch the complete stack (Frontend + Backend + PostgreSQL) in containers:

```bash
docker-compose up --build
```

---

## MVP API Endpoints

- `GET  /health` - Backend status check
- `POST /api/upload` - Upload medical scan (DICOM/PNG)
- `POST /api/analyze` - Trigger AI analysis pipeline (returns mock findings, impression, confidence, and RAG evidence)
- `GET  /api/report/{id}` - Fetch generated report by ID

---

## Current MVP Limitations

- **Mock Data**: `/api/analyze` returns structured mock results for initial workflow testing.
- **No Heavy Model Weights**: Heavy vision models (MedGemma, Rad-DINO) and large datasets (MIMIC-CXR) are excluded from this initial skeleton to ensure instant zero-dependency execution.
- **SQLite Fallback**: Uses local SQLite if PostgreSQL is not active.

---

## Future Roadmap

1. **Phase 1 (Current)**: Monorepo architecture, skeleton endpoints, Next.js UI dashboard, documentation.
2. **Phase 2**: DICOM image preprocessing (MONAI/pydicom) & vision feature extraction integration.
3. **Phase 3**: Vector store indexing (ChromaDB) with medical journal embeddings & RAG retriever.
4. **Phase 4**: Multi-label disease classification fine-tuning & Grad-CAM visual explainability.
5. **Phase 5**: LLM prompt engineering, clinical metric evaluation, and dockerized staging deployment.
