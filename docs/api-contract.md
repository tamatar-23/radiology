# REST API Contract Specification (`docs/api-contract.md`)

**Base URL (Local)**: `http://localhost:8000`  
**Interactive OpenAPI Docs**: `http://localhost:8000/docs`

---

## 1. `GET /health`

Checks backend health, database connectivity, active `AI_MODE`, benchmark accuracy, and integration readiness.

### Response (`200 OK` in `AI_MODE=real`)
```json
{
  "status": "healthy",
  "service": "radiology-backend",
  "database_status": "connected (sqlite)",
  "ai_mode": "real",
  "modality_scope": "Chest X-Ray (DX)",
  "integrations": {
    "vision_model": "densenet121-cxr-calibrated (ready, 79.8% benchmark acc)",
    "explainability": "Grad-CAM spatial activation overlay (ready)",
    "rag_vector_store": "ChromaDB (19 chunks indexed)",
    "llm_synthesizer": "gemini (gemini-2.0-flash)"
  },
  "timestamp": "2026-11-02T17:45:00Z"
}
```

---

## 2. `POST /api/upload`

Uploads and validates a Chest X-Ray image (`PNG` or `JPEG/JPG`). Binary image headers are verified with `Pillow`—the server never trusts the client filename or declared MIME type alone.

### Request
- **Content-Type**: `multipart/form-data`
- **Form Field**: `file` (Binary PNG or JPEG file, max `15 MB` by default)

### Response (`200 OK`)
```json
{
  "scan_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "sample_cxr.png",
  "media_type": "image/png",
  "status": "uploaded",
  "file_type": "image/png",
  "file_size_bytes": 42180,
  "image_metadata": {
    "width": 512,
    "height": 512,
    "channels": 1,
    "format": "PNG",
    "color_mode": "L",
    "mean_intensity": 0.4412,
    "std_intensity": 0.2195,
    "modality": "DX"
  },
  "message": "Scan validated and stored successfully."
}
```

---

## 3. `POST /api/analyze`

Executes Chest X-Ray preprocessing, 64-D multiscale feature extraction, 14-label abnormality classification, Grad-CAM spatial heatmap generation, and the 5-stage LangGraph Agentic RAG pipeline for a previously uploaded scan, persisting the resulting report in SQLAlchemy.

### Request (`application/json`)
```json
{
  "scan_id": "550e8400-e29b-41d4-a716-446655440000",
  "notes": "Evaluate right lower lung zone opacity for basal atelectasis or pleural effusion"
}
```

### Response (`200 OK` in `AI_MODE=real`)
```json
{
  "report_id": "238600ad-d5c4-4737-8ab6-342ca1cf4558",
  "id": "238600ad-d5c4-4737-8ab6-342ca1cf4558",
  "scan_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "mode": "real",
  "disclaimer": "RESEARCH PROTOTYPE: Output is generated for academic research evaluation only and must not be used for clinical diagnosis.",
  "model_name": "densenet121-cxr-calibrated",
  "predictions": [
    { "label": "Atelectasis", "score": 0.84 },
    { "label": "Effusion", "score": 0.72 },
    { "label": "Consolidation", "score": 0.61 }
  ],
  "heatmap_available": true,
  "heatmap_url": "/api/scans/550e8400-e29b-41d4-a716-446655440000/heatmap",
  "explainability_message": "Generated spatial Grad-CAM activation overlay for top candidate class 'Atelectasis'.",
  "findings": [
    "CLINICAL INDICATION & CORRELATION: Evaluation performed for 'Evaluate right lower lung zone opacity for basal atelectasis or pleural effusion'...",
    "PRIMARY RADIOLOGIC PATTERN — ATELECTASIS (Model Score: 0.84): Elevated spatial activation detected for Atelectasis. Per Fleischner Society: Glossary of Terms for Thoracic Imaging — Atelectasis [fleischner-glossary-2008-atelectasis#chunk-0]..."
  ],
  "impression": "GROUNDED RADIOLOGY SYNTHESIS: Multi-label Chest X-Ray analysis demonstrates elevated findings for Atelectasis (0.84), Effusion (0.72), Consolidation (0.61). Findings are corroborated by retrieved thoracic imaging criteria [fleischner-glossary-2008-atelectasis#chunk-0], [acr-chest-radiography-pleural-effusion#chunk-0]...",
  "evidence_status": "sufficient",
  "evidence": [
    {
      "evidence_id": "f4c1a9c2-91f2-5908-8f30-6b4d93f1d821",
      "document_title": "Fleischner Society: Glossary of Terms for Thoracic Imaging — Atelectasis",
      "source_url": "https://doi.org/10.1148/radiol.2462070712",
      "passage": "Atelectasis refers to reduced inflation of all or part of the lung...",
      "chunk_id": "fleischner-glossary-2008-atelectasis#chunk-0",
      "relevance_score": 0.96
    }
  ],
  "agent_trace": [
    {
      "stage_id": "image_preprocessing",
      "stage_name": "Image Preprocessing",
      "status": "completed",
      "mode": "real",
      "duration_ms": 5.8,
      "details": "Validated PNG Chest X-Ray (512x512), converted to grayscale and normalized to 224x224."
    },
    {
      "stage_id": "vision_inference",
      "stage_name": "Vision Inference",
      "status": "completed",
      "mode": "real",
      "duration_ms": 18.4,
      "details": "Executed densenet121-cxr-calibrated (real mode)... Grad-CAM: available."
    }
  ],
  "image_metadata": {
    "width": 512,
    "height": 512,
    "channels": 1,
    "format": "PNG",
    "color_mode": "L",
    "mean_intensity": 0.4412,
    "std_intensity": 0.2195,
    "modality": "DX"
  },
  "confidence": 0.84,
  "created_at": "2026-11-02T17:45:05Z"
}
```

---

## 4. `GET /api/report/{report_id}`

Fetches a previously generated and persisted report by its `report_id`.

---

## 5. `GET /api/scans/{scan_id}/image`

Streams the persisted Chest X-Ray image file (`image/png` or `image/jpeg`) for the given `scan_id`.

---

## 6. `GET /api/scans/{scan_id}/heatmap`

Streams the generated Grad-CAM spatial activation overlay image (`image/png`) for a scan analyzed in `AI_MODE=real`. Returns `404 Not Found` if no heatmap exists for the given `scan_id`.

---

## 7. `GET /api/samples/cxr`

Streams the default Chest X-Ray sample image (`data/samples/dataset-card.jpeg`, `image/jpeg`, with fallback to `data/samples/sample_cxr.png`) so the frontend preloads and analyzes the default radiograph out of the box.
