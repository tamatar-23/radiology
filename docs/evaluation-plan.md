# Evaluation Plan & Benchmark Results (`docs/evaluation-plan.md`)

**Project**: Multimodal Agentic Radiology Assistant (Chest X-Ray Prototype)  
**Primary Owners**: Vineet (Classification & Explainability Evaluation), Gourav (RAG & Report Evaluation), Shreyansh (Dataset Curation & Splits)

---

## 1. Executed Benchmark Results (`data/models/evaluation_metrics.json`)

Reproducible via `python scripts/train_and_evaluate.py`:

| Category | Metric | Value | Notes |
| :--- | :--- | :--- | :--- |
| **Dataset Split** | Training / Held-Out Test Scans | `308 train / 168 test` | Stratified across all 14 NIH ChestX-ray14 pathologies |
| **Vision Classification** | **Balanced Accuracy (14 Classes)** | **79.82%** | $\frac{1}{2}(\text{Sensitivity} + \text{Specificity})$ across positive & negative test cases |
| **Vision Classification** | **Macro ROC-AUC** | **80.59%** | Threshold-independent area under ROC curve across 14 classes |
| **Vision Classification** | **Macro Binary Accuracy** | **93.11%** | Overall binary accuracy across 14 independent sigmoid heads |
| **Vision Classification** | **Macro Recall / Sensitivity** | **64.07%** | Evaluated under realistic exposure gamma, blur & anatomical noise |
| **Agentic RAG** | **Indexed Literature Chunks** | **19 chunks** | Chunked (`500` chars, `50` overlap) from 15 thoracic guideline docs |
| **Agentic RAG** | **Retrieval Recall@3 (14 Classes)** | **100.00%** | Every pathology retrieves its ground-truth Fleischner/ACR guideline in Top-3 |
| **Agentic RAG** | **Citation Groundedness Rate** | **100.00%** | 100% of generated impressions contain verified inline `[chunk_id]` citations |

---

## 2. Per-Label Classification Breakdown (168 Held-Out Test Scans)

| Pathology Label | Binary Accuracy | ROC-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Effusion** | 93.45% | 97.03% | 56.52% | 92.86% | 70.27% |
| **Fibrosis** | 92.26% | 93.59% | 47.37% | 75.00% | 58.06% |
| **Nodule** | 97.02% | 89.64% | 76.92% | 83.33% | 80.00% |
| **Consolidation** | 92.86% | 82.07% | 75.00% | 50.00% | 60.00% |
| **Pneumothorax** | 95.24% | 81.14% | 66.67% | 66.67% | 66.67% |
| **Emphysema** | 97.02% | 79.65% | 81.82% | 75.00% | 78.26% |
| **Mass** | 94.64% | 79.54% | 80.00% | 33.33% | 47.06% |
| **Pneumonia** | 91.67% | 79.36% | 55.56% | 62.50% | 58.82% |
| **Hernia** | 92.26% | 77.56% | 46.67% | 58.33% | 51.85% |
| **Infiltration** | 93.45% | 77.51% | 53.33% | 66.67% | 59.26% |
| **Edema** | 89.29% | 76.48% | 38.89% | 50.00% | 43.75% |
| **Pleural_Thickening** | 94.05% | 74.68% | 57.14% | 66.67% | 61.54% |
| **Atelectasis** | 87.50% | 71.42% | 30.43% | 58.33% | 40.00% |
| **Cardiomegaly** | 92.86% | 68.59% | 50.00% | 58.33% | 53.85% |

---

## 3. Agentic RAG & Report Synthesis Evaluation Protocol (Owner: Gourav)

1. **Retrieval Relevance (Precision@3, Recall@3)**:
   - Evaluates whether top-$K$ ($K=3$) chunks returned by `MedicalVectorStore` and `EvidenceReranker` match the ground-truth thoracic pathology guidelines for each predicted abnormality.
2. **Citation Groundedness & Hallucination Prevention**:
   - Verifies that every inline `[chunk_id]` citation in the generated report corresponds to a passage retrieved during that execution and that `verify_evidence_node` flags reports when `evidence_status == "insufficient"`.
3. **Pipeline Latency Tracking**:
   - Tracks per-node execution time (`duration_ms` in `AgentStageTrace`) across `image_preprocessing`, `vision_inference`, `query_planning`, `evidence_retrieval`, `reranking`, `evidence_verification`, and `report_generation`.
