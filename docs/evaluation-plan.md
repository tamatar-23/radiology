# Evaluation Plan (`docs/evaluation-plan.md`)

**Project**: Multimodal Agentic Radiology Assistant (Chest X-Ray Prototype)  
**Primary Owners**: Vineet (Classification & Explainability Evaluation), Gourav (RAG & Report Evaluation), Shreyansh (Dataset Curation & Splits)

---

## 1. Current Implementation Status vs. Planned Evaluation

- **Implemented in Boilerplate**:
  - `ModelEvaluator.compute_metrics(y_true, y_pred, threshold)` in `ai/classification/evaluation.py` computes actual binary/per-label **Accuracy**, **Precision**, **Recall**, and **F1-Score** over ground-truth and prediction arrays.
  - `RadiologyRAGAgent.verify_evidence_node` in `ai/rag/agent.py` evaluates citation coverage of top predicted Chest X-Ray abnormalities (`evidence_status`: `"sufficient"`, `"insufficient"`, `"unverified_claims"`).
  - Automated unit and integration tests (`ai/tests/test_ai_pipeline.py` and `backend/tests/test_api.py`) verify schema integrity, deterministic mock behavior, and conditional RAG routing.
- **Planned for Model Training & Literature Indexing Phases**:
  - Macro/Micro **ROC-AUC** and **PR-AUC** across the 14 NIH ChestX-ray14 / CheXpert pathology labels once `RealPredictionProvider` is connected to trained DenseNet121 / Rad-DINO checkpoints.
  - **Retrieval Precision@K** and **Recall@K** over a benchmark set of thoracic radiology questions indexed in ChromaDB.
  - **Faithfulness / Groundedness Audits** comparing LLM-generated impressions (`RealReportGenerator`) against retrieved guideline chunks.

---

## 2. Vision & Multi-Label Classification Evaluation Protocol

### Target Dataset Splits (Owner: Shreyansh)
- Patient-level stratified split (`70% Train / 15% Validation / 15% Test`) on NIH ChestX-ray14 or MIMIC-CXR subsets to prevent data leakage across radiographs from the same patient.

### Quantitative Metrics (Owner: Vineet)
1. **Per-Label ROC-AUC & PR-AUC**: Primary threshold-independent metrics across all 14 Chest X-Ray labels.
2. **Thresholded F1-Score, Precision, and Recall**: Computed using validation-tuned per-class thresholds via `ModelEvaluator.compute_metrics()`.
3. **Calibration Assessment (Expected Calibration Error - ECE)**: Evaluates whether raw sigmoid outputs require temperature scaling before thresholding.
4. **Grad-CAM Localization Sanity Check**: Pointing-game / bounding-box overlap evaluation on the NIH ChestX-ray14 bounding-box subset when `AI_MODE=real` is active.

---

## 3. Agentic RAG & Report Synthesis Evaluation Protocol (Owner: Gourav)

1. **Retrieval Relevance (Precision@3, Recall@3, MRR)**:
   - Evaluate whether top-$K$ ($K=3$) chunks returned by `MedicalVectorStore` and `EvidenceReranker` match the ground-truth thoracic pathology guidelines for each predicted abnormality.
2. **Citation Groundedness & Hallucination Rate**:
   - Verify that every inline `[chunk_id]` citation in the generated report corresponds to a passage retrieved during that execution and that no unsupported clinical claims appear when `evidence_status == "insufficient"`.
3. **Pipeline Latency Tracking**:
   - Track per-node execution time (`duration_ms` in `AgentStageTrace`) across `image_preprocessing`, `vision_inference`, `query_planning`, `evidence_retrieval`, `reranking`, `evidence_verification`, and `report_generation`.
