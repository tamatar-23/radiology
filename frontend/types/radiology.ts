/**
 * Shared TypeScript Contracts for the Chest X-Ray Research Dashboard
 * Owner: Shreyash (Frontend & Integration)
 */

export type ExecutionMode = 'mock' | 'real';

export interface ImageMetadata {
  width: number;
  height: number;
  channels: number;
  format: string;
  color_mode: string;
  mean_intensity: number;
  std_intensity: number;
  modality: string;
}

export interface LabelPrediction {
  label: string;
  score: number;
}

export interface EvidenceCitation {
  evidence_id: string;
  document_title: string;
  source_url: string;
  passage: string;
  chunk_id: string;
  relevance_score: number;
}

export interface AgentStageTrace {
  stage_id: string;
  stage_name: string;
  status: 'completed' | 'skipped' | 'flagged' | 'failed';
  mode: ExecutionMode;
  duration_ms: number;
  details: string;
}

export interface HealthCheckResponse {
  status: string;
  service: string;
  database_status: string;
  ai_mode: ExecutionMode;
  modality_scope: string;
  integrations: {
    vision_model?: string;
    explainability?: string;
    rag_vector_store?: string;
    llm_synthesizer?: string;
    [key: string]: string | undefined;
  };
  timestamp: string;
}

export interface ScanUploadResponse {
  scan_id: string;
  filename: string;
  media_type: string;
  status: string;
  file_type: string;
  file_size_bytes: number;
  image_metadata?: ImageMetadata | null;
  message: string;
}

export interface AnalysisRequest {
  scan_id: string;
  notes?: string | null;
}

export interface ReportResponse {
  report_id: string;
  id: string;
  scan_id: string;
  status: string;
  mode: ExecutionMode;
  disclaimer: string;
  model_name: string;
  predictions: LabelPrediction[];
  heatmap_available: boolean;
  heatmap_url?: string | null;
  explainability_message: string;
  findings: string[];
  impression: string;
  evidence_status: 'sufficient' | 'insufficient' | 'unverified_claims' | string;
  evidence: EvidenceCitation[];
  agent_trace: AgentStageTrace[];
  image_metadata?: ImageMetadata | null;
  confidence: number;
  created_at: string;
}

export type AnalysisResponse = ReportResponse;
