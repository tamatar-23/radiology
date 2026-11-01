/**
 * Central Typed API Client
 * Owner: Shreyash (Frontend-Backend Integration)
 *
 * Centralizes all HTTP communication with the FastAPI backend using NEXT_PUBLIC_API_BASE_URL.
 */

import {
  HealthCheckResponse,
  ScanUploadResponse,
  AnalysisRequest,
  AnalysisResponse,
  ReportResponse,
} from '@/types/radiology';

const RAW_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, '');

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function parseErrorResponse(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (payload && typeof payload.detail === 'string') {
      return payload.detail;
    }
    if (payload && Array.isArray(payload.detail)) {
      return payload.detail.map((item: { msg?: string }) => item.msg || JSON.stringify(item)).join('; ');
    }
  } catch {
    // Fallback to status text
  }
  return `Request failed with HTTP ${response.status} (${response.statusText || 'Error'})`;
}

export async function checkBackendHealth(): Promise<HealthCheckResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: 'GET',
    cache: 'no-store',
  });
  if (!response.ok) {
    throw new ApiError(await parseErrorResponse(response), response.status);
  }
  return response.json();
}

export async function uploadScanFile(file: File): Promise<ScanUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorResponse(response), response.status);
  }
  return response.json();
}

export async function analyzeUploadedScan(payload: AnalysisRequest): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorResponse(response), response.status);
  }
  return response.json();
}

export async function fetchReportById(reportId: string): Promise<ReportResponse> {
  const response = await fetch(`${API_BASE_URL}/api/report/${encodeURIComponent(reportId)}`, {
    method: 'GET',
    cache: 'no-store',
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorResponse(response), response.status);
  }
  return response.json();
}

export function getScanImageEndpointUrl(scanId: string): string {
  return `${API_BASE_URL}/api/scans/${encodeURIComponent(scanId)}/image`;
}
