'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  HealthCheckResponse,
  ScanUploadResponse,
  ReportResponse,
} from '@/types/radiology';
import {
  checkBackendHealth,
  uploadScanFile,
  analyzeUploadedScan,
  fetchSampleScanFile,
} from '@/lib/api';
import { SystemStatusBar } from '@/components/SystemStatusBar';
import { UploadPanel } from '@/components/UploadPanel';
import { ImageViewer } from '@/components/ImageViewer';
import { PredictionPanel } from '@/components/PredictionPanel';
import { ReportPanel } from '@/components/ReportPanel';
import { AgentTracePanel } from '@/components/AgentTracePanel';

export default function RadiologyDashboardPage() {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [checkingHealth, setCheckingHealth] = useState<boolean>(true);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>('/dataset-card.jpeg');
  const [notes, setNotes] = useState<string>('');
  const [uploadedScan, setUploadedScan] = useState<ScanUploadResponse | null>(null);

  const [loadingSample, setLoadingSample] = useState<boolean>(false);
  const [uploading, setUploading] = useState<boolean>(false);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshHealth = useCallback(async () => {
    setCheckingHealth(true);
    setHealthError(null);
    try {
      const data = await checkBackendHealth();
      setHealth(data);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Unable to reach FastAPI backend.';
      setHealth(null);
      setHealthError(message);
    } finally {
      setCheckingHealth(false);
    }
  }, []);

  const handleSelectFile = useCallback((file: File) => {
    setSelectedFile(file);
    setUploadedScan(null);
    setReport(null);
    setError(null);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  }, []);

  const handleLoadSample = useCallback(async () => {
    setLoadingSample(true);
    setError(null);
    try {
      const sampleFile = await fetchSampleScanFile();
      handleSelectFile(sampleFile);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Failed to load default image.';
      setError(message);
    } finally {
      setLoadingSample(false);
    }
  }, [handleSelectFile]);

  useEffect(() => {
    refreshHealth();
    handleLoadSample();
  }, [refreshHealth, handleLoadSample]);

  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleUploadAndAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a PNG or JPEG image first.');
      return;
    }

    setError(null);

    try {
      let currentScan = uploadedScan;
      if (!currentScan) {
        setUploading(true);
        currentScan = await uploadScanFile(selectedFile);
        setUploadedScan(currentScan);
        setUploading(false);
      }

      setAnalyzing(true);
      const analysisReport = await analyzeUploadedScan({
        scan_id: currentScan.scan_id,
        notes: notes.trim() ? notes.trim() : null,
      });
      setReport(analysisReport);
      refreshHealth();
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'An unexpected error occurred during analysis.';
      setError(message);
    } finally {
      setUploading(false);
      setAnalyzing(false);
    }
  };

  const activeMetadata =
    report?.image_metadata || uploadedScan?.image_metadata || null;

  return (
    <main className="max-w-7xl mx-auto px-6 py-5 space-y-4">
      <div className="bg-white border border-slate-200 rounded-md px-5 py-3 flex flex-col md:flex-row md:items-center md:justify-between gap-3 shadow-sm">
        <h1 className="text-base font-semibold text-slate-900 tracking-tight">
          Multimodal Agentic Radiology Assistant
        </h1>

        <SystemStatusBar
          health={health}
          healthError={healthError}
          checkingHealth={checkingHealth}
          onRefresh={refreshHealth}
        />
      </div>

      {error && (
        <div
          role="alert"
          className="bg-rose-50 border border-rose-200 text-rose-900 px-4 py-3 rounded-md text-xs"
        >
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
        <div className="lg:col-span-5 flex flex-col gap-4">
          <UploadPanel
            selectedFile={selectedFile}
            uploadedScan={uploadedScan}
            uploading={uploading}
            analyzing={analyzing}
            loadingSample={loadingSample}
            notes={notes}
            validationError={null}
            onSelectFile={handleSelectFile}
            onLoadSample={handleLoadSample}
            onNotesChange={setNotes}
            onUploadAndAnalyze={handleUploadAndAnalyze}
          />

          <ImageViewer
            previewUrl={previewUrl}
            filename={uploadedScan?.filename || selectedFile?.name || 'dataset-card.jpeg'}
            scanId={report?.scan_id || uploadedScan?.scan_id || null}
            metadata={activeMetadata}
            heatmapAvailable={Boolean(report?.heatmap_available)}
            heatmapUrl={report?.heatmap_url || null}
            explainabilityMessage={report?.explainability_message || null}
          />

          <PredictionPanel
            predictions={report?.predictions || []}
            modelName={report?.model_name || null}
            mode={report?.mode || null}
          />
        </div>

        <div className="lg:col-span-7 flex flex-col">
          <ReportPanel report={report} loading={analyzing} />
        </div>
      </div>

      <AgentTracePanel
        trace={report?.agent_trace || []}
        mode={report?.mode || null}
      />
    </main>
  );
}
