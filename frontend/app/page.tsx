'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Activity, ShieldAlert, AlertCircle } from 'lucide-react';
import {
  HealthCheckResponse,
  ScanUploadResponse,
  ReportResponse,
} from '@/types/radiology';
import {
  checkBackendHealth,
  uploadScanFile,
  analyzeUploadedScan,
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
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [notes, setNotes] = useState<string>('');
  const [uploadedScan, setUploadedScan] = useState<ScanUploadResponse | null>(null);

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

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  // Revoke object URL on unmount or change
  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleSelectFile = (file: File) => {
    setSelectedFile(file);
    setUploadedScan(null);
    setReport(null);
    setError(null);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleUploadAndAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a PNG or JPEG Chest X-Ray image first.');
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
    <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
      {/* Research Disclaimer Banner */}
      <div
        role="note"
        className="bg-amber-50 border-l-4 border-amber-600 p-3.5 rounded-r-md text-amber-950 text-xs flex items-start gap-3"
      >
        <ShieldAlert className="w-4 h-4 text-amber-700 flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold">Academic Research Prototype Disclaimer:</span>{' '}
          This system is a 4th-year engineering capstone research prototype focused on{' '}
          <strong>Chest X-Ray analysis</strong>. It is <strong>NOT</strong> a certified
          clinical diagnostic device and must never be used to replace a radiologist or
          guide patient care.
        </div>
      </div>

      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-200 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2.5">
            <Activity className="w-6 h-6 text-slate-800" />
            Multimodal Agentic Radiology Assistant
          </h1>
          <p className="text-slate-600 text-xs mt-1">
            Chest X-Ray Preprocessing • Multi-Label Abnormality Classification • Grad-CAM
            Explainability • LangGraph Agentic RAG
          </p>
        </div>
        <div className="text-xs text-slate-500 font-mono">
          B.Tech CS Capstone • 6-Module Architecture
        </div>
      </header>

      {/* System Status Bar */}
      <SystemStatusBar
        health={health}
        healthError={healthError}
        checkingHealth={checkingHealth}
        onRefresh={refreshHealth}
      />

      {/* API / Pipeline Error Alert */}
      {error && (
        <div
          role="alert"
          className="bg-rose-50 border border-rose-200 text-rose-900 p-4 rounded-lg text-xs flex items-start gap-2.5"
        >
          <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">Pipeline Request Failed</div>
            <div className="mt-0.5 text-rose-800">{error}</div>
          </div>
        </div>
      )}

      {/* Main Two-Column Research Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload, Image Viewer, Abnormality Scores */}
        <div className="lg:col-span-5 space-y-6">
          <UploadPanel
            selectedFile={selectedFile}
            uploadedScan={uploadedScan}
            uploading={uploading}
            analyzing={analyzing}
            notes={notes}
            validationError={null}
            onSelectFile={handleSelectFile}
            onNotesChange={setNotes}
            onUploadAndAnalyze={handleUploadAndAnalyze}
          />

          <ImageViewer
            previewUrl={previewUrl}
            filename={uploadedScan?.filename || selectedFile?.name || null}
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

        {/* Right Column: Structured Report, Citations & Agent Execution Trace */}
        <div className="lg:col-span-7 space-y-6">
          <ReportPanel report={report} loading={analyzing} />

          <AgentTracePanel
            trace={report?.agent_trace || []}
            mode={report?.mode || null}
          />
        </div>
      </div>
    </main>
  );
}
