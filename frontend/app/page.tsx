'use client';

import React, { useState, useEffect } from 'react';
import {
  Activity,
  Upload,
  FileText,
  CheckCircle,
  AlertCircle,
  Search,
  BookOpen,
  Cpu,
  RefreshCw,
  ShieldAlert
} from 'lucide-react';

interface Evidence {
  source: string;
  text: string;
}

interface AnalysisResult {
  status: string;
  scan_id?: string;
  findings: string[];
  impression: string;
  confidence: number;
  evidence: Evidence[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [backendHealth, setBackendHealth] = useState<string>('Checking...');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check Backend Health on Mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then((res) => res.json())
      .then((data) => setBackendHealth(data.status === 'healthy' ? 'Connected' : 'Degraded'))
      .catch(() => setBackendHealth('Offline (Ensure FastAPI is running)'));
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setError(null);

      // Create preview for image files
      if (file.type.startsWith('image/')) {
        setPreviewUrl(URL.createObjectURL(file));
      } else {
        setPreviewUrl(null);
      }
    }
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Optional: upload file first if selected
      if (selectedFile) {
        const formData = new FormData();
        formData.append('file', selectedFile);
        await fetch(`${API_BASE_URL}/api/upload`, {
          method: 'POST',
          body: formData,
        });
      }

      // Call Analyze endpoint
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_name: selectedFile ? selectedFile.name : 'sample_cxr.dcm',
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze scan');
      // Fallback mock payload for offline testing demonstration
      setResult({
        status: 'mock_fallback',
        scan_id: 'scan_demo_fallback',
        findings: ['Mock finding: Ground-glass opacity detected in lower lung field.'],
        impression: 'Mock radiology impression: Features suggestive of localized inflammatory process.',
        confidence: 0.85,
        evidence: [
          {
            source: 'Mock Radiology Literature',
            text: 'Ground glass opacities in basal lobes correlate with early infiltrates.'
          }
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-7xl mx-auto p-6 space-y-8">
      {/* Disclaimer Alert */}
      <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded text-amber-900 text-sm flex items-start gap-3">
        <ShieldAlert className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold">Research & Development Prototype Notice:</span> This application is a barebones MVP skeleton designed solely for architectural alignment and engineering development. It is <strong>NOT</strong> a medical diagnostic tool and must not be used for clinical decision-making.
        </div>
      </div>

      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-600" />
            Multimodal Agentic Radiology Assistant
          </h1>
          <p className="text-slate-600 text-sm mt-1">
            4th-Year Capstone Project Architecture Skeleton & Agentic RAG Pipeline Dashboard
          </p>
        </div>

        {/* System Health Badge */}
        <div className="flex items-center gap-2 bg-white px-4 py-2 border rounded-lg shadow-sm text-sm">
          <span className="text-slate-500 font-medium">Backend API:</span>
          <span
            className={`font-semibold px-2 py-0.5 rounded text-xs ${
              backendHealth === 'Connected'
                ? 'bg-emerald-100 text-emerald-800'
                : 'bg-rose-100 text-rose-800'
            }`}
          >
            {backendHealth}
          </span>
        </div>
      </header>

      {/* Team Ownership Quick Bar */}
      <section className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-xs grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="border-r border-slate-100 pr-2">
          <div className="font-semibold text-slate-700">Gourav</div>
          <div className="text-slate-500">RAG & Vector DB</div>
        </div>
        <div className="border-r border-slate-100 pr-2">
          <div className="font-semibold text-slate-700">Shanks</div>
          <div className="text-slate-500">Medical Vision</div>
        </div>
        <div className="border-r border-slate-100 pr-2">
          <div className="font-semibold text-slate-700">Shreyansh</div>
          <div className="text-slate-500">Datasets & Cleaning</div>
        </div>
        <div className="border-r border-slate-100 pr-2">
          <div className="font-semibold text-slate-700">Vineet</div>
          <div className="text-slate-500">Classification & Eval</div>
        </div>
        <div className="border-r border-slate-100 pr-2">
          <div className="font-semibold text-slate-700">Shivam</div>
          <div className="text-slate-500">Backend & Docker</div>
        </div>
        <div>
          <div className="font-semibold text-slate-700">Shreyash</div>
          <div className="text-slate-500">Frontend & Docs</div>
        </div>
      </section>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

        {/* Left Column: Upload & Controls */}
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-5">
            <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
              <Upload className="w-5 h-5 text-blue-600" />
              1. Medical Image Upload
            </h2>

            {/* Drag & Drop Upload Container */}
            <label className="border-2 border-dashed border-slate-300 rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:border-blue-500 hover:bg-blue-50/50 transition">
              <Upload className="w-10 h-10 text-slate-400 mb-2" />
              <span className="text-sm font-medium text-slate-700">
                Click to browse or drop DICOM / Image file
              </span>
              <span className="text-xs text-slate-400 mt-1">
                Supports DICOM (.dcm), PNG, JPEG, CT, MRI, X-Ray
              </span>
              <input
                type="file"
                className="hidden"
                accept="image/*,.dcm"
                onChange={handleFileChange}
              />
            </label>

            {/* Selected File Preview Card */}
            {selectedFile && (
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileText className="w-8 h-8 text-blue-600" />
                  <div>
                    <div className="text-sm font-medium text-slate-900 truncate max-w-[200px]">
                      {selectedFile.name}
                    </div>
                    <div className="text-xs text-slate-500">
                      {(selectedFile.size / 1024).toFixed(1)} KB • {selectedFile.type || 'DICOM Scan'}
                    </div>
                  </div>
                </div>
                <span className="bg-blue-100 text-blue-800 text-xs px-2.5 py-1 rounded font-medium">
                  Ready
                </span>
              </div>
            )}

            {/* Image Preview Box if image */}
            {previewUrl && (
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Selected Scan Preview
                </span>
                <div className="border rounded-lg overflow-hidden bg-black flex justify-center items-center h-48">
                  <img
                    src={previewUrl}
                    alt="Scan Preview"
                    className="max-h-full object-contain"
                  />
                </div>
              </div>
            )}

            {/* Analyze Scan Action Button */}
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold py-3 px-4 rounded-lg shadow transition flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  Analyzing Scan with Multimodal AI...
                </>
              ) : (
                <>
                  <Cpu className="w-5 h-5" />
                  Analyze Scan
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: AI Analysis & Radiology Report Results */}
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-6 min-h-[450px]">
            <div className="flex items-center justify-between border-b pb-4">
              <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-600" />
                2. AI Pipeline Analysis & Report
              </h2>
              {result && (
                <span className="bg-emerald-100 text-emerald-800 text-xs px-2.5 py-1 rounded font-medium flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5" />
                  Analysis Complete
                </span>
              )}
            </div>

            {/* Loading State Skeleton */}
            {loading && (
              <div className="space-y-4 py-8 animate-pulse">
                <div className="h-4 bg-slate-200 rounded w-3/4"></div>
                <div className="h-4 bg-slate-200 rounded w-1/2"></div>
                <div className="h-20 bg-slate-100 rounded"></div>
                <div className="h-16 bg-slate-100 rounded"></div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="bg-rose-50 text-rose-800 p-3 rounded-lg text-xs flex items-center gap-2 border border-rose-200">
                <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
                <span>Notice: {error}. Showing development mock result below.</span>
              </div>
            )}

            {/* Empty State */}
            {!loading && !result && (
              <div className="flex flex-col items-center justify-center py-16 text-slate-400 space-y-3">
                <Search className="w-12 h-12 text-slate-300" />
                <p className="text-sm font-medium">Upload a scan and click "Analyze Scan" to view AI results.</p>
              </div>
            )}

            {/* Results Section */}
            {result && !loading && (
              <div className="space-y-6 text-sm">
                
                {/* Confidence Card */}
                <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
                      AI Diagnostic Confidence
                    </span>
                    <span className="text-2xl font-bold text-slate-900">
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-xs text-slate-500 block">Scan ID</span>
                    <span className="font-mono text-xs text-slate-700 font-semibold">
                      {result.scan_id}
                    </span>
                  </div>
                </div>

                {/* Generated Impression Section */}
                <div className="space-y-2">
                  <h3 className="font-semibold text-slate-900 uppercase text-xs tracking-wider flex items-center gap-1.5">
                    <Activity className="w-4 h-4 text-blue-600" />
                    Impression
                  </h3>
                  <div className="bg-blue-50/70 border border-blue-100 p-3.5 rounded-lg text-slate-800 font-medium leading-relaxed">
                    {result.impression}
                  </div>
                </div>

                {/* Findings Section */}
                <div className="space-y-2">
                  <h3 className="font-semibold text-slate-900 uppercase text-xs tracking-wider flex items-center gap-1.5">
                    <FileText className="w-4 h-4 text-slate-600" />
                    Detailed Findings
                  </h3>
                  <ul className="list-disc list-inside space-y-1 bg-slate-50 p-3.5 rounded-lg border border-slate-200 text-slate-700">
                    {result.findings.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>

                {/* Retrieved Evidence Section (Agentic RAG) */}
                <div className="space-y-2">
                  <h3 className="font-semibold text-slate-900 uppercase text-xs tracking-wider flex items-center gap-1.5">
                    <BookOpen className="w-4 h-4 text-indigo-600" />
                    Retrieved Knowledge Evidence (RAG)
                  </h3>
                  <div className="space-y-2">
                    {result.evidence.map((ev, idx) => (
                      <div key={idx} className="bg-indigo-50/50 border border-indigo-100 p-3 rounded-lg">
                        <div className="font-semibold text-indigo-900 text-xs mb-1">
                          Source: {ev.source}
                        </div>
                        <div className="text-indigo-800 text-xs italic">
                          "{ev.text}"
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            )}
          </div>
        </div>

      </div>
    </main>
  );
}
