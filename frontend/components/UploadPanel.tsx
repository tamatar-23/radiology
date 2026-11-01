'use client';

import React, { useState } from 'react';
import {
  Upload,
  FileImage,
  CheckCircle2,
  AlertCircle,
  Play,
  Loader2,
} from 'lucide-react';
import { ScanUploadResponse } from '@/types/radiology';

interface UploadPanelProps {
  selectedFile: File | null;
  uploadedScan: ScanUploadResponse | null;
  uploading: boolean;
  analyzing: boolean;
  notes: string;
  validationError: string | null;
  onSelectFile: (file: File) => void;
  onNotesChange: (notes: string) => void;
  onUploadAndAnalyze: () => void;
}

const ACCEPTED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];
const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024;

export function UploadPanel({
  selectedFile,
  uploadedScan,
  uploading,
  analyzing,
  notes,
  validationError,
  onSelectFile,
  onNotesChange,
  onUploadAndAnalyze,
}: UploadPanelProps) {
  const [dragActive, setDragActive] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const validateAndForward = (file: File) => {
    setLocalError(null);
    const lowerName = file.name.toLowerCase();
    const isValidExt =
      lowerName.endsWith('.png') ||
      lowerName.endsWith('.jpg') ||
      lowerName.endsWith('.jpeg');
    const isValidMime = !file.type || ACCEPTED_MIME_TYPES.includes(file.type.toLowerCase());

    if (!isValidExt && !isValidMime) {
      setLocalError(
        'Unsupported file format. Please select a PNG or JPEG Chest X-Ray image.'
      );
      return;
    }
    if (file.size === 0) {
      setLocalError('Selected file is empty (0 bytes).');
      return;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setLocalError('Selected file exceeds the 15 MB maximum upload size.');
      return;
    }

    onSelectFile(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndForward(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndForward(e.dataTransfer.files[0]);
    }
  };

  const busy = uploading || analyzing;
  const displayError = localError || validationError;

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <h2 className="text-base font-semibold text-slate-900 flex items-center gap-2">
          <Upload className="w-4 h-4 text-slate-700" />
          1. Chest X-Ray Upload
        </h2>
        <span className="text-xs text-slate-500 font-medium">PNG / JPEG • Max 15 MB</span>
      </div>

      {/* Drag & Drop Picker */}
      <label
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setDragActive(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setDragActive(false);
        }}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer transition ${
          dragActive
            ? 'border-blue-600 bg-blue-50/40'
            : 'border-slate-300 hover:border-slate-400 bg-slate-50/50'
        }`}
      >
        <Upload className="w-8 h-8 text-slate-400 mb-2" />
        <span className="text-sm font-medium text-slate-800 text-center">
          Drag and drop a Chest X-Ray image here, or click to browse
        </span>
        <span className="text-xs text-slate-500 mt-1 text-center">
          Supported prototype formats: PNG (.png), JPEG (.jpg, .jpeg)
        </span>
        <input
          type="file"
          className="hidden"
          accept="image/png,image/jpeg,.png,.jpg,.jpeg"
          onChange={handleFileInput}
          disabled={busy}
        />
      </label>

      {/* Selected File State */}
      {selectedFile && (
        <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <FileImage className="w-6 h-6 text-slate-700 flex-shrink-0" />
            <div className="min-w-0">
              <div className="text-sm font-medium text-slate-900 truncate">
                {selectedFile.name}
              </div>
              <div className="text-xs text-slate-500">
                {(selectedFile.size / 1024).toFixed(1)} KB •{' '}
                {uploadedScan ? uploadedScan.media_type : selectedFile.type || 'image'}
              </div>
            </div>
          </div>

          {uploadedScan ? (
            <span className="inline-flex items-center gap-1 bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs px-2.5 py-1 rounded font-medium flex-shrink-0">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              Uploaded
            </span>
          ) : (
            <span className="bg-slate-200 text-slate-800 text-xs px-2.5 py-1 rounded font-medium flex-shrink-0">
              Ready to Upload
            </span>
          )}
        </div>
      )}

      {/* Optional Clinical Context Notes */}
      <div className="space-y-1.5">
        <label
          htmlFor="clinical-notes"
          className="block text-xs font-medium text-slate-700"
        >
          Clinical Context / Indication Notes (Optional)
        </label>
        <input
          id="clinical-notes"
          type="text"
          value={notes}
          onChange={(e) => onNotesChange(e.target.value)}
          disabled={busy}
          placeholder="e.g., Cough and dyspnea, evaluate for basal consolidation or effusion"
          className="w-full text-sm px-3 py-2 rounded-md border border-slate-300 bg-white text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent"
        />
      </div>

      {/* Validation / Upload Error */}
      {displayError && (
        <div
          role="alert"
          className="bg-rose-50 border border-rose-200 text-rose-800 p-3 rounded-md text-xs flex items-start gap-2"
        >
          <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
          <span>{displayError}</span>
        </div>
      )}

      {/* Action Button */}
      <button
        type="button"
        onClick={onUploadAndAnalyze}
        disabled={!selectedFile || busy}
        className="w-full bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 disabled:text-slate-500 text-white text-sm font-semibold py-2.5 px-4 rounded-md transition flex items-center justify-center gap-2"
      >
        {uploading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Validating & Uploading Scan...
          </>
        ) : analyzing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Running Vision & Agentic RAG Pipeline...
          </>
        ) : (
          <>
            <Play className="w-4 h-4" />
            {uploadedScan ? 'Re-Run Chest X-Ray Analysis' : 'Upload & Analyze Chest X-Ray'}
          </>
        )}
      </button>
    </div>
  );
}
