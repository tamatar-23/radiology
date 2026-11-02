'use client';

import React, { useState } from 'react';
import { ScanUploadResponse } from '@/types/radiology';

interface UploadPanelProps {
  selectedFile: File | null;
  uploadedScan: ScanUploadResponse | null;
  uploading: boolean;
  analyzing: boolean;
  loadingSample?: boolean;
  notes: string;
  validationError: string | null;
  onSelectFile: (file: File) => void;
  onLoadSample?: () => void;
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
  loadingSample = false,
  notes,
  validationError,
  onSelectFile,
  onLoadSample,
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
      setLocalError('Unsupported file format. Please select a PNG or JPEG image.');
      return;
    }
    if (file.size === 0) {
      setLocalError('Selected file is empty (0 bytes).');
      return;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setLocalError('Selected file exceeds the 15 MB limit.');
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

  const busy = uploading || analyzing || loadingSample;
  const displayError = localError || validationError;

  return (
    <div className="bg-white border border-slate-200 rounded-md p-4 space-y-3 shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <h2 className="text-sm font-semibold text-slate-900">Upload</h2>
        {onLoadSample && (
          <button
            type="button"
            onClick={onLoadSample}
            disabled={busy}
            className="text-xs font-medium text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 border border-slate-200 px-2.5 py-1 rounded transition disabled:opacity-50"
          >
            {loadingSample ? 'Loading...' : 'Default Image (dataset-card.jpeg)'}
          </button>
        )}
      </div>

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
        className={`border border-dashed rounded p-4 flex flex-col items-center justify-center cursor-pointer transition ${
          dragActive
            ? 'border-slate-700 bg-slate-100'
            : 'border-slate-300 hover:border-slate-400 bg-slate-50'
        }`}
      >
        <span className="text-xs font-medium text-slate-700 text-center">
          Click or drag a PNG/JPEG image to replace current image
        </span>
        <input
          type="file"
          className="hidden"
          accept="image/png,image/jpeg,.png,.jpg,.jpeg"
          onChange={handleFileInput}
          disabled={busy}
        />
      </label>

      {selectedFile && (
        <div className="bg-slate-50 px-3 py-2 rounded border border-slate-200 flex items-center justify-between text-xs">
          <span className="font-medium text-slate-900 truncate">
            {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
          </span>
          <span className="text-slate-600 font-mono">
            {uploadedScan ? 'Uploaded' : 'Ready'}
          </span>
        </div>
      )}

      <div>
        <input
          id="clinical-notes"
          type="text"
          value={notes}
          onChange={(e) => onNotesChange(e.target.value)}
          disabled={busy}
          placeholder="Optional clinical notes..."
          className="w-full text-xs px-3 py-2 rounded border border-slate-300 bg-white text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-slate-600"
        />
      </div>

      {displayError && (
        <div
          role="alert"
          className="bg-rose-50 border border-rose-200 text-rose-800 p-2.5 rounded text-xs"
        >
          {displayError}
        </div>
      )}

      <button
        type="button"
        onClick={onUploadAndAnalyze}
        disabled={!selectedFile || busy}
        className="w-full bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 disabled:text-slate-500 text-white text-xs font-semibold py-2.5 px-4 rounded transition"
      >
        {uploading
          ? 'Validating & Uploading Scan...'
          : analyzing
          ? 'Running Vision & Agentic RAG Pipeline...'
          : 'Analyze Chest X-Ray'}
      </button>
    </div>
  );
}
