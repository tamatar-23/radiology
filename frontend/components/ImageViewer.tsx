'use client';

import React, { useState } from 'react';
import { Eye, Layers, Info } from 'lucide-react';
import { ImageMetadata } from '@/types/radiology';

interface ImageViewerProps {
  previewUrl: string | null;
  filename: string | null;
  scanId: string | null;
  metadata: ImageMetadata | null;
  heatmapAvailable: boolean;
  heatmapUrl: string | null;
  explainabilityMessage: string | null;
}

export function ImageViewer({
  previewUrl,
  filename,
  scanId,
  metadata,
  heatmapAvailable,
  heatmapUrl,
  explainabilityMessage,
}: ImageViewerProps) {
  const [activeTab, setActiveTab] = useState<'original' | 'heatmap'>('original');

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <h2 className="text-base font-semibold text-slate-900 flex items-center gap-2">
          <Eye className="w-4 h-4 text-slate-700" />
          2. Chest X-Ray Viewer & Explainability
        </h2>

        {/* View Toggle */}
        <div className="inline-flex rounded-md border border-slate-200 p-0.5 bg-slate-100 text-xs">
          <button
            type="button"
            onClick={() => setActiveTab('original')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              activeTab === 'original'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Original Radiograph
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('heatmap')}
            className={`px-2.5 py-1 rounded font-medium transition flex items-center gap-1 ${
              activeTab === 'heatmap'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Grad-CAM Heatmap
          </button>
        </div>
      </div>

      {/* Viewport */}
      <div className="rounded-lg border border-slate-300 bg-slate-950 h-72 flex items-center justify-center overflow-hidden relative">
        {!previewUrl ? (
          <div className="text-center p-6 text-slate-400 text-xs space-y-1">
            <p className="font-medium text-slate-300">No Chest X-Ray Loaded</p>
            <p>Select a PNG or JPEG chest radiograph in the upload panel above.</p>
          </div>
        ) : activeTab === 'original' ? (
          <img
            src={previewUrl}
            alt={filename || 'Uploaded Chest X-Ray'}
            className="max-h-full max-w-full object-contain"
          />
        ) : heatmapAvailable && heatmapUrl ? (
          <img
            src={heatmapUrl}
            alt="Grad-CAM Spatial Activation Heatmap"
            className="max-h-full max-w-full object-contain"
          />
        ) : (
          <div className="max-w-md text-center p-6 text-slate-200 space-y-2">
            <div className="inline-flex items-center justify-center w-9 h-9 rounded-full bg-slate-800 text-amber-400">
              <Info className="w-5 h-5" />
            </div>
            <p className="text-sm font-semibold">Grad-CAM Heatmap Unavailable</p>
            <p className="text-xs text-slate-400 leading-relaxed">
              {explainabilityMessage ||
                'Grad-CAM spatial activation maps are not generated in mock mode. No synthetic or arbitrary image filter is applied.'}
            </p>
          </div>
        )}
      </div>

      {/* Scan Metadata Footer */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1 text-xs">
        <div className="bg-slate-50 p-2.5 rounded border border-slate-200">
          <span className="text-slate-400 block">Filename</span>
          <span className="font-medium text-slate-800 truncate block">
            {filename || '—'}
          </span>
        </div>
        <div className="bg-slate-50 p-2.5 rounded border border-slate-200">
          <span className="text-slate-400 block">Scan ID</span>
          <span className="font-mono font-medium text-slate-800 truncate block">
            {scanId || 'Not uploaded'}
          </span>
        </div>
        <div className="bg-slate-50 p-2.5 rounded border border-slate-200">
          <span className="text-slate-400 block">Dimensions / Format</span>
          <span className="font-medium text-slate-800 block">
            {metadata
              ? `${metadata.width} × ${metadata.height} (${metadata.format})`
              : '—'}
          </span>
        </div>
        <div className="bg-slate-50 p-2.5 rounded border border-slate-200">
          <span className="text-slate-400 block">Intensity (Mean ± SD)</span>
          <span className="font-mono font-medium text-slate-800 block">
            {metadata
              ? `${metadata.mean_intensity.toFixed(2)} ± ${metadata.std_intensity.toFixed(2)}`
              : '—'}
          </span>
        </div>
      </div>
    </div>
  );
}
