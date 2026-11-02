'use client';

import React, { useState } from 'react';
import { ImageMetadata } from '@/types/radiology';
import { resolveBackendUrl } from '@/lib/api';

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
  metadata,
  heatmapAvailable,
  heatmapUrl,
  explainabilityMessage,
}: ImageViewerProps) {
  const [activeTab, setActiveTab] = useState<'original' | 'heatmap'>('original');
  const resolvedHeatmapSrc = heatmapUrl ? resolveBackendUrl(heatmapUrl) : null;
  const displaySrc = previewUrl || '/dataset-card.jpeg';

  return (
    <div className="bg-white border border-slate-200 rounded-md p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <h2 className="text-sm font-semibold text-slate-900">
          {filename || 'dataset-card.jpeg'}
        </h2>

        <div className="inline-flex rounded border border-slate-200 p-0.5 bg-slate-100 text-xs">
          <button
            type="button"
            onClick={() => setActiveTab('original')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              activeTab === 'original'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Original
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('heatmap')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              activeTab === 'heatmap'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Grad-CAM
          </button>
        </div>
      </div>

      <div className="rounded border border-slate-300 bg-slate-950 h-72 flex items-center justify-center overflow-hidden">
        {activeTab === 'original' ? (
          <img
            src={displaySrc}
            alt={filename || 'dataset-card.jpeg'}
            className="max-h-full max-w-full object-contain"
          />
        ) : heatmapAvailable && resolvedHeatmapSrc ? (
          <img
            src={resolvedHeatmapSrc}
            alt="Grad-CAM Spatial Activation Heatmap"
            className="max-h-full max-w-full object-contain"
          />
        ) : (
          <div className="text-center p-6 text-slate-300 text-xs">
            {explainabilityMessage || 'Run analysis to generate Grad-CAM heatmap.'}
          </div>
        )}
      </div>

      {metadata && (
        <div className="flex items-center justify-between text-xs text-slate-500 pt-1">
          <span>
            {metadata.width} x {metadata.height} ({metadata.format})
          </span>
          <span className="font-mono">
            Mean: {metadata.mean_intensity.toFixed(2)} | SD: {metadata.std_intensity.toFixed(2)}
          </span>
        </div>
      )}
    </div>
  );
}
