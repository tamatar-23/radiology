'use client';

import React from 'react';
import { BarChart3, Info } from 'lucide-react';
import { LabelPrediction, ExecutionMode } from '@/types/radiology';

interface PredictionPanelProps {
  predictions: LabelPrediction[];
  modelName: string | null;
  mode: ExecutionMode | null;
}

export function PredictionPanel({
  predictions,
  modelName,
  mode,
}: PredictionPanelProps) {
  const displayPredictions = predictions.slice(0, 8);

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <h2 className="text-base font-semibold text-slate-900 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-slate-700" />
          3. Multi-Label Abnormality Scores
        </h2>

        {mode && (
          <div className="flex items-center gap-2 text-xs">
            <span className="font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              {modelName || 'unknown-model'}
            </span>
            <span
              className={`px-2 py-0.5 rounded font-semibold uppercase ${
                mode === 'mock'
                  ? 'bg-amber-50 text-amber-900 border border-amber-300'
                  : 'bg-blue-50 text-blue-900 border border-blue-200'
              }`}
            >
              {mode} mode
            </span>
          </div>
        )}
      </div>

      {!predictions.length ? (
        <div className="py-10 text-center text-xs text-slate-400">
          Run Chest X-Ray analysis to view multi-label abnormality classification scores.
        </div>
      ) : (
        <>
          <div className="space-y-2.5">
            {displayPredictions.map((item) => {
              const pct = Math.round(item.score * 100);
              const isElevated = item.score >= 0.5;
              return (
                <div key={item.label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span
                      className={`font-medium ${
                        isElevated ? 'text-slate-900 font-semibold' : 'text-slate-600'
                      }`}
                    >
                      {item.label.replace(/_/g, ' ')}
                    </span>
                    <span className="font-mono text-slate-700">
                      {item.score.toFixed(2)} ({pct}%)
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                    <div
                      className={`h-full rounded-full ${
                        isElevated ? 'bg-slate-800' : 'bg-slate-400'
                      }`}
                      style={{ width: `${Math.max(3, pct)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded p-2.5 text-xs text-slate-600 flex items-start gap-2">
            <Info className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
            <span>
              <strong>Terminology Note:</strong> Values shown are uncalibrated model output
              scores in [0.00, 1.00] and do not represent calibrated clinical diagnostic
              probability.
            </span>
          </div>
        </>
      )}
    </div>
  );
}
