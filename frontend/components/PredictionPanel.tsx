'use client';

import React from 'react';
import { LabelPrediction, ExecutionMode } from '@/types/radiology';

interface PredictionPanelProps {
  predictions: LabelPrediction[];
  modelName: string | null;
  mode: ExecutionMode | null;
}

export function PredictionPanel({
  predictions,
}: PredictionPanelProps) {
  const displayPredictions = predictions.slice(0, 6);

  // Values represent uncalibrated model output scores in [0.00, 1.00]
  return (
    <div className="bg-white border border-slate-200 rounded-md p-4 space-y-3 shadow-sm flex-1 flex flex-col justify-between">
      <div className="space-y-3">
        <div className="flex items-center justify-between border-b border-slate-100 pb-2">
          <h2 className="text-sm font-semibold text-slate-900">Abnormality Scores</h2>
          <span className="text-[11px] text-slate-400">uncalibrated model output</span>
        </div>

        {displayPredictions.length === 0 ? (
          <div className="py-6 text-center text-xs text-slate-400">
            Run analysis to view abnormality scores.
          </div>
        ) : (
          <div className="space-y-3 pt-1">
            {displayPredictions.map((item) => {
              const pct = Math.round(item.score * 100);
              const isElevated = item.score >= 0.5;
              return (
                <div key={item.label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className={isElevated ? 'text-slate-900 font-semibold' : 'text-slate-600'}>
                      {item.label.replace(/_/g, ' ')}
                    </span>
                    <span className="font-mono text-slate-700">
                      {item.score.toFixed(2)}
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
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
        )}
      </div>
    </div>
  );
}
