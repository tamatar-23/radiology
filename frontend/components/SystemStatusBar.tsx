'use client';

import React from 'react';
import {
  Server,
  Database,
  Cpu,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  XCircle,
} from 'lucide-react';
import { HealthCheckResponse } from '@/types/radiology';

interface SystemStatusBarProps {
  health: HealthCheckResponse | null;
  healthError: string | null;
  checkingHealth: boolean;
  onRefresh: () => void;
}

export function SystemStatusBar({
  health,
  healthError,
  checkingHealth,
  onRefresh,
}: SystemStatusBarProps) {
  const isConnected = Boolean(health && health.status === 'healthy');
  const aiMode = health?.ai_mode || 'mock';

  return (
    <section
      aria-label="System Status"
      className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm"
    >
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-4 text-xs">
          {/* Backend Connection */}
          <div className="flex items-center gap-2 pr-4 border-r border-slate-200">
            <Server className="w-4 h-4 text-slate-500" />
            <span className="text-slate-500 font-medium">Backend:</span>
            {checkingHealth ? (
              <span className="text-slate-600 font-semibold">Checking...</span>
            ) : isConnected ? (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                Online
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-50 text-rose-800 border border-rose-200 font-semibold">
                <XCircle className="w-3.5 h-3.5 text-rose-600" />
                Offline
              </span>
            )}
          </div>

          {/* Database Status */}
          <div className="flex items-center gap-2 pr-4 border-r border-slate-200">
            <Database className="w-4 h-4 text-slate-500" />
            <span className="text-slate-500 font-medium">Database:</span>
            <span className="font-mono text-slate-800 font-medium">
              {health ? health.database_status : 'Unavailable'}
            </span>
          </div>

          {/* Active AI Mode */}
          <div className="flex items-center gap-2 pr-4 border-r border-slate-200">
            <Cpu className="w-4 h-4 text-slate-500" />
            <span className="text-slate-500 font-medium">AI Mode:</span>
            {aiMode === 'mock' ? (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-50 text-amber-900 border border-amber-300 font-semibold">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                MOCK (Simulated Demo)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-50 text-blue-900 border border-blue-200 font-semibold">
                REAL AI
              </span>
            )}
          </div>

          {/* Modality Scope */}
          <div className="flex items-center gap-2">
            <span className="text-slate-500 font-medium">Scope:</span>
            <span className="text-slate-800 font-semibold">
              {health?.modality_scope || 'Chest X-Ray (DX)'}
            </span>
          </div>
        </div>

        <button
          type="button"
          onClick={onRefresh}
          disabled={checkingHealth}
          className="self-start lg:self-auto inline-flex items-center gap-1.5 text-xs font-medium text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 px-3 py-1.5 rounded border border-slate-200 transition disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${checkingHealth ? 'animate-spin' : ''}`} />
          Refresh Status
        </button>
      </div>

      {/* Model Integration Breakdown */}
      {health?.integrations && (
        <div className="mt-3 pt-3 border-t border-slate-100 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 text-xs text-slate-600">
          <div>
            <span className="text-slate-400 block">Vision Classifier</span>
            <span className="font-medium text-slate-800">
              {health.integrations.vision_model}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block">Explainability (Grad-CAM)</span>
            <span className="font-medium text-slate-800">
              {health.integrations.explainability}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block">RAG Evidence Store</span>
            <span className="font-medium text-slate-800">
              {health.integrations.rag_vector_store}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block">Report Synthesizer</span>
            <span className="font-medium text-slate-800">
              {health.integrations.llm_synthesizer}
            </span>
          </div>
        </div>
      )}

      {healthError && (
        <div className="mt-3 pt-2 border-t border-rose-100 text-xs text-rose-700 flex items-center gap-1.5">
          <XCircle className="w-3.5 h-3.5 flex-shrink-0" />
          <span>Backend unreachable: {healthError}</span>
        </div>
      )}
    </section>
  );
}
