'use client';

import React from 'react';
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
  const aiMode = health?.ai_mode || 'real';

  return (
    <section
      aria-label="Backend Health"
      className="bg-white border border-slate-200 rounded-md px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs"
    >
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">Backend:</span>
          {checkingHealth ? (
            <span className="text-slate-700 font-medium">Checking...</span>
          ) : isConnected ? (
            <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 font-semibold">
              Online
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded bg-rose-50 text-rose-800 border border-rose-200 font-semibold">
              Offline
            </span>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">Database:</span>
          <span className="font-mono text-slate-800">
            {health ? health.database_status : 'Unavailable'}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">Mode:</span>
          <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-800 border border-slate-200 font-mono uppercase">
            {aiMode}
          </span>
        </div>

        {healthError && (
          <span className="text-rose-700">{healthError}</span>
        )}
      </div>

      <button
        type="button"
        onClick={onRefresh}
        disabled={checkingHealth}
        className="text-xs font-medium text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 px-2.5 py-1 rounded border border-slate-200 transition disabled:opacity-50"
      >
        Refresh
      </button>
    </section>
  );
}
