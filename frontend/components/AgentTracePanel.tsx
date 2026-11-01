'use client';

import React from 'react';
import {
  GitBranch,
  CheckCircle2,
  AlertTriangle,
  MinusCircle,
  XCircle,
} from 'lucide-react';
import { AgentStageTrace, ExecutionMode } from '@/types/radiology';

interface AgentTracePanelProps {
  trace: AgentStageTrace[];
  mode: ExecutionMode | null;
}

export function AgentTracePanel({ trace, mode }: AgentTracePanelProps) {
  const renderStatusBadge = (status: AgentStageTrace['status']) => {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-emerald-50 text-emerald-800 border border-emerald-200">
            <CheckCircle2 className="w-3 h-3 text-emerald-600" />
            Completed
          </span>
        );
      case 'flagged':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200">
            <AlertTriangle className="w-3 h-3 text-amber-600" />
            Flagged
          </span>
        );
      case 'skipped':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-600 border border-slate-200">
            <MinusCircle className="w-3 h-3 text-slate-500" />
            Skipped
          </span>
        );
      case 'failed':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-rose-50 text-rose-800 border border-rose-200">
            <XCircle className="w-3 h-3 text-rose-600" />
            Failed
          </span>
        );
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <h2 className="text-base font-semibold text-slate-900 flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-slate-700" />
          5. Agent Execution Trace (LangGraph Pipeline)
        </h2>

        {mode === 'mock' && trace.length > 0 && (
          <span className="text-xs px-2 py-0.5 rounded bg-amber-50 text-amber-900 border border-amber-300 font-medium">
            Simulated Execution Trace (Mock Mode)
          </span>
        )}
      </div>

      {trace.length === 0 ? (
        <div className="py-8 text-center text-xs text-slate-400">
          No pipeline execution trace available yet. Trigger an analysis to inspect stage
          transitions.
        </div>
      ) : (
        <ol className="space-y-2.5">
          {trace.map((step, index) => (
            <li
              key={step.stage_id}
              className="p-3 rounded-md border border-slate-200 bg-slate-50/70 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-slate-400 font-semibold">
                    0{index + 1}
                  </span>
                  <span className="font-semibold text-slate-900">
                    {step.stage_name}
                  </span>
                  {renderStatusBadge(step.status)}
                </div>
                <p className="text-slate-600 pl-6">{step.details}</p>
              </div>

              <div className="pl-6 sm:pl-0 text-right font-mono text-slate-500 flex-shrink-0">
                {step.duration_ms.toFixed(1)} ms
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
