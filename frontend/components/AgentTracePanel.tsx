'use client';

import React from 'react';
import { AgentStageTrace, ExecutionMode } from '@/types/radiology';

interface AgentTracePanelProps {
  trace: AgentStageTrace[];
  mode: ExecutionMode | null;
}

export function AgentTracePanel({ trace, mode }: AgentTracePanelProps) {
  return (
    <div className="bg-white border border-slate-200 rounded-md p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <h2 className="text-sm font-semibold text-slate-900">
          Agent Execution Trace (LangGraph Pipeline)
        </h2>

        {mode === 'mock' && trace.length > 0 && (
          <span className="text-xs px-2 py-0.5 rounded bg-amber-50 text-amber-900 border border-amber-300 font-medium">
            Simulated Execution Trace (Mock Mode)
          </span>
        )}
      </div>

      {trace.length === 0 ? (
        <div className="py-4 text-center text-xs text-slate-400">
          No execution trace yet.
        </div>
      ) : (
        <ol className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {trace.map((step, index) => (
            <li
              key={step.stage_id}
              className="p-2.5 rounded border border-slate-200 bg-slate-50 flex items-start justify-between gap-2 text-xs"
            >
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-slate-400">0{index + 1}</span>
                  <span className="font-semibold text-slate-900">{step.stage_name}</span>
                  <span className="text-[11px] font-mono uppercase text-slate-500">
                    [{step.status}]
                  </span>
                </div>
                <p className="text-slate-600">{step.details}</p>
              </div>

              <span className="font-mono text-slate-500 flex-shrink-0">
                {step.duration_ms.toFixed(1)} ms
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
