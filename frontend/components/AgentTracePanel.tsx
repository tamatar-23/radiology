'use client';

import React from 'react';
import { AgentStageTrace, ExecutionMode } from '@/types/radiology';

interface AgentTracePanelProps {
  trace: AgentStageTrace[];
  mode: ExecutionMode | null;
}

export function AgentTracePanel({ trace, mode }: AgentTracePanelProps) {
  return (
    <div className="bg-white border border-slate-200 rounded-md p-4 space-y-4 shadow-sm">
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
        <ol className="relative border-l-2 border-emerald-200 ml-3 space-y-3">
          {trace.map((step, index) => {
            const isCompleted = step.status === 'completed';
            return (
              <li key={step.stage_id} className="relative pl-5">
                <span
                  className={`absolute -left-[9px] top-3.5 h-4 w-4 rounded-full border-2 flex items-center justify-center ${
                    isCompleted
                      ? 'bg-emerald-500 border-emerald-100'
                      : 'bg-amber-500 border-amber-100'
                  }`}
                />
                <div className="p-3 rounded border border-slate-200 bg-slate-50 flex items-start justify-between gap-3 text-xs">
                  <div className="space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-slate-400">0{index + 1}</span>
                      <span className="font-semibold text-slate-900">{step.stage_name}</span>
                      <span
                        className={`px-2 py-0.5 rounded text-[11px] font-mono uppercase border font-semibold ${
                          isCompleted
                            ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                            : 'bg-amber-50 text-amber-800 border-amber-200'
                        }`}
                      >
                        {step.status}
                      </span>
                    </div>
                    <p className="text-slate-600">{step.details}</p>
                  </div>

                  <span className="font-mono text-slate-500 flex-shrink-0">
                    {step.duration_ms.toFixed(1)} ms
                  </span>
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
