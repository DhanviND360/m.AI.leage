import React from 'react';
import { PipelineStatus, PipelineStage } from '../types';

interface PipelineViewProps {
  status: PipelineStatus;
}

interface StepNode {
  id: string;
  name: string;
  role: string;
  stageKey: PipelineStage;
  icon: string;
}

const PIPELINE_STEPS: StepNode[] = [
  { id: 'planner', name: 'Planner', role: 'ActionPlan & Spec', stageKey: 'planning', icon: '🧠' },
  { id: 'router', name: 'Router', role: 'Model Dispatch', stageKey: 'routing', icon: '🔀' },
  { id: 'builder', name: 'Builder', role: 'Code & Execution', stageKey: 'building', icon: '🛠️' },
  { id: 'evaluator', name: 'Evaluator', role: 'Criteria & Verify', stageKey: 'evaluating', icon: '⚖️' },
];

export const PipelineView: React.FC<PipelineViewProps> = ({ status }) => {
  const currentStage = status.stage;

  // Determine stage state for a step: 'pending' | 'active' | 'passed' | 'failed'
  const getStepState = (index: number, stepKey: PipelineStage) => {
    if (currentStage === 'complete') return 'passed';
    if (currentStage === 'failed' || currentStage === 'escalating') {
      const activeIdx = PIPELINE_STEPS.findIndex((s) => s.stageKey === currentStage);
      if (activeIdx !== -1 && index === activeIdx) return 'failed';
    }

    const currentIdx = PIPELINE_STEPS.findIndex((s) => s.stageKey === currentStage);
    if (currentIdx === -1) {
      return currentStage === 'idle' ? 'idle' : 'pending';
    }

    if (index < currentIdx) return 'passed';
    if (index === currentIdx) return 'active';
    return 'pending';
  };

  return (
    <div className="glass-card p-5 sm:p-6 border border-mileage-border/70 relative overflow-hidden">
      {/* Background glow when active */}
      <div className="absolute -top-24 -right-24 w-48 h-48 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-24 -left-24 w-48 h-48 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-mileage-border/50 gap-2">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold text-white tracking-wide">
              Live Agent Execution Pipeline
            </h2>
            <span
              className={`text-[11px] font-mono px-2 py-0.5 rounded-full uppercase ${
                currentStage === 'idle'
                  ? 'bg-slate-800 text-slate-400'
                  : currentStage === 'complete'
                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                  : currentStage === 'escalating' || currentStage === 'failed'
                  ? 'bg-rose-950 text-rose-300 border border-rose-800/60 animate-pulse'
                  : 'bg-blue-950 text-cyan-300 border border-cyan-800/60 animate-pulse'
              }`}
            >
              ● {currentStage}
            </span>
          </div>
          <p className="text-xs text-mileage-muted mt-0.5">
            Autonomous multi-agent orchestration running locally on Ollama
          </p>
        </div>

        {status.model_name && (
          <div className="flex items-center gap-2 self-start sm:self-auto bg-mileage-surface px-3 py-1.5 rounded-lg border border-mileage-border">
            <span className="text-[11px] text-mileage-muted font-mono">Model:</span>
            <span className="text-xs font-mono text-cyan-300 font-medium">
              {status.model_name}
            </span>
          </div>
        )}
      </div>

      {/* Pipeline Sequence */}
      <div className="py-6 sm:py-8 overflow-x-auto">
        <div className="min-w-[620px] flex items-center justify-between px-2">
          {PIPELINE_STEPS.map((step, idx) => {
            const state = getStepState(idx, step.stageKey);
            const isLast = idx === PIPELINE_STEPS.length - 1;

            return (
              <React.Fragment key={step.id}>
                {/* Agent Node */}
                <div className="flex flex-col items-center group relative cursor-default">
                  {/* Outer circle */}
                  <div
                    className={`w-14 h-14 sm:w-16 sm:h-16 rounded-2xl flex items-center justify-center text-xl transition-all duration-300 ${
                      state === 'active'
                        ? 'bg-blue-950/80 border-2 border-cyan-400 text-white shadow-[0_0_20px_rgba(6,182,212,0.45)] scale-105'
                        : state === 'passed'
                        ? 'bg-emerald-950/40 border border-emerald-500/70 text-emerald-300 shadow-[0_0_12px_rgba(16,185,129,0.2)]'
                        : state === 'failed'
                        ? 'bg-rose-950/60 border border-rose-500 text-rose-300 shadow-[0_0_12px_rgba(239,68,68,0.3)]'
                        : 'bg-mileage-surface/80 border border-mileage-border/80 text-slate-500 opacity-70'
                    }`}
                  >
                    <span>{step.icon}</span>
                  </div>

                  {/* Step Name */}
                  <div className="mt-2.5 text-center">
                    <div
                      className={`text-sm font-semibold tracking-wide ${
                        state === 'active'
                          ? 'text-cyan-300'
                          : state === 'passed'
                          ? 'text-emerald-400'
                          : state === 'failed'
                          ? 'text-rose-400'
                          : 'text-slate-400'
                      }`}
                    >
                      {step.name}
                    </div>
                    <div className="text-[10px] text-mileage-muted font-mono mt-0.5">
                      {step.role}
                    </div>
                  </div>

                  {/* Status badge underneath */}
                  <div className="mt-1.5">
                    {state === 'active' && (
                      <span className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950/90 text-cyan-300 border border-cyan-800">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                        Running
                      </span>
                    )}
                    {state === 'passed' && (
                      <span className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-950/60 text-emerald-400 border border-emerald-800/60">
                        ✓ Done
                      </span>
                    )}
                    {state === 'failed' && (
                      <span className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full bg-rose-950/60 text-rose-400 border border-rose-800/60">
                        ✕ Failed
                      </span>
                    )}
                    {state === 'pending' && (
                      <span className="text-[10px] font-mono text-slate-600">
                        Waiting
                      </span>
                    )}
                    {state === 'idle' && (
                      <span className="text-[10px] font-mono text-slate-600">
                        Ready
                      </span>
                    )}
                  </div>
                </div>

                {/* Connecting arrow / line between nodes */}
                {!isLast && (
                  <div className="flex-1 mx-2 sm:mx-4 flex items-center relative mb-8">
                    <div
                      className={`h-0.5 w-full transition-all duration-500 ${
                        state === 'passed'
                          ? 'bg-emerald-500/70 shadow-[0_0_8px_rgba(16,185,129,0.3)]'
                          : state === 'active'
                          ? 'bg-gradient-to-r from-cyan-400 to-slate-700 animate-pulse'
                          : 'bg-mileage-border/80'
                      }`}
                    />
                    <div
                      className={`absolute right-0 text-[10px] -mr-1.5 ${
                        state === 'passed'
                          ? 'text-emerald-400'
                          : state === 'active'
                          ? 'text-cyan-400'
                          : 'text-mileage-border'
                      }`}
                    >
                      ▶
                    </div>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Real-time status message feed at the bottom of the pipeline */}
      {status.message && (
        <div className="mt-2 pt-3 border-t border-mileage-border/40 flex items-center gap-2.5 font-mono text-xs">
          <span className="text-cyan-400">⚡ Status:</span>
          <span className="text-slate-300 truncate">{status.message}</span>
        </div>
      )}
    </div>
  );
};
