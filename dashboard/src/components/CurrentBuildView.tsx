import React from 'react';
import { PipelineStatus } from '../types';

interface CurrentBuildViewProps {
  status: PipelineStatus;
}

export const CurrentBuildView: React.FC<CurrentBuildViewProps> = ({ status }) => {
  const isRunning = status.stage !== 'idle';
  const progressPercent =
    status.requirements_total > 0
      ? Math.round((status.requirements_passed / status.requirements_total) * 100)
      : 0;

  const formatElapsed = (ms: number) => {
    if (ms <= 0) return '0s';
    const totalSecs = Math.floor(ms / 1000);
    const mins = Math.floor(totalSecs / 60);
    const secs = totalSecs % 60;
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="glass-card p-5 border border-mileage-border/70 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-mileage-border/50">
          <div className="flex items-center gap-2">
            <span className="text-base font-semibold text-white">
              Current Active Build
            </span>
            {isRunning && (
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            )}
          </div>
          <span className="text-xs font-mono text-mileage-muted">
            {isRunning ? 'IN PROGRESS' : 'IDLE / READY'}
          </span>
        </div>

        {isRunning ? (
          <div className="mt-4 space-y-4">
            {/* Goal Title */}
            <div>
              <div className="text-[11px] font-mono text-mileage-muted uppercase tracking-wider mb-1">
                Goal Objective
              </div>
              <div className="text-sm sm:text-base font-medium text-slate-100 bg-mileage-surface/80 p-3 rounded-lg border border-mileage-border/60">
                {status.goal || 'Autonomous workspace task execution'}
              </div>
            </div>

            {/* Iteration & Elapsed & Model */}
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-mileage-surface/60 p-2.5 rounded-lg border border-mileage-border/40">
                <div className="text-[10px] font-mono text-mileage-muted uppercase">Iteration</div>
                <div className="text-sm font-semibold text-white mt-0.5 font-mono">
                  {status.iteration}
                  {status.max_iterations > 0 ? ` / ${status.max_iterations}` : ''}
                </div>
              </div>
              <div className="bg-mileage-surface/60 p-2.5 rounded-lg border border-mileage-border/40">
                <div className="text-[10px] font-mono text-mileage-muted uppercase">Elapsed</div>
                <div className="text-sm font-semibold text-cyan-300 mt-0.5 font-mono">
                  {formatElapsed(status.elapsed_ms)}
                </div>
              </div>
              <div className="bg-mileage-surface/60 p-2.5 rounded-lg border border-mileage-border/40 truncate">
                <div className="text-[10px] font-mono text-mileage-muted uppercase">Stage</div>
                <div className="text-sm font-semibold text-purple-300 mt-0.5 font-mono capitalize truncate">
                  {status.stage}
                </div>
              </div>
            </div>

            {/* Acceptance Criteria Progress */}
            {status.requirements_total > 0 && (
              <div className="mt-2">
                <div className="flex items-center justify-between text-xs mb-1.5 font-mono">
                  <span className="text-mileage-muted">Criteria Verified</span>
                  <span className="text-emerald-400 font-medium">
                    {status.requirements_passed} / {status.requirements_total} ({progressPercent}%)
                  </span>
                </div>
                <div className="w-full bg-mileage-surface rounded-full h-2 overflow-hidden border border-mileage-border">
                  <div
                    className="bg-gradient-to-r from-cyan-500 to-emerald-400 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="py-8 text-center space-y-3">
            <div className="w-12 h-12 rounded-xl bg-mileage-surface border border-mileage-border flex items-center justify-center mx-auto text-xl text-slate-400">
              💤
            </div>
            <div>
              <div className="text-sm font-medium text-slate-300">
                Agent is standing by
              </div>
              <p className="text-xs text-mileage-muted max-w-sm mx-auto mt-1">
                Start a local coding session, planning, or benchmark from your terminal.
              </p>
            </div>
            <div className="pt-2">
              <code className="text-xs bg-mileage-surface text-cyan-300 px-3 py-1.5 rounded-md border border-mileage-border/80 font-mono inline-block">
                mileage code "build my feature"
              </code>
            </div>
          </div>
        )}
      </div>

      {isRunning && (
        <div className="mt-4 pt-3 border-t border-mileage-border/40 text-[11px] text-mileage-muted flex items-center justify-between">
          <span className="font-mono">Streaming live telemetry</span>
          <span className="text-cyan-400 font-mono">● Real-time</span>
        </div>
      )}
    </div>
  );
};
