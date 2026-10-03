import React from 'react';
import { BuildRecord } from '../types';

interface BuildHistoryTableProps {
  history: BuildRecord[];
}

export const BuildHistoryTable: React.FC<BuildHistoryTableProps> = ({ history }) => {
  const formatTime = (isoString: string) => {
    if (!isoString) return '-';
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return isoString;
    }
  };

  const formatDuration = (ms: number) => {
    if (ms <= 0) return '-';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  return (
    <div className="glass-card p-5 border border-mileage-border/70">
      <div className="flex items-center justify-between pb-4 border-b border-mileage-border/50">
        <div className="flex items-center gap-2">
          <h2 className="text-base font-semibold text-white">
            Build & Execution History
          </h2>
          <span className="text-xs font-mono text-mileage-muted bg-mileage-surface px-2 py-0.5 rounded-full border border-mileage-border">
            {history.length} records
          </span>
        </div>
        <span className="text-xs text-mileage-muted font-mono hidden sm:inline">
          Local SQLite & Telemetry Log
        </span>
      </div>

      {history.length === 0 ? (
        <div className="py-12 text-center text-slate-500 font-mono text-xs">
          No builds recorded yet today. Completed agent runs will appear here automatically.
        </div>
      ) : (
        <>
          {/* Desktop Table View */}
          <div className="hidden md:block overflow-x-auto mt-3">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-mileage-border/40 text-[11px] font-mono text-mileage-muted uppercase tracking-wider">
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Goal</th>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3 text-right">Tokens</th>
                  <th className="py-2.5 px-3 text-right">Criteria</th>
                  <th className="py-2.5 px-3 text-right">Duration</th>
                  <th className="py-2.5 px-3 text-right">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-mileage-border/30 text-xs">
                {history.map((item, idx) => (
                  <tr
                    key={item.build_id || idx}
                    className="hover:bg-mileage-surface/50 transition-colors"
                  >
                    <td className="py-3 px-3">
                      {item.escalated ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-950/80 text-amber-300 border border-amber-800/60 font-mono text-[10px]">
                          ⚡ Copilot Escalated
                        </span>
                      ) : item.status === 'complete' ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-800/60 font-mono text-[10px]">
                          ✓ Local Pass
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-rose-950/80 text-rose-300 border border-rose-800/60 font-mono text-[10px]">
                          ✕ Failed
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3 font-medium text-slate-200 max-w-xs truncate" title={item.goal}>
                      {item.goal || 'Code generation task'}
                    </td>
                    <td className="py-3 px-3 font-mono text-cyan-300">
                      {item.model_name || 'qwen2.5-coder'}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-slate-300">
                      {item.tokens_used ? item.tokens_used.toLocaleString() : '-'}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-emerald-400">
                      {item.requirements_total > 0
                        ? `${item.requirements_passed}/${item.requirements_total}`
                        : '-'}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-purple-300">
                      {formatDuration(item.duration_ms)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-mileage-muted">
                      {formatTime(item.completed_at || item.started_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile Card List View */}
          <div className="md:hidden mt-3 space-y-2.5">
            {history.map((item, idx) => (
              <div
                key={item.build_id || idx}
                className="bg-mileage-surface/70 border border-mileage-border/60 p-3 rounded-lg space-y-2"
              >
                <div className="flex items-center justify-between">
                  {item.escalated ? (
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-800/60">
                      ⚡ Escalated
                    </span>
                  ) : item.status === 'complete' ? (
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/60">
                      ✓ Pass
                    </span>
                  ) : (
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950/80 text-rose-300 border border-rose-800/60">
                      ✕ Failed
                    </span>
                  )}
                  <span className="text-[11px] font-mono text-mileage-muted">
                    {formatTime(item.completed_at || item.started_at)}
                  </span>
                </div>

                <div className="text-xs font-medium text-slate-100 line-clamp-2">
                  {item.goal || 'Code generation task'}
                </div>

                <div className="flex items-center justify-between text-[11px] font-mono text-mileage-muted pt-1 border-t border-mileage-border/40">
                  <span className="text-cyan-400 truncate max-w-[120px]">{item.model_name}</span>
                  <div className="flex gap-2">
                    <span>{formatDuration(item.duration_ms)}</span>
                    <span className="text-slate-400">
                      {item.tokens_used ? `${item.tokens_used.toLocaleString()} tok` : ''}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};
