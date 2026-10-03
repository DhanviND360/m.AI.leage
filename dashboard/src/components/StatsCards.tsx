import React from 'react';
import { DashboardStats } from '../types';

interface StatsCardsProps {
  stats: DashboardStats;
}

export const StatsCards: React.FC<StatsCardsProps> = ({ stats }) => {
  // Format token counts nicely (e.g. 1.2k, 45.2k)
  const formatTokens = (val: number) => {
    if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(2)}M`;
    if (val >= 1_000) return `${(val / 1_000).toFixed(1)}k`;
    return val.toLocaleString();
  };

  const formatDuration = (ms: number) => {
    if (ms <= 0) return '0s';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
      {/* 1. Projects Built */}
      <div className="glass-card p-4 sm:p-5 relative overflow-hidden transition-all duration-300 hover:border-blue-500/40 hover:shadow-lg hover:shadow-blue-500/5">
        <div className="flex items-center justify-between text-mileage-muted text-xs font-medium uppercase tracking-wider mb-2">
          <span>Projects Built</span>
          <span className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400">
            🏗️
          </span>
        </div>
        <div className="stat-number text-white">
          {stats.projects_built || stats.total_builds_today}
        </div>
        <div className="mt-2 text-[11px] text-mileage-muted flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          <span>{stats.tasks_completed_locally.toFixed(0)}% completed locally</span>
        </div>
      </div>

      {/* 2. Tokens Saved */}
      <div className="glass-card p-4 sm:p-5 relative overflow-hidden transition-all duration-300 hover:border-cyan-500/40 hover:shadow-lg hover:shadow-cyan-500/5">
        <div className="flex items-center justify-between text-mileage-muted text-xs font-medium uppercase tracking-wider mb-2">
          <span>Tokens Saved</span>
          <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400">
            ⚡
          </span>
        </div>
        <div className="stat-number text-cyan-300">
          {formatTokens(stats.tokens_saved_estimate || stats.total_tokens_used)}
        </div>
        <div className="mt-2 text-[11px] text-cyan-400/80 font-mono">
          100% on-device inference
        </div>
      </div>

      {/* 3. Estimated Cost Avoided */}
      <div className="glass-card p-4 sm:p-5 relative overflow-hidden transition-all duration-300 hover:border-emerald-500/40 hover:shadow-lg hover:shadow-emerald-500/5">
        <div className="flex items-center justify-between text-mileage-muted text-xs font-medium uppercase tracking-wider mb-2">
          <span>Cloud Cost Avoided</span>
          <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400">
            💵
          </span>
        </div>
        <div className="stat-number text-emerald-400">
          ${stats.estimated_cost_avoided.toFixed(2)}
        </div>
        <div className="mt-2 text-[11px] text-emerald-400/80">
          vs standard cloud API rates
        </div>
      </div>

      {/* 4. Avg Build Time & Models Used */}
      <div className="glass-card p-4 sm:p-5 relative overflow-hidden transition-all duration-300 hover:border-purple-500/40 hover:shadow-lg hover:shadow-purple-500/5">
        <div className="flex items-center justify-between text-mileage-muted text-xs font-medium uppercase tracking-wider mb-2">
          <span>Avg Build Time</span>
          <span className="p-1.5 rounded-lg bg-purple-500/10 text-purple-400">
            ⏱️
          </span>
        </div>
        <div className="stat-number text-purple-300">
          {formatDuration(stats.avg_build_time_ms)}
        </div>
        <div className="mt-2 flex flex-wrap gap-1 items-center">
          {stats.models_used && stats.models_used.length > 0 ? (
            stats.models_used.map((model) => (
              <span
                key={model}
                className="text-[10px] px-1.5 py-0.5 rounded bg-purple-950/60 border border-purple-800/40 text-purple-300 font-mono truncate max-w-[120px]"
                title={model}
              >
                {model}
              </span>
            ))
          ) : (
            <span className="text-[11px] text-mileage-muted font-mono">
              Ollama local runner
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
