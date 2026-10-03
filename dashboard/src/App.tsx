import React from 'react';
import { useDashboard } from './hooks/useDashboard';
import { Header } from './components/Header';
import { StatsCards } from './components/StatsCards';
import { PipelineView } from './components/PipelineView';
import { CurrentBuildView } from './components/CurrentBuildView';
import { BuildHistoryTable } from './components/BuildHistoryTable';

export function App() {
  const { status, stats, history, isConnected, lastUpdated, refresh, recentEvents } =
    useDashboard();

  return (
    <div className="min-h-screen bg-mileage-bg text-mileage-text flex flex-col selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Top Navigation / Status Header */}
      <Header
        isConnected={isConnected}
        lastUpdated={lastUpdated}
        onRefresh={refresh}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 py-6 space-y-6">
        {/* KPI / Statistics Row */}
        <section aria-label="Metrics Overview">
          <StatsCards stats={stats} />
        </section>

        {/* Live Multi-Agent Pipeline */}
        <section aria-label="Live Agent Pipeline">
          <PipelineView status={status} />
        </section>

        {/* 2-Column Split: Active Build + Live Event Stream */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <CurrentBuildView status={status} />
          </div>

          {/* Live Activity Stream (Feed) */}
          <div className="glass-card p-5 border border-mileage-border/70 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-mileage-border/50">
                <div className="flex items-center gap-2">
                  <span className="text-base font-semibold text-white">Live Stream</span>
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                </div>
                <span className="text-xs font-mono text-mileage-muted">SSE Feed</span>
              </div>

              <div className="mt-3 space-y-2 max-h-64 overflow-y-auto pr-1">
                {recentEvents.length === 0 ? (
                  <div className="text-center py-8 text-xs font-mono text-slate-500">
                    Awaiting SSE broadcast events...
                  </div>
                ) : (
                  recentEvents.slice(0, 10).map((ev, i) => (
                    <div
                      key={i}
                      className="p-2 rounded bg-mileage-surface/60 border border-mileage-border/40 font-mono text-[11px]"
                    >
                      <div className="flex items-center justify-between text-mileage-muted text-[10px]">
                        <span className="text-cyan-400 uppercase font-semibold">
                          {ev.event_type}
                        </span>
                        <span>
                          {ev.timestamp
                            ? new Date(ev.timestamp).toLocaleTimeString()
                            : ''}
                        </span>
                      </div>
                      <div className="mt-1 text-slate-300 truncate">
                        {ev.data?.message ||
                          ev.data?.goal ||
                          ev.data?.status ||
                          JSON.stringify(ev.data).slice(0, 50)}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-mileage-border/40 text-[10px] text-mileage-muted font-mono flex justify-between">
              <span>Read-only client</span>
              <span className="text-cyan-400">Python CLI is truth</span>
            </div>
          </div>
        </section>

        {/* Historical Runs */}
        <section aria-label="Build History">
          <BuildHistoryTable history={history} />
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-mileage-border/50 bg-mileage-surface/40 py-4 px-4 text-center text-xs text-mileage-muted font-mono">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            m.AI.leage Command Center • 100% Local Multi-Agent Execution
          </div>
          <div className="text-slate-500">
            Visit <span className="text-cyan-400">http://&lt;your-ip&gt;:3000</span> on mobile
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
