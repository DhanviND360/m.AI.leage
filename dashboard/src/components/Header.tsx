import React from 'react';

interface HeaderProps {
  isConnected: boolean;
  lastUpdated: Date;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({ isConnected, lastUpdated, onRefresh }) => {
  return (
    <header className="border-b border-mileage-border/80 bg-mileage-surface/70 backdrop-blur-md sticky top-0 z-30 px-4 lg:px-8 py-3.5 transition-all">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
        {/* Left: Brand / Title */}
        <div className="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-start">
          <div className="flex items-center gap-2.5">
            <img
              src="/speedy_mascot.png"
              alt="m.AI.leage Mascot"
              className="w-9 h-9 object-contain drop-shadow-[0_0_8px_rgba(34,211,238,0.3)] hover:scale-105 transition-transform"
            />
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-lg sm:text-xl tracking-tight text-white font-mono">
                  m<span className="text-cyan-400">.AI.</span>leage
                </span>
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800/60 uppercase tracking-wider">
                  Command Center
                </span>
              </div>
              <p className="text-[11px] text-mileage-muted hidden sm:block">
                Local-first AI Agent Orchestration & Workspace Telemetry
              </p>
            </div>
          </div>

          {/* Mobile SSE indicator */}
          <div className="sm:hidden flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-emerald-400 animate-ping' : 'bg-amber-400'
              }`}
            />
            <span className="text-xs text-mileage-muted font-mono">
              {isConnected ? 'LIVE' : 'SYNCING'}
            </span>
          </div>
        </div>

        {/* Right: Remote Phone Hint & Status */}
        <div className="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-end">
          <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-md bg-mileage-card border border-mileage-border text-xs text-mileage-muted font-mono">
            <span className="text-cyan-400">📱 Mobile View:</span>
            <span>Connect on same Wi-Fi via your host IP</span>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-full bg-mileage-card border border-mileage-border">
              <span
                className={`w-2 h-2 rounded-full ${
                  isConnected
                    ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]'
                    : 'bg-amber-400 animate-pulse'
                }`}
              />
              <span className="text-xs font-medium text-slate-300">
                {isConnected ? 'Live SSE Stream' : 'Connecting...'}
              </span>
            </div>

            <button
              onClick={onRefresh}
              title={`Last updated ${lastUpdated.toLocaleTimeString()}`}
              className="px-2.5 py-1 text-xs font-mono text-slate-400 hover:text-white bg-mileage-card hover:bg-mileage-border/80 border border-mileage-border rounded transition-colors flex items-center gap-1.5"
            >
              <svg
                className="w-3.5 h-3.5 text-cyan-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              <span>{lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
