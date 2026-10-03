import { useState, useEffect, useRef, useCallback } from 'react';
import { PipelineStatus, DashboardStats, BuildRecord, DashboardEvent } from '../types';

const DEFAULT_STATUS: PipelineStatus = {
  stage: 'idle',
  goal: '',
  model_name: '',
  iteration: 0,
  max_iterations: 0,
  message: 'Standby for commands or agent sessions...',
  requirements_passed: 0,
  requirements_total: 0,
  started_at: null,
  elapsed_ms: 0,
};

const DEFAULT_STATS: DashboardStats = {
  projects_built: 0,
  total_tokens_used: 0,
  tokens_saved_estimate: 0,
  estimated_cost_avoided: 0,
  avg_requirement_satisfaction: 0,
  tasks_completed_locally: 100,
  total_builds_today: 0,
  models_used: [],
  avg_build_time_ms: 0,
};

export function useDashboard() {
  const [status, setStatus] = useState<PipelineStatus>(DEFAULT_STATUS);
  const [stats, setStats] = useState<DashboardStats>(DEFAULT_STATS);
  const [history, setHistory] = useState<BuildRecord[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [recentEvents, setRecentEvents] = useState<DashboardEvent[]>([]);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const eventSourceRef = useRef<EventSource | null>(null);
  const retryTimeoutRef = useRef<number | null>(null);

  // Initial fetch for baseline REST data
  const fetchInitialData = useCallback(async () => {
    try {
      const [resStatus, resStats, resHist] = await Promise.all([
        fetch('/api/status').then((r) => (r.ok ? r.json() : null)),
        fetch('/api/stats').then((r) => (r.ok ? r.json() : null)),
        fetch('/api/history').then((r) => (r.ok ? r.json() : null)),
      ]);

      if (resStatus) setStatus(resStatus);
      if (resStats) setStats(resStats);
      if (Array.isArray(resHist)) setHistory(resHist);
      setLastUpdated(new Date());
    } catch (err) {
      // Backend might still be starting
      console.warn('Initial dashboard fetch waiting for backend...', err);
    }
  }, []);

  useEffect(() => {
    fetchInitialData();

    function connectSSE() {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const es = new EventSource('/api/events');
      eventSourceRef.current = es;

      es.onopen = () => {
        setIsConnected(true);
      };

      es.onmessage = (event) => {
        try {
          const parsed: DashboardEvent = JSON.parse(event.data);
          setLastUpdated(new Date());

          setRecentEvents((prev) => [parsed, ...prev.slice(0, 49)]);

          if (parsed.event_type === 'connected') {
            if (parsed.data?.pipeline) setStatus(parsed.data.pipeline);
            if (parsed.data?.stats) setStats(parsed.data.stats);
          } else if (parsed.event_type === 'pipeline_update') {
            setStatus(parsed.data as PipelineStatus);
          } else if (parsed.event_type === 'build_complete') {
            const newBuild = parsed.data as BuildRecord;
            setHistory((prev) => [newBuild, ...prev]);
            // Re-fetch stats to sync computed metrics
            fetch('/api/stats')
              .then((r) => (r.ok ? r.json() : null))
              .then((data) => data && setStats(data))
              .catch(() => {});
          } else if (parsed.event_type === 'metric') {
            setStats((prev) => ({
              ...prev,
              ...parsed.data,
            }));
          }
        } catch {
          // Keepalive comments or unparsable frames
        }
      };

      es.onerror = () => {
        setIsConnected(false);
        es.close();
        if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
        retryTimeoutRef.current = window.setTimeout(connectSSE, 3000);
      };
    }

    connectSSE();

    // Fallback polling every 5s if SSE disconnects
    const pollInterval = window.setInterval(() => {
      if (!isConnected) {
        fetchInitialData();
      }
    }, 5000);

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      clearInterval(pollInterval);
    };
  }, [fetchInitialData, isConnected]);

  return {
    status,
    stats,
    history,
    isConnected,
    recentEvents,
    lastUpdated,
    refresh: fetchInitialData,
  };
}
