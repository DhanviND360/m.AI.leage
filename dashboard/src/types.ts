export type PipelineStage =
  | 'idle'
  | 'listening'
  | 'planning'
  | 'routing'
  | 'building'
  | 'evaluating'
  | 'escalating'
  | 'complete'
  | 'failed';

export interface PipelineStatus {
  stage: PipelineStage;
  goal: string;
  model_name: string;
  iteration: number;
  max_iterations: number;
  message: string;
  requirements_passed: number;
  requirements_total: number;
  started_at?: string | null;
  elapsed_ms: number;
}

export interface BuildRecord {
  build_id: string;
  goal: string;
  model_name: string;
  status: 'complete' | 'failed' | 'escalated' | string;
  started_at: string;
  completed_at: string;
  duration_ms: number;
  tokens_used: number;
  requirements_passed: number;
  requirements_total: number;
  escalated: boolean;
}

export interface DashboardStats {
  projects_built: number;
  total_tokens_used: number;
  tokens_saved_estimate: number;
  estimated_cost_avoided: number;
  avg_requirement_satisfaction: number;
  tasks_completed_locally: number;
  total_builds_today: number;
  models_used: string[];
  avg_build_time_ms: number;
}

export interface DashboardEvent {
  event_type: string;
  timestamp: string;
  data: Record<string, any>;
}
