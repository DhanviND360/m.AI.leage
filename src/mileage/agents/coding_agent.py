"""Autonomous Local Coding Agent with inspect-plan-edit-test-repair loop and stagnation controls."""

import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from mileage.agents.base import BaseAgent
from mileage.agents.coding_schemas import (
    AgentEvent,
    AgentPhase,
    AgentStatus,
    CodingSessionRecord,
    CodingStepRecord,
)
from mileage.agents.project_tester import ProjectTester, TestRunResult
from mileage.agents.stagnation import StagnationDetector, StagnationReport
from mileage.core.logger import logger, setup_logger
from mileage.core.workspace import WorkspaceManager
from mileage.metrics.schemas import MetricRecord
from mileage.metrics.tracker import MetricsTracker
from mileage.models.ollama_client import OllamaClient
from mileage.models.schemas import ChatMessage, MessageRole
from mileage.tools.base import BaseTool, ToolResult
from mileage.tools.execution_tools import ExecuteCommandTool
from mileage.tools.file_tools import (
    FindWorkspaceFilesTool,
    ListWorkspaceFilesTool,
    PatchWorkspaceFileTool,
    ReadWorkspaceFileTool,
    SearchWorkspaceFilesTool,
    WorkspaceOverviewTool,
    WriteWorkspaceFileTool,
)
from mileage.tools.permissions import ToolPermission


CODING_AGENT_SYSTEM_PROMPT = """\
You are m.AI.leage Coding Agent — an autonomous local software engineer.
You inspect workspaces, plan modifications, edit files, and repair code based on test results.

RULES:
1. Always output valid JSON matching the format below. No markdown outside JSON.
2. In the INSPECT phase: call read_file, search_files, or find_files to understand the code.
3. In the EDIT phase: call write_file or patch_file to apply precise code changes.
4. Only edit files within the workspace root.
5. In the REPAIR phase: address the exact test failure or error trace provided.
6. When tests pass and requirements are satisfied, set phase to "complete".

OUTPUT FORMAT:
{
  "thought": "Reasoning about the current state, what to inspect or edit",
  "phase": "inspect" | "plan" | "edit" | "repair" | "complete",
  "summary": "Short summary of actions taken or conclusion",
  "actions": [
    {
      "tool": "read_file" | "write_file" | "patch_file" | "search_files" | "find_files",
      "args": { ... }
    }
  ]
}

TOOLS AVAILABLE:
- read_file(relative_path, start_line, end_line)
- write_file(relative_path, content)
- patch_file(relative_path, target, replacement)
- search_files(query, is_regex, file_pattern)
- find_files(pattern)
"""


class CodingAgent(BaseAgent):
    """Local coding agent implementing the inspect → plan → edit → test → repair loop."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model_name: str,
        workspace: WorkspaceManager,
        permission: Optional[ToolPermission] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
        max_iterations: int = 10,
        repeated_error_threshold: int = 3,
        unchanged_edits_threshold: int = 2,
        stalled_threshold: int = 3,
        auto_test: bool = True,
        custom_test_cmd: Optional[str] = None,
        temperature: float = 0.2,
        dry_run: bool = False,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
    ):
        super().__init__(name="CodingAgent", system_prompt=CODING_AGENT_SYSTEM_PROMPT)
        self.client = ollama_client
        self.model_name = model_name
        self.workspace = workspace
        self.permission = permission or ToolPermission(
            allowed_directories=[str(workspace.root_dir)]
        )
        self.metrics_tracker = metrics_tracker
        self.max_iterations = max_iterations
        self.auto_test = auto_test
        self.custom_test_cmd = custom_test_cmd
        self.temperature = temperature
        self.dry_run = dry_run
        self.on_event = on_event

        # Stagnation Detector
        self.stagnation_detector = StagnationDetector(
            repeated_error_threshold=repeated_error_threshold,
            unchanged_edits_threshold=unchanged_edits_threshold,
            max_iterations=max_iterations,
            stalled_threshold=stalled_threshold,
        )

        # Native Project Tester
        self.project_tester = ProjectTester(
            workspace_root=workspace.root_dir,
            permission=self.permission,
            custom_test_cmd=custom_test_cmd,
        )

        # Register Controlled Tools
        self._init_tools()

    def _init_tools(self) -> None:
        """Register all controlled workspace tools."""
        self.register_tool(ReadWorkspaceFileTool(self.workspace, self.permission))
        self.register_tool(WriteWorkspaceFileTool(self.workspace, self.permission))
        self.register_tool(PatchWorkspaceFileTool(self.workspace, self.permission))
        self.register_tool(SearchWorkspaceFilesTool(self.workspace, self.permission))
        self.register_tool(FindWorkspaceFilesTool(self.workspace, self.permission))
        self.register_tool(ListWorkspaceFilesTool(self.workspace))
        self.register_tool(WorkspaceOverviewTool(self.workspace))
        self.register_tool(ExecuteCommandTool(self.workspace, self.permission))

    def _emit(self, event_type: str, phase: AgentPhase, message: str, iteration: int = 0, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event for real-time UI display."""
        if self.on_event:
            ev = AgentEvent(
                event_type=event_type,
                iteration=iteration,
                phase=phase,
                message=message,
                data=data or {},
            )
            try:
                self.on_event(ev)
            except Exception as e:
                logger.debug("Error in event callback: %s", e)

    def _call_model(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Query Ollama model and return parsed JSON action plan with token estimates."""
        start_t = time.perf_counter()
        raw_text = ""
        prompt_tokens = sum(max(1, len(m.content) // 4) for m in messages)
        completion_tokens = 0

        try:
            # Attempt structured JSON mode
            payload = {
                "model": self.model_name,
                "messages": [{"role": m.role.value, "content": m.content} for m in messages],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 3072,
                },
            }

            self.client.start_daemon(timeout_seconds=5.0)
            with self.client._get_http_client(timeout=90.0) as http_client:
                resp = http_client.post("/api/chat", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data.get("message", {}).get("content", "")
                    prompt_tokens = data.get("prompt_eval_count", prompt_tokens)
                    completion_tokens = data.get("eval_count", max(1, len(raw_text) // 4))
                else:
                    # Fallback to standard chat
                    chat_resp = self.client.chat(messages=messages, model=self.model_name, temperature=self.temperature)
                    raw_text = chat_resp.get("message", {}).get("content", "")
                    prompt_tokens = chat_resp.get("prompt_eval_count", prompt_tokens)
                    completion_tokens = chat_resp.get("eval_count", max(1, len(raw_text) // 4))

        except Exception as e:
            logger.warning("Model invocation failed (%s), attempting text-only fallback: %s", type(e).__name__, e)
            try:
                chat_resp = self.client.chat(messages=messages, model=self.model_name, temperature=self.temperature)
                raw_text = chat_resp.get("message", {}).get("content", "")
            except Exception as inner_e:
                logger.error("All model invocation attempts failed: %s", inner_e)
                raise inner_e

        duration_ms = (time.perf_counter() - start_t) * 1000
        parsed = self._extract_json(raw_text)

        return {
            "parsed": parsed,
            "raw_text": raw_text,
            "latency_ms": duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract valid JSON from raw model output, handling markdown blocks."""
        if not text:
            return {"phase": "inspect", "thought": "No response text", "actions": []}

        clean = text.strip()

        # Handle ```json ... ``` blocks
        if "```" in clean:
            matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
            if matches:
                clean = matches[0].strip()

        # Direct JSON parse
        try:
            return json.loads(clean)
        except Exception:
            pass

        # Try to find first { and last }
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(clean[start : end + 1])
            except Exception:
                pass

        return {
            "phase": "inspect",
            "thought": "Failed to parse JSON, defaulting to inspection",
            "summary": clean[:200],
            "actions": [],
        }

    def execute_tool_action(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a controlled tool action safely."""
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Unknown tool: '{tool_name}'",
                output_str=f"Tool '{tool_name}' is not recognized.",
            )

        try:
            return tool.run(**args)
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                output_str=f"Error executing tool '{tool_name}': {str(e)}",
            )

    def run(self, prompt: str) -> str:
        """Synchronous run fulfilling BaseAgent abstract protocol."""
        session = self.execute_task(prompt)
        return session.final_summary or f"Session completed with status: {session.status.value}"

    def execute_task(self, goal: str) -> CodingSessionRecord:
        """Run the complete autonomous execution loop: inspect → plan → edit → test → repair."""
        session_id = f"code_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        session_start_t = time.perf_counter()

        # Setup dedicated session logger
        logs_dir = self.workspace.logs_dir if self.workspace.is_initialized() else self.workspace.root_dir / ".mileage" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        session_log_file = logs_dir / f"coding_agent_{session_id}.log"
        sess_logger = setup_logger(f"mileage.agent.{session_id}", log_file=session_log_file)
        sess_logger.info("Starting Coding Agent session %s for goal: %s", session_id, goal)

        session = CodingSessionRecord(
            session_id=session_id,
            goal=goal,
            model_name=self.model_name,
            status=AgentStatus.RUNNING,
        )

        self._emit("session_start", AgentPhase.INSPECT, f"Started task: {goal}", iteration=0)

        # Initial workspace overview for context
        overview_tool = self.tools.get("workspace_overview")
        overview_str = overview_tool.run().output_str if overview_tool else ""

        # Seed conversation history
        self.clear_history()
        initial_prompt = (
            f"GOAL:\n{goal}\n\n"
            f"WORKSPACE OVERVIEW:\n{overview_str}\n\n"
            f"TASK INSTRUCTIONS:\n"
            f"1. INSPECT: Read the relevant project files or find symbols.\n"
            f"2. PLAN: Determine exact file changes.\n"
            f"3. EDIT: Use write_file or patch_file to implement the solution.\n"
            f"Begin by inspecting files or proposing your plan."
        )
        self.add_user_message(initial_prompt)

        step_counter = 0

        for iteration in range(1, self.max_iterations + 1):
            self.stagnation_detector.record_iteration_start()
            session.total_iterations = iteration
            self._emit("step_start", AgentPhase.INSPECT, f"Iteration {iteration}/{self.max_iterations}", iteration=iteration)
            sess_logger.info("=== Iteration %d / %d ===", iteration, self.max_iterations)

            # ── 1. MODEL INFERENCE (Inspect / Plan / Edit decision) ──
            step_counter += 1
            call_res = self._call_model(self.messages)
            parsed = call_res["parsed"]
            raw_text = call_res["raw_text"]

            session.total_prompt_tokens += call_res["prompt_tokens"]
            session.total_completion_tokens += call_res["completion_tokens"]
            session.total_tokens += call_res["total_tokens"]

            sess_logger.info("Model response (%d ms): %s", call_res["latency_ms"], raw_text[:300])

            current_phase_str = str(parsed.get("phase", "inspect")).lower()
            try:
                current_phase = AgentPhase(current_phase_str)
            except ValueError:
                current_phase = AgentPhase.INSPECT

            thought = parsed.get("thought", "")
            actions = parsed.get("actions", [])
            summary = parsed.get("summary", "")

            # Record model step
            session.steps.append(
                CodingStepRecord(
                    step_index=step_counter,
                    iteration=iteration,
                    phase=current_phase,
                    tool_output_summary=thought or summary,
                    model=self.model_name,
                    prompt_tokens=call_res["prompt_tokens"],
                    completion_tokens=call_res["completion_tokens"],
                    total_tokens=call_res["total_tokens"],
                    latency_ms=round(call_res["latency_ms"], 2),
                )
            )

            self.add_assistant_message(raw_text)

            # ── 2. EXECUTE ACTIONS (Inspect / Edit) ──
            action_results_text = []

            for act in actions:
                tool_name = act.get("tool", "")
                args = act.get("args", {})
                step_counter += 1

                sess_logger.info("Executing action: %s with args: %s", tool_name, args)

                if self.dry_run and tool_name in ["write_file", "patch_file"]:
                    sess_logger.info("[Dry Run] Skipped write tool %s", tool_name)
                    res = ToolResult(
                        success=True,
                        output_str=f"[Dry Run] Skipped modifying {args.get('relative_path')}",
                    )
                else:
                    res = self.execute_tool_action(tool_name, args)

                # Stagnation check on edits
                if tool_name in ["write_file", "patch_file"] and res.success:
                    path = args.get("relative_path", "")
                    if path and path not in session.files_modified:
                        session.files_modified.append(path)

                    is_changed = res.metadata.get("is_changed", True)
                    content = args.get("content")

                    stag_report = self.stagnation_detector.record_edit(
                        path=path, is_changed=is_changed, content=content
                    )

                    self._emit(
                        "edit",
                        AgentPhase.EDIT,
                        f"Edited '{path}' (is_changed={is_changed})",
                        iteration=iteration,
                        data={"path": path, "is_changed": is_changed},
                    )

                    if stag_report and stag_report.is_stagnant:
                        session.status = AgentStatus.STAGNATED
                        session.stagnation_report = stag_report
                        session.final_summary = f"Stagnation Halting: {stag_report.message}"
                        sess_logger.warning("Stagnation triggered on edit: %s", stag_report.message)
                        self._emit("stagnation", AgentPhase.EDIT, stag_report.message, iteration=iteration)
                        break
                elif tool_name in ["read_file", "search_files", "find_files"]:
                    self._emit(
                        "inspect",
                        AgentPhase.INSPECT,
                        f"Tool {tool_name}: {res.output_str[:80]}",
                        iteration=iteration,
                    )

                action_results_text.append(f"Tool [{tool_name}] output:\n{res.output_str}")

                session.steps.append(
                    CodingStepRecord(
                        step_index=step_counter,
                        iteration=iteration,
                        phase=current_phase,
                        tool_name=tool_name,
                        tool_args=args,
                        tool_output_summary=res.output_str[:300],
                        success=res.success,
                        error_message=res.error,
                        latency_ms=res.duration_ms,
                    )
                )

            if session.status == AgentStatus.STAGNATED:
                break

            # ── 3. AUTOMATIC TESTING (Native project test runner) ──
            test_res: Optional[TestRunResult] = None
            if self.auto_test and not self.dry_run:
                step_counter += 1
                self._emit("test", AgentPhase.TEST, "Running project-native tests...", iteration=iteration)
                test_res = self.project_tester.run()
                session.tests_passed = test_res.passed
                session.test_summary = (
                    f"{test_res.framework}: {test_res.tests_passed} passed, "
                    f"{test_res.tests_failed} failed, {test_res.tests_errors} errors"
                )
                sess_logger.info("Test Run Result: %s (passed=%s)", session.test_summary, test_res.passed)

                session.steps.append(
                    CodingStepRecord(
                        step_index=step_counter,
                        iteration=iteration,
                        phase=AgentPhase.TEST,
                        tool_name="project_tester",
                        tool_args={"framework": test_res.framework, "command": test_res.command},
                        tool_output_summary=session.test_summary,
                        success=test_res.passed,
                        error_message=test_res.failure_summary if not test_res.passed else None,
                        latency_ms=test_res.duration_ms,
                    )
                )

                self._emit(
                    "test_result",
                    AgentPhase.TEST,
                    session.test_summary,
                    iteration=iteration,
                    data={"passed": test_res.passed, "details": session.test_summary},
                )

                # Check if all tests pass and model says complete or we achieved objective
                if test_res.passed and (current_phase == AgentPhase.COMPLETE or session.files_modified):
                    session.status = AgentStatus.COMPLETED
                    session.final_summary = (
                        f"All tests passed successfully ({session.test_summary}). "
                        f"Modified {len(session.files_modified)} file(s)."
                    )
                    sess_logger.info("Goal Achieved! All tests pass.")
                    self._emit("complete", AgentPhase.COMPLETE, session.final_summary, iteration=iteration)
                    break

            # ── 4. REPAIR PHASE / ERROR FEEDBACK ──
            if test_res and not test_res.passed:
                failure_text = test_res.failure_summary or test_res.raw_output
                self._emit("repair", AgentPhase.REPAIR, f"Test failed: {failure_text[:120]}", iteration=iteration)

                # Check error stagnation
                stag_report = self.stagnation_detector.record_error(failure_text)
                if stag_report and stag_report.is_stagnant:
                    session.status = AgentStatus.STAGNATED
                    session.stagnation_report = stag_report
                    session.final_summary = f"Stagnation Halting: {stag_report.message}"
                    sess_logger.warning("Stagnation triggered on test error: %s", stag_report.message)
                    self._emit("stagnation", AgentPhase.REPAIR, stag_report.message, iteration=iteration)
                    break

                # Prepare repair prompt
                repair_prompt = (
                    f"TESTS FAILED:\n{failure_text}\n\n"
                    f"Please analyze the failure, inspect relevant code, and apply targeted repairs."
                )
                self.add_user_message(repair_prompt)
            else:
                # If actions were executed, provide their output back to conversation
                if action_results_text:
                    feedback = "\n\n".join(action_results_text)
                    if current_phase == AgentPhase.COMPLETE:
                        session.status = AgentStatus.COMPLETED
                        session.final_summary = summary or "Coding task completed."
                        self._emit("complete", AgentPhase.COMPLETE, session.final_summary, iteration=iteration)
                        break
                    else:
                        self.add_user_message(f"Action Results:\n{feedback}\nProceed to next step.")
                else:
                    if current_phase == AgentPhase.COMPLETE:
                        session.status = AgentStatus.COMPLETED
                        session.final_summary = summary or "Coding task completed."
                        self._emit("complete", AgentPhase.COMPLETE, session.final_summary, iteration=iteration)
                        break
                    self.add_user_message("Please produce concrete write_file or patch_file actions.")

            # Check iteration limit stagnation
            iter_stag = self.stagnation_detector.record_iteration_end(tests_passed=session.tests_passed)
            if iter_stag and iter_stag.is_stagnant:
                session.status = AgentStatus.STAGNATED
                session.stagnation_report = iter_stag
                session.final_summary = f"Stagnation Halting: {iter_stag.message}"
                sess_logger.warning("Stagnation triggered at iteration end: %s", iter_stag.message)
                self._emit("stagnation", AgentPhase.REPAIR, iter_stag.message, iteration=iteration)
                break

        # If loop exited without setting status
        if session.status == AgentStatus.RUNNING:
            if session.tests_passed:
                session.status = AgentStatus.COMPLETED
                session.final_summary = "Task completed with all tests passing."
            else:
                session.status = AgentStatus.FAILED
                session.final_summary = "Task halted before completing all requirements."

        # Finalize telemetry and metrics
        total_time_ms = (time.perf_counter() - session_start_t) * 1000
        session.total_duration_ms = round(total_time_ms, 2)
        session.total_steps = len(session.steps)
        session.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Save session file in .mileage/sessions/<session_id>.json
        if self.workspace.is_initialized():
            sessions_dir = self.workspace.mileage_dir / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            session_file = sessions_dir / f"{session.session_id}.json"
            session_file.write_text(session.model_dump_json(indent=2), encoding="utf-8")
            sess_logger.info("Saved session record to %s", session_file)

        # Record to MetricsTracker
        if self.metrics_tracker:
            self.metrics_tracker.record(
                MetricRecord(
                    command="code",
                    model=self.model_name,
                    prompt_tokens=session.total_prompt_tokens,
                    completion_tokens=session.total_completion_tokens,
                    total_tokens=session.total_tokens,
                    latency_ms=session.total_duration_ms,
                    success=(session.status == AgentStatus.COMPLETED),
                    error_type=session.stagnation_report.reason.value if session.stagnation_report and session.stagnation_report.reason else None,
                )
            )

        sess_logger.info("Session %s finished: status=%s duration=%.2fms tokens=%d", session_id, session.status.value, total_time_ms, session.total_tokens)
        return session
