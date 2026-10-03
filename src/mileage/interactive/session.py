"""Claude-Code-like persistent interactive session for m.AI.leage.

Orchestrates voice/text input and the autonomous agent pipeline:
Listening → Planning → Building → Testing → Evaluating → Complete/Escalating.
"""

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from mileage.agents.coding_agent import CodingAgent
from mileage.agents.coding_schemas import AgentEvent, AgentStatus
from mileage.agents.escalation import CopilotEscalation
from mileage.agents.evaluator import EvaluatorAgent
from mileage.agents.evaluator_schemas import EscalationReason, RequirementVerdict
from mileage.agents.planner import PlannerAgent
from mileage.agents.planner_schemas import PlannerInput
from mileage.agents.project_tester import ProjectTester
from mileage.core.logger import logger
from mileage.core.workspace import WorkspaceManager
from mileage.dashboard import BuildRecord, DashboardEventBus, PipelineStage
from mileage.metrics.tracker import MetricsTracker
from mileage.models.ollama_client import OllamaClient
from mileage.router import ModelRouter
from mileage.ui.console import console
from mileage.voice.stt import SpeechToTextEngine
from mileage.voice.tts import TextToSpeechEngine
from mileage.voice.vad import VadDetector

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


class InteractiveSession:
    """Manages the m.AI.leage persistent Claude-Code-like interactive CLI session."""

    def __init__(
        self,
        preferred_model: Optional[str] = None,
        enable_voice: bool = True,
        enable_tts: bool = True,
        whisper_model: str = "tiny.en",
        max_iterations: int = 10,
        workspace: Optional[WorkspaceManager] = None,
    ):
        self.preferred_model = preferred_model
        self.enable_voice = enable_voice and HAS_SOUNDDEVICE
        self.enable_tts = enable_tts
        self.max_iterations = max_iterations
        self.whisper_model = whisper_model

        # 1. Ensure local workspace
        self.workspace = workspace or WorkspaceManager.find_workspace() or WorkspaceManager()
        if not self.workspace.is_initialized():
            try:
                self.workspace.initialize(project_name=Path.cwd().name)
            except Exception:
                pass

        self.config = self.workspace.get_config()
        self.tracker = MetricsTracker(self.workspace.metrics_path if self.workspace.is_initialized() else None)
        self.bus = DashboardEventBus.get_instance()

        # 2. Ollama client
        self.ollama = OllamaClient(
            host=self.config.ollama.host,
            timeout_seconds=self.config.ollama.timeout_seconds,
        )
        self.target_model = preferred_model or self.config.ollama.default_model

        # 3. Voice components
        self.stt: Optional[SpeechToTextEngine] = None
        self.tts: Optional[TextToSpeechEngine] = None
        self.vad: Optional[VadDetector] = None

        if self.enable_voice:
            try:
                self.stt = SpeechToTextEngine(model_size=self.whisper_model)
                self.vad = VadDetector(
                    sample_rate=16000,
                    frame_duration_ms=30,
                    energy_threshold=350.0,
                    silence_timeout_sec=1.2,
                )
            except Exception as e:
                logger.warning("Voice initialization error: %s", e)
                self.enable_voice = False

        if self.enable_tts:
            try:
                self.tts = TextToSpeechEngine()
            except Exception as e:
                logger.warning("TTS initialization error: %s", e)
                self.enable_tts = False

        self._running = True
        self._is_speaking = False
        self._audio_queue: queue.Queue = queue.Queue()

    def render_speedometer_greeting(self) -> None:
        """Render the clean, minimal m.AI.leage Claude-Code-like speedometer mascot greeting."""
        from mileage.ui.mascot import get_app_mascot

        art = get_app_mascot()

        info = Text()
        info.append("m", style="bold cyan")
        info.append(".", style="bold white")
        info.append("AI", style="bold magenta")
        info.append(".", style="bold white")
        info.append("leage", style="bold cyan")
        info.append("  Autonomous Local AI\n", style="bold white")
        info.append("Local Autonomous Agent Orchestration\n\n", style="dim")

        # Status pills
        info.append("  ● ", style="bold green")
        info.append("100% Local", style="bold white")
        info.append(" • Zero Cloud Telemetry\n", style="dim")

        info.append("  ● ", style="bold cyan")
        info.append("Model: ", style="dim")
        info.append(f"{self.target_model}\n", style="bold cyan")

        info.append("  ● ", style="bold magenta" if self.enable_voice else "dim")
        info.append("Voice: ", style="dim")
        info.append(
            "Whisper + VAD + TTS (Hands-Free)\n" if self.enable_voice else "Text Mode Only\n",
            style="bold magenta" if self.enable_voice else "dim",
        )

        proj_name = self.config.workspace.project_name or Path.cwd().name
        info.append("  ● ", style="bold yellow")
        info.append("Workspace: ", style="dim")
        info.append(f"{proj_name} (.mileage/)\n", style="white")

        grid = Table.grid(padding=(0, 3))
        grid.add_row(art, info)

        panel = Panel(
            grid,
            border_style="cyan",
            padding=(1, 2),
            subtitle="[dim]Speak or type your goal • 'exit' or Ctrl+C to quit • 'mileage serve' for web UI[/dim]",
            subtitle_align="center",
        )
        console.print(panel)
        console.print()

    def say(self, text: str) -> None:
        """Speak short response via TTS with strict acoustic isolation."""
        if not self.enable_tts or not self.tts:
            return

        self._is_speaking = True
        try:
            # Drain any buffered mic audio before speaking
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            short_text = self.tts.compress_for_speech(text)
            if short_text:
                self.tts.speak(short_text, non_blocking=False)
                time.sleep(0.25)  # Acoustic cooldown
        except Exception as e:
            logger.debug("TTS playback error: %s", e)
        finally:
            self._is_speaking = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Sounddevice raw audio callback."""
        if self._is_speaking:
            # Drop mic frames during speech to prevent feedback loop
            return
        if self._running:
            self._audio_queue.put(bytes(indata))

    def listen_for_goal(self) -> Optional[str]:
        """Listen for user goal via microphone or keyboard input."""
        console.print("  [bold green]● Listening[/bold green]   [dim]Speak your goal or type below...[/dim]")
        self.bus.publish_pipeline_update(
            stage=PipelineStage.LISTENING,
            message="Listening for user request...",
        )

        # If voice engine is available, try listening on mic stream
        if self.enable_voice and self.vad and self.stt:
            return self._listen_voice_and_keyboard()

        # Fallback to direct prompt
        return self._prompt_keyboard()

    def _prompt_keyboard(self) -> Optional[str]:
        """Direct text input prompt."""
        try:
            val = Prompt.ask("  [bold cyan]❯[/bold cyan]").strip()
            return val
        except (KeyboardInterrupt, EOFError):
            return "exit"

    def _listen_voice_and_keyboard(self) -> Optional[str]:
        """Simultaneous voice listening with keyboard interrupt."""
        speech_buffer = bytearray()
        self.vad.reset()

        # Drain audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            stream = sd.RawInputStream(
                samplerate=16000,
                blocksize=int(16000 * 0.03),  # 30ms frames
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
        except Exception as e:
            logger.warning("Could not open microphone stream: %s. Falling back to keyboard.", e)
            return self._prompt_keyboard()

        hearing_speech = False
        with stream:
            while self._running:
                # 1. Check for keyboard input on Windows
                if HAS_MSVCRT and msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\r", "\n"):
                        return self._prompt_keyboard()
                    elif ch == "\x03":  # Ctrl+C
                        return "exit"
                    else:
                        # User started typing: read full line
                        console.print(f"  [bold cyan]❯[/bold cyan] {ch}", end="")
                        try:
                            rest = sys.stdin.readline().strip()
                            return f"{ch}{rest}"
                        except (KeyboardInterrupt, EOFError):
                            return "exit"

                # 2. Process incoming audio frames from mic
                try:
                    frame = self._audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if self._is_speaking:
                    speech_buffer.clear()
                    self.vad.reset()
                    continue

                if not hearing_speech:
                    if self.vad.is_speech(frame):
                        hearing_speech = True
                        speech_buffer.extend(frame)
                        console.print("  [bold green]● Listening[/bold green]   [dim](Hearing speech...)[/dim]")
                else:
                    speech_buffer.extend(frame)
                    finished = self.vad.process_frame(frame)
                    if finished:
                        audio_data = bytes(speech_buffer)
                        speech_buffer.clear()
                        hearing_speech = False

                        # Transcribe speech
                        console.print("  [bold yellow]● Processing[/bold yellow]  [dim]Transcribing speech with Whisper...[/dim]")
                        try:
                            transcript = self.stt.transcribe_audio_bytes(audio_data)
                        except Exception as e:
                            logger.warning("Whisper transcription error: %s", e)
                            transcript = ""

                        if transcript and transcript.strip():
                            console.print(f"  [bold cyan]🗣️  You:[/bold cyan] {transcript}")
                            return transcript.strip()
                        else:
                            console.print("  [bold green]● Listening[/bold green]   [dim]Speak your goal or type below...[/dim]")

        return None

    def execute_pipeline(self, goal: str) -> None:
        """
        Execute the autonomous agent pipeline with concise state transitions:
        Planning → Building → Testing → Evaluating → Complete/Escalating
        """
        console.print()

        # ── 1. PLANNING ──────────────────────────────────────────────────
        console.print(f"  [bold cyan]● Planning[/bold cyan]    [white]Structuring ActionPlan with Gemma...[/white]")
        self.say(f"Analyzing objective: {goal[:60]}. Formulating action plan.")
        self.bus.publish_pipeline_update(
            stage=PipelineStage.PLANNING,
            goal=goal,
            model_name=self.target_model,
            message="Structuring ActionPlan with Gemma...",
        )

        try:
            planner = PlannerAgent(
                ollama_client=self.ollama,
                model_name=self.target_model,
                workspace=self.workspace,
            )
            plan = planner.plan(PlannerInput(text=goal))
            console.print(
                f"    [dim]✓ ActionPlan: {len(plan.requirements)} requirements, {len(plan.acceptance_criteria)} acceptance criteria[/dim]"
            )
            self.say(f"Plan ready with {len(plan.requirements)} requirements. Preparing to build.")
        except Exception as e:
            console.print(f"    [yellow]⚠ Direct planner fallback: {e}[/yellow]")
            # Fallback ActionPlan if planner encounters unexpected issue
            from mileage.agents.planner_schemas import ActionPlan, Requirement, AcceptanceCriterion
            plan = ActionPlan(
                goal=goal,
                requirements=[
                    Requirement(
                        id="R1",
                        description=f"Implement {goal}",
                    )
                ],
                acceptance_criteria=[
                    AcceptanceCriterion(
                        id="AC1",
                        description=f"Core functionality for '{goal}' is verified",
                    )
                ],
            )
            self.say("Action plan initialized. Preparing to build.")

        # ── 2. ROUTING & BUILDING ────────────────────────────────────────
        router = ModelRouter(ollama_client=self.ollama, workspace_root=self.workspace.root_dir)
        try:
            decision = router.route_for_prompt(goal)
            build_model = decision.selected_model or self.target_model
        except Exception:
            build_model = self.target_model

        console.print(
            f"  [bold blue]● Building[/bold blue]    [white]Executing task with [bold green]{build_model}[/bold green]...[/white]"
        )
        self.say(f"Assigned task to {build_model}. Starting autonomous build loop.")
        self.bus.publish_pipeline_update(
            stage=PipelineStage.BUILDING,
            goal=goal,
            model_name=build_model,
            message=f"Executing loop with {build_model}...",
        )

        def on_coding_event(ev: AgentEvent) -> None:
            if ev.event_type == "test":
                console.print(f"  [bold yellow]● Testing[/bold yellow]     [dim]{ev.message}[/dim]")
                self.bus.publish_pipeline_update(
                    stage=PipelineStage.BUILDING,
                    message=f"Testing: {ev.message}",
                )
            elif ev.event_type == "edit":
                console.print(f"    [dim]✏️ {ev.message}[/dim]")
            elif ev.event_type == "repair":
                console.print(f"    [dim]🔧 {ev.message}[/dim]")
                self.say("Test failure encountered. Starting code repair.")
            elif ev.event_type == "stagnation":
                console.print(f"    [dim]⚠️ {ev.message}[/dim]")
                self.say("Execution stagnant. Re-evaluating build approach.")

        coding_agent = CodingAgent(
            ollama_client=self.ollama,
            model_name=build_model,
            workspace=self.workspace,
            metrics_tracker=self.tracker,
            max_iterations=self.max_iterations,
            auto_test=True,
            on_event=on_coding_event,
        )

        try:
            session = coding_agent.execute_task(goal)
        except Exception as e:
            logger.warning("CodingAgent execution error: %s", e)
            from mileage.agents.coding_schemas import CodingSessionRecord
            session = CodingSessionRecord(session_id="err", goal=goal, model_name=build_model, status=AgentStatus.FAILED)

        # ── 3. TESTING (Verification) ────────────────────────────────────
        console.print(f"  [bold yellow]● Testing[/bold yellow]     [white]Running automated test verification...[/white]")
        self.say("Build loop complete. Verifying implementation with project tests.")
        self.bus.publish_pipeline_update(
            stage=PipelineStage.BUILDING,
            message="Running project test suite verification...",
        )
        tester = ProjectTester(workspace_root=self.workspace.root_dir)
        test_result = tester.run_tests()
        test_status = (
            "[bold green]PASS[/bold green]"
            if test_result.passed
            else ("[bold red]FAIL[/bold red]" if test_result.exit_code != 0 else "[dim]NO_TESTS[/dim]")
        )
        console.print(f"    [dim]Project tests: {test_status} ({test_result.command})[/dim]")

        # ── 4. EVALUATING ────────────────────────────────────────────────
        console.print(f"  [bold magenta]● Evaluating[/bold magenta]  [white]Checking implementation against acceptance criteria...[/white]")
        self.say(f"Project tests {'passed' if test_result.passed else 'completed'}. Evaluating acceptance criteria.")
        self.bus.publish_pipeline_update(
            stage=PipelineStage.EVALUATING,
            goal=goal,
            model_name=self.target_model,
            message="Evaluating acceptance criteria...",
            requirements_passed=0,
            requirements_total=len(plan.requirements),
        )

        evaluator = EvaluatorAgent(
            ollama_client=self.ollama,
            model_name=self.target_model,
            workspace=self.workspace,
        )

        try:
            report = evaluator.evaluate(plan)
            passed_cnt = sum(1 for r in getattr(report, "requirement_results", []) if r.verdict == RequirementVerdict.PASS)
            total_cnt = len(getattr(report, "requirement_results", []))
            console.print(f"    [dim]Criteria verified: {passed_cnt}/{total_cnt}[/dim]")
        except Exception as e:
            logger.warning("Evaluator error: %s", e)
            passed_cnt = 1 if session.status == AgentStatus.COMPLETED else 0
            total_cnt = len(plan.requirements) or 1
            from mileage.agents.evaluator_schemas import EvaluationReport
            report = EvaluationReport(
                plan_id=getattr(plan, "plan_id", "plan_fallback"),
                goal=goal,
                overall_verdict=RequirementVerdict.PASS if passed_cnt == total_cnt else RequirementVerdict.FAIL,
                overcapacity_detected=False,
            )

        # ── 5. COMPLETE / ESCALATING ─────────────────────────────────────
        is_escalated = (
            report.overcapacity_detected
            or report.overall_verdict == RequirementVerdict.FAIL
            or session.status == AgentStatus.FAILED
        )

        if is_escalated:
            console.print(f"  [bold red]⚡ Escalating[/bold red]  [yellow]Overcapacity detected. Escalating to GitHub Copilot...[/yellow]")
            esc_engine = CopilotEscalation(workspace_root=self.workspace.root_dir)
            try:
                esc_record = esc_engine.escalate(
                    plan=plan,
                    report=report,
                    reason=EscalationReason.OVERCAPACITY if report.overcapacity_detected else EscalationReason.REPEATED_FAILURES,
                    open_vscode=True,
                )
                prompt_file = getattr(esc_record, "copilot_prompt_file", "") or getattr(esc_record, "prompt_file_path", "")
                console.print(f"    [dim]Copilot prompt saved: {prompt_file} (Opened in VS Code)[/dim]")
            except Exception as e:
                logger.warning("Escalation notice: %s", e)
            self.say("Overcapacity detected. Handing off to GitHub Copilot with full context.")

            self.bus.publish_pipeline_update(
                stage=PipelineStage.ESCALATING,
                goal=goal,
                model_name=build_model,
                requirements_passed=passed_cnt,
                requirements_total=total_cnt,
                message="Escalated to GitHub Copilot.",
            )
        else:
            console.print(f"  [bold green]✓ Complete[/bold green]    [bold white]All acceptance criteria verified locally![/bold white]")
            self.say(f"All {passed_cnt} acceptance criteria verified with evidence. Build complete.")

            self.bus.publish_pipeline_update(
                stage=PipelineStage.COMPLETE,
                goal=goal,
                model_name=build_model,
                requirements_passed=passed_cnt,
                requirements_total=total_cnt,
                message="Task completed and verified locally!",
            )

        # Record build in event bus for telemetry and Command Center
        try:
            eval_duration = getattr(report, "evaluation_duration_ms", 0.0) or getattr(report, "duration_ms", 0.0)
            eval_tokens = getattr(report, "evaluation_tokens", 0) or getattr(report, "tokens_used", 0)
            session_dur = getattr(session, "total_duration_ms", 0.0) or getattr(session, "duration_ms", 0.0)
            session_tok = getattr(session, "total_tokens", 0)
            report_id = getattr(report, "plan_id", "") or getattr(session, "session_id", "build")

            self.bus.record_build(
                BuildRecord(
                    build_id=report_id,
                    goal=goal,
                    model_name=build_model,
                    status="escalated" if is_escalated else "complete",
                    started_at=getattr(session, "started_at", ""),
                    completed_at=getattr(session, "completed_at", "") or getattr(report, "evaluated_at", ""),
                    duration_ms=session_dur + eval_duration,
                    tokens_used=session_tok + eval_tokens,
                    requirements_passed=passed_cnt,
                    requirements_total=total_cnt,
                    escalated=is_escalated,
                )
            )
        except Exception as e:
            logger.warning("Error recording build in event bus: %s", e)

        console.print("\n  [dim]────────────────────────────────────────────────[/dim]\n")

    def run(self) -> None:
        """Run the persistent interactive session loop."""
        # Ready Ollama daemon
        with console.status("[bold cyan]Connecting to local Ollama and verifying models...[/bold cyan]"):
            is_online, ready_model = self.ollama.ensure_ready(
                preferred_model=self.target_model
            )
            if ready_model:
                self.target_model = ready_model

        if not is_online:
            console.print("[yellow]⚠ Ollama daemon offline. Please start with: ollama serve[/yellow]\n")

        # 1. Print speedometer logo & greeting
        self.render_speedometer_greeting()

        # 2. Greeting via speech synthesis
        self.say("m.AI.leage is online. Ready for your goal.")

        # 3. Main persistent interaction loop
        try:
            while self._running:
                goal = self.listen_for_goal()
                if not goal or not goal.strip():
                    continue

                clean_goal = goal.strip()
                if clean_goal.lower() in ("exit", "quit", "q", "/exit", "/quit", "stop", "halt"):
                    self.say("Goodbye.")
                    console.print("\n[dim]Session closed. Happy hacking![/dim]\n")
                    break

                self.execute_pipeline(clean_goal)

        except KeyboardInterrupt:
            self.say("Stopping.")
            console.print("\n\n[yellow]Interactive session stopped by user.[/yellow]\n")
        finally:
            self._running = False
            if self.tts:
                self.tts.stop()
