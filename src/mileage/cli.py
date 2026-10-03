"""Typer CLI entry point for m.AI.leage."""

import sys
from pathlib import Path
from typing import Optional
import typer
from rich.markdown import Markdown

from mileage import __version__, __app_name__
from mileage.agents.local_agent import LocalAgent
from mileage.agents.planner import PlannerAgent
from mileage.agents.planner_schemas import PlannerInput
from mileage.core.config import MileageConfig
from mileage.core.exceptions import MileageError
from mileage.core.workspace import WorkspaceManager
from mileage.metrics.tracker import MetricsTracker
from mileage.models.ollama_client import OllamaClient
from mileage.agents.coding_agent import CodingAgent
from mileage.agents.coding_schemas import AgentEvent, AgentStatus
from mileage.ui.components import (
    render_action_plan,
    render_coding_session,
    render_error,
    render_metrics_table,
    render_models_table,
    render_success,
    render_workspace_table,
)
from mileage.ui.console import console, print_banner, print_speedometer_greeting
from mileage.ui.doctor_view import DoctorDiagnostics, render_doctor_report

app = typer.Typer(
    name="mileage",
    help=f"{__app_name__} - Lightweight local-first AI CLI & Workspace Telemetry",
    add_completion=False,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Display m.AI.leage version and exit.",
        is_eager=True,
    ),
):
    """m.AI.leage: 100% local AI execution, workspace intelligence, and metrics."""
    if version:
        console.print(f"[bold cyan]{__app_name__}[/bold cyan] version [bold green]{__version__}[/bold green]")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        print_speedometer_greeting(model_name="local", voice_active=True)
        console.print("[bold white]Primary One-Command Experience:[/bold white]")
        console.print("  [bold green]mileage start[/bold green]       Launch interactive agent session (Voice + Plan + Build + Eval)")
        console.print("\n[bold white]Advanced / Debug Commands:[/bold white]")
        console.print("  [cyan]mileage serve[/cyan]       Launch web dashboard command center with live SSE")
        console.print("  [cyan]mileage doctor[/cyan]      Run system health & local Ollama diagnostic checks")
        console.print("  [cyan]mileage init[/cyan]        Initialize a local .mileage workspace")
        console.print("  [cyan]mileage status[/cyan]      Inspect current workspace and Ollama daemon status")
        console.print("  [cyan]mileage models[/cyan]      List and verify locally installed Ollama models")
        console.print("  [cyan]mileage workspace[/cyan]   Scan and analyze local files & token footprints")
        console.print("  [cyan]mileage code[/cyan]        Autonomous local coding agent (inspect-plan-edit-test-repair)")
        console.print("  [cyan]mileage plan[/cyan]        Generate a structured ActionPlan via Gemma planner")
        console.print("  [cyan]mileage route[/cyan]       Auto-select the best model for a task category")
        console.print("  [cyan]mileage benchmark[/cyan]   Run standardized benchmarks on local models")
        console.print("  [cyan]mileage voice[/cyan]       Hands-free local voice assistant (Whisper + VAD + TTS)")
        console.print("  [cyan]mileage metrics[/cyan]     View local telemetry, token usage, and latencies")
        console.print("  [cyan]mileage evaluate[/cyan]    Evaluate implementation against ActionPlan")
        console.print("\n[dim]Run [bold]mileage <command> --help[/bold] for detailed command usage.[/dim]\n")


@app.command(name="start")
def start_cmd(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the default local Ollama model."
    ),
    voice: bool = typer.Option(
        True, "--voice/--no-voice", help="Enable hands-free voice input and speech synthesis."
    ),
    tts: bool = typer.Option(
        True, "--tts/--no-tts", help="Enable text-to-speech spoken summaries."
    ),
    max_iterations: int = typer.Option(
        10, "--max-iterations", "-i", help="Maximum execution loops before halting."
    ),
    whisper_model: str = typer.Option(
        "tiny.en", "--whisper-model", help="Whisper model size for voice STT."
    ),
):
    """Primary one-command experience: interactive autonomous agent session."""
    from mileage.interactive import InteractiveSession

    session = InteractiveSession(
        preferred_model=model,
        enable_voice=voice,
        enable_tts=tts,
        whisper_model=whisper_model,
        max_iterations=max_iterations,
    )
    session.run()


@app.command(name="doctor")
def doctor_cmd():
    """Run comprehensive diagnostics for Python, Ollama, local models, and workspace."""
    print_banner()
    console.print("[dim]Analyzing local environment, Ollama connectivity, models, and workspace...[/dim]\n")
    try:
        diagnostics = DoctorDiagnostics()
        report = diagnostics.run_all_checks()
        render_doctor_report(report)
        if not report.is_healthy:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        render_error("Doctor Diagnostic Failed", str(e))
        raise typer.Exit(code=1)


@app.command(name="init")
def init_cmd(
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Custom project name for this workspace."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite existing workspace configuration."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override default local Ollama model."
    ),
):
    """Initialize a local m.AI.leage workspace in the current directory."""
    print_banner()
    workspace = WorkspaceManager()

    # Automatically probe Ollama for installed models to recommend
    chosen_model = model
    if not chosen_model:
        ollama = OllamaClient()
        ollama.start_daemon(timeout_seconds=4.0)
        is_ok, installed = ollama.verify_installed_models()
        if is_ok and installed:
            chosen_model = installed[0].name
            console.print(f"[dim]Automatically detected local model: [bold green]{chosen_model}[/bold green][/dim]")

    try:
        config = workspace.initialize(
            project_name=name,
            force=force,
            preferred_model=chosen_model,
        )
        msg = (
            f"Initialized [bold cyan]{config.workspace.project_name}[/bold cyan] at [dim]{workspace.root_dir}[/dim]\n"
            f"Default Model: [bold green]{config.ollama.default_model}[/bold green]\n"
            f"Local Config: [dim]{workspace.config_path}[/dim]\n"
            f"Local Metrics: [dim]{workspace.metrics_path}[/dim]"
        )
        render_success("Workspace Ready", msg)
        console.print("[dim]Next steps: Run [bold]mileage doctor[/bold] or [bold]mileage workspace[/bold][/dim]\n")
    except MileageError as me:
        render_error("Initialization Failed", me.message, me.suggestion)
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        render_error("Initialization Error", str(e))
        raise typer.Exit(code=1)


@app.command(name="status")
def status_cmd():
    """Display quick status of the current workspace, active model, and Ollama."""
    print_banner()
    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    config = workspace.get_config()

    ollama = OllamaClient(host=config.ollama.host)
    # Auto-start if stopped
    ollama.start_daemon(timeout_seconds=3.0)
    health = ollama.check_health(quick_timeout=2.0)

    # Auto-resolve active model to installed model if needed
    active_model = ollama.resolve_active_model(config.ollama.default_model)

    console.print(f"[bold white]Project:[/bold white] [cyan]{config.workspace.project_name}[/cyan]")
    console.print(f"[bold white]Workspace Root:[/bold white] [dim]{workspace.root_dir}[/dim]")
    console.print(
        f"[bold white]Workspace Status:[/bold white] "
        f"{'[bold green]INITIALIZED[/bold green]' if workspace.is_initialized() else '[bold yellow]NOT INITIALIZED (run mileage init)[/bold yellow]'}"
    )

    ollama_status = (
        f"[bold green]ONLINE[/bold green] (v{health.version}, {health.response_time_ms}ms, auto-managed)"
        if health.is_running
        else f"[bold red]OFFLINE[/bold red] ({health.error_message})"
    )
    console.print(f"[bold white]Ollama Daemon:[/bold white] {ollama_status}")
    console.print(f"[bold white]Active Model:[/bold white] [bold green]{active_model}[/bold green]")

    # Check metrics
    tracker = MetricsTracker(workspace.metrics_path)
    summary = tracker.get_summary()
    console.print(f"[bold white]Local Runs:[/bold white] {summary.total_runs}  |  [bold white]Total Tokens:[/bold white] {summary.total_tokens:,}\n")


@app.command(name="models")
def models_cmd():
    """List and verify locally installed Ollama models."""
    print_banner()
    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    config = workspace.get_config()
    ollama = OllamaClient(host=config.ollama.host)

    try:
        with console.status("[bold cyan]Connecting to local Ollama daemon (auto-starting if needed)...[/bold cyan]"):
            ollama.start_daemon(timeout_seconds=5.0)
            models = ollama.list_models()
        active_model = ollama.resolve_active_model(config.ollama.default_model)
        render_models_table(models, active_model=active_model)
    except MileageError as me:
        render_error("Ollama Offline", me.message, me.suggestion)
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        render_error("Failed to list models", str(e), "Ensure Ollama is installed locally.")
        raise typer.Exit(code=1)


@app.command(name="workspace")
def workspace_cmd(
    scan: bool = typer.Option(
        False, "--scan", "-s", help="Perform a fresh file scan and list tracked files."
    ),
):
    """Scan and inspect local workspace files, size, and token estimates."""
    print_banner()
    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()

    try:
        with console.status("[bold cyan]Analyzing local workspace files...[/bold cyan]"):
            files = workspace.scan_files()
            stats = workspace.get_stats()

        render_workspace_table(stats, files if scan or len(files) <= 15 else files[:10])
    except typer.Exit:
        raise
    except Exception as e:
        render_error("Workspace Scan Error", str(e))
        raise typer.Exit(code=1)


@app.command(name="run")
def run_cmd(
    prompt: str = typer.Argument(..., help="The prompt or query for the local agent."),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the default local model."
    ),
    context: bool = typer.Option(
        True, "--context/--no-context", help="Inject local workspace context into prompt."
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Stream response tokens in real-time."
    ),
):
    """Execute a prompt locally using Ollama and track performance metrics."""
    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    config = workspace.get_config()

    ollama = OllamaClient(host=config.ollama.host, timeout_seconds=config.ollama.timeout_seconds)

    with console.status("[bold cyan]Auto-starting Ollama and readying local model...[/bold cyan]"):
        is_online, target_model = ollama.ensure_ready(preferred_model=model or config.ollama.default_model)

    tracker = MetricsTracker(workspace.metrics_path if workspace.is_initialized() else None)

    agent = LocalAgent(
        ollama_client=ollama,
        model_name=target_model,
        workspace=workspace if context else None,
        metrics_tracker=tracker,
        temperature=config.ollama.temperature,
    )

    if context:
        agent.inject_workspace_context()

    console.print(f"[dim]Executing query with local model: [bold green]{target_model}[/bold green][/dim]\n")

    try:
        if stream:
            console.print("[bold cyan]m.AI.leage >[/bold cyan] ", end="")
            for chunk in agent.stream_run(prompt):
                console.print(chunk, end="")
            console.print()
        else:
            with console.status("[bold cyan]Generating local response...[/bold cyan]"):
                response = agent.run(prompt)
            console.print(Markdown(response))

    except MileageError as me:
        render_error("Local Execution Failed", me.message, me.suggestion)
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        render_error("Execution Error", str(e), "Verify that Ollama is installed.")
        raise typer.Exit(code=1)


@app.command(name="plan")
def plan_cmd(
    prompt: Optional[str] = typer.Argument(
        None, help="Text prompt or task description for the planner."
    ),
    image: Optional[str] = typer.Option(
        None, "--image", "-i", help="Path to an image file for multimodal analysis."
    ),
    files: Optional[str] = typer.Option(
        None, "--files", "-f",
        help="Comma-separated list of workspace-relative file paths to include as context."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the default Ollama model for planning."
    ),
    max_tokens: int = typer.Option(
        4096, "--max-tokens", help="Maximum token budget for file context injection."
    ),
    raw: bool = typer.Option(
        False, "--raw", "-r", help="Also display the raw JSON plan output."
    ),
):
    """Generate a structured ActionPlan from text, images, or project files using Gemma."""
    print_banner()

    if not prompt and not image and not files:
        console.print("[yellow]Provide a prompt, --image, or --files to generate a plan.[/yellow]")
        console.print("[dim]Example: mileage plan 'Add user authentication to the API'[/dim]")
        console.print("[dim]Example: mileage plan --image screenshot.png 'Implement this UI'[/dim]")
        console.print("[dim]Example: mileage plan --files src/app.py,src/models.py 'Refactor models'[/dim]")
        raise typer.Exit(code=1)

    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    config = workspace.get_config()

    ollama = OllamaClient(host=config.ollama.host, timeout_seconds=config.ollama.timeout_seconds)

    with console.status("[bold cyan]Auto-starting Ollama and readying local model...[/bold cyan]"):
        is_online, target_model = ollama.ensure_ready(preferred_model=model or config.ollama.default_model)

    if not is_online:
        render_error(
            "Ollama Offline",
            "Could not start or connect to local Ollama daemon.",
            "Ensure Ollama is installed. Run 'ollama serve' manually.",
        )
        raise typer.Exit(code=1)

    # Parse file paths
    file_paths: list[str] = []
    if files:
        file_paths = [f.strip() for f in files.split(",") if f.strip()]

    # Parse image paths
    image_paths: list[str] = []
    if image:
        img_path = Path(image)
        if not img_path.is_absolute():
            img_path = (workspace.root_dir / img_path).resolve()
        if not img_path.is_file():
            render_error("Image Not Found", f"Could not locate image: {image}")
            raise typer.Exit(code=1)
        image_paths = [str(img_path)]

    planner = PlannerAgent(
        ollama_client=ollama,
        model_name=target_model,
        workspace=workspace,
        max_context_tokens=max_tokens,
    )

    planner_input = PlannerInput(
        text=prompt,
        image_paths=image_paths,
        file_paths=file_paths,
        max_context_tokens=max_tokens,
    )

    console.print(f"[dim]Planning with local model: [bold green]{target_model}[/bold green][/dim]")
    if file_paths:
        console.print(f"[dim]Including {len(file_paths)} file(s) as context[/dim]")
    if image_paths:
        console.print(f"[dim]Analyzing image: {Path(image_paths[0]).name}[/dim]")
    console.print()

    try:
        with console.status("[bold cyan]🧠 Generating structured action plan...[/bold cyan]"):
            plan = planner.plan(planner_input)

        render_action_plan(plan, show_raw=raw)

        # Save plan to .mileage/plans/ for reference
        if workspace.is_initialized():
            plans_dir = workspace.mileage_dir / "plans"
            plans_dir.mkdir(exist_ok=True)
            plan_file = plans_dir / f"{plan.plan_id}.json"
            plan_file.write_text(
                plan.model_dump_json(indent=2), encoding="utf-8"
            )
            console.print(f"\n[dim]Plan saved to: {plan_file}[/dim]")

    except MileageError as me:
        render_error("Planning Failed", me.message, me.suggestion)
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        render_error("Planner Error", str(e), "Check that Ollama is running and the model supports structured output.")
        raise typer.Exit(code=1)


@app.command(name="metrics")
def metrics_cmd():
    """Display local execution metrics, token counts, and latencies."""
    print_banner()
    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    tracker = MetricsTracker(workspace.metrics_path)
    summary = tracker.get_summary()

    if summary.total_runs == 0:
        console.print("[yellow]No local metrics recorded yet.[/yellow]")
        console.print("[dim]Metrics are saved automatically to .mileage/metrics.jsonl when running local commands.[/dim]\n")
        return

    render_metrics_table(summary)


@app.command(name="route")
def route_cmd(
    task: str = typer.Argument(
        "general",
        help="Task category: code_generation, debugging, code_review, planning, reasoning, chat, summarization, image_analysis, general.",
    ),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", help="Infer task category automatically from a prompt."
    ),
    raw: bool = typer.Option(
        False, "--raw", "-r", help="Show raw JSON routing decision."
    ),
):
    """Auto-select the best local model for a task using deterministic scoring."""
    print_banner()

    from mileage.router import ModelRouter, TaskCategory as TC
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    config = workspace.get_config()
    ollama = OllamaClient(host=config.ollama.host, timeout_seconds=config.ollama.timeout_seconds)

    router = ModelRouter(ollama_client=ollama, workspace_root=workspace.root_dir)

    with console.status("[bold cyan]Discovering local models and computing capabilities...[/bold cyan]"):
        profiles = router.discover()

    if not profiles:
        render_error("No Models Found", "No Ollama models discovered.", "Pull a model: ollama pull gemma3:4b")
        raise typer.Exit(code=1)

    # Determine task category
    if prompt:
        decision = router.route_for_prompt(prompt)
    else:
        try:
            category = TC(task.lower().strip())
        except ValueError:
            console.print(f"[yellow]Unknown task category '{task}'. Using 'general'.[/yellow]")
            category = TC.GENERAL
        decision = router.route(category)

    # ── Render decision ──
    conf_color = "green" if decision.confidence >= 0.6 else "yellow" if decision.confidence >= 0.4 else "red"

    header = Text()
    header.append("🎯 Selected: ", style="bold white")
    header.append(f"{decision.selected_model}\n", style="bold green")
    if decision.fallback_model:
        header.append("🔄 Fallback: ", style="bold white")
        header.append(f"{decision.fallback_model}\n", style="dim")
    header.append("\n📋 Reason: ", style="bold white")
    header.append(f"{decision.reason}\n\n", style="white")
    header.append("⚡ Confidence: ", style="bold white")
    header.append(f"{decision.confidence:.2f}", style=f"bold {conf_color}")
    header.append("    📂 Task: ", style="bold white")
    header.append(f"{decision.task_category.value}", style="cyan")
    if decision.expected_latency_ms:
        header.append(f"    ⏱️ Expected: ", style="bold white")
        header.append(f"{decision.expected_latency_ms:.0f}ms", style="yellow")
    if decision.expected_tokens_per_sec:
        header.append(f" ({decision.expected_tokens_per_sec:.1f} tok/s)", style="dim")

    console.print(Panel(
        header,
        title="[bold magenta]🧭 Model Router Decision[/bold magenta]",
        border_style="magenta",
        padding=(1, 2),
    ))

    # ── Scores table ──
    if decision.scores:
        scores_table = Table(
            title="[bold cyan]📊 Model Scores[/bold cyan]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        scores_table.add_column("Model", style="white")
        scores_table.add_column("Score", justify="right", style="green")
        scores_table.add_column("Selected", justify="center")

        for model_name, score in sorted(decision.scores.items(), key=lambda x: -x[1]):
            is_selected = model_name == decision.selected_model
            badge = "[bold green]✓ SELECTED[/bold green]" if is_selected else "[dim]—[/dim]"
            style = "bold green" if is_selected else "white"
            scores_table.add_row(
                Text(model_name, style=style),
                f"{score:.4f}",
                badge,
            )
        console.print(scores_table)

    # ── Capabilities table ──
    if profiles:
        cap_table = Table(
            title="[bold cyan]🧠 Model Capabilities[/bold cyan]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        cap_table.add_column("Model", style="white")
        cap_table.add_column("Size", justify="right", style="green")
        cap_table.add_column("Context", justify="right", style="yellow")
        cap_table.add_column("Code", justify="center")
        cap_table.add_column("Reason", justify="center")
        cap_table.add_column("Vision", justify="center")
        cap_table.add_column("Speed", justify="center")
        cap_table.add_column("Plan", justify="center")

        from mileage.router.schemas import Capability as Cap

        def _cap_badge(score: float) -> str:
            if score >= 0.8:
                return f"[bold green]{score:.2f}[/bold green]"
            elif score >= 0.5:
                return f"[yellow]{score:.2f}[/yellow]"
            elif score >= 0.3:
                return f"[dim]{score:.2f}[/dim]"
            else:
                return f"[red dim]—[/red dim]"

        for p in profiles:
            cap_table.add_row(
                p.name,
                p.size_human,
                f"{p.context_length:,}",
                _cap_badge(p.get_capability_score(Cap.CODING)),
                _cap_badge(p.get_capability_score(Cap.REASONING)),
                _cap_badge(p.get_capability_score(Cap.VISION)),
                _cap_badge(p.get_capability_score(Cap.SPEED)),
                _cap_badge(p.get_capability_score(Cap.PLANNING)),
            )
        console.print(cap_table)

    console.print(f"\n[dim]Decision time: {decision.decision_time_ms:.1f}ms | {decision.decided_at}[/dim]")
    console.print("[dim]Configure routing: .mileage/router_config.json[/dim]\n")

    if raw:
        import json as json_mod
        console.print(Panel(
            Text(json_mod.dumps(decision.model_dump(mode="json"), indent=2), style="dim"),
            title="[dim]Raw JSON[/dim]",
            border_style="dim",
        ))


@app.command(name="benchmark")
def benchmark_cmd(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Benchmark a specific model (default: all installed)."
    ),
):
    """Run standardized benchmark tasks against local models."""
    print_banner()

    from mileage.router import ModelRouter
    from rich.table import Table
    from rich.text import Text

    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    config = workspace.get_config()
    ollama = OllamaClient(host=config.ollama.host, timeout_seconds=config.ollama.timeout_seconds)

    router = ModelRouter(ollama_client=ollama, workspace_root=workspace.root_dir)

    with console.status("[bold cyan]Discovering local models...[/bold cyan]"):
        profiles = router.discover()

    if not profiles:
        render_error("No Models Found", "No Ollama models discovered.")
        raise typer.Exit(code=1)

    models_to_bench = [model] if model else None
    target_names = [model] if model else [p.name for p in profiles]

    console.print(f"[bold white]Benchmarking {len(target_names)} model(s):[/bold white] {', '.join(target_names)}")
    console.print("[dim]Running 7 standardized tasks (coding, reasoning, chat, summarization, planning)...[/dim]\n")

    try:
        all_results = router.benchmark_all(model_names=models_to_bench)
    except Exception as e:
        render_error("Benchmark Failed", str(e))
        raise typer.Exit(code=1)

    # Render results
    for model_name, results in all_results.items():
        table = Table(
            title=f"[bold magenta]📊 Benchmark: {model_name}[/bold magenta]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Task", style="white")
        table.add_column("Status", justify="center", width=8)
        table.add_column("Latency", justify="right", style="yellow")
        table.add_column("Tokens", justify="right")
        table.add_column("tok/s", justify="right", style="green")
        table.add_column("Preview", style="dim", max_width=40)

        passes = 0
        total_ms = 0.0
        for r in results:
            status = "[bold green]PASS[/bold green]" if r.success else "[bold red]FAIL[/bold red]"
            preview = r.response_preview[:40].replace("\n", " ") if r.response_preview else "—"
            table.add_row(
                r.task_id.replace("bench_", ""),
                status,
                f"{r.latency_ms:.0f}ms",
                str(r.tokens_generated),
                f"{r.tokens_per_sec:.1f}",
                preview,
            )
            if r.success:
                passes += 1
            total_ms += r.latency_ms

        console.print(table)
        console.print(
            f"[dim]  Results: {passes}/{len(results)} passed | "
            f"Total: {total_ms:.0f}ms | "
            f"Avg: {total_ms / len(results):.0f}ms/task[/dim]\n"
        )

    console.print("[dim]Benchmark data saved to .mileage/router.db for routing decisions.[/dim]\n")


@app.command(name="voice")
def voice_cmd(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the local Ollama model."
    ),
    whisper_model: str = typer.Option(
        "tiny.en", "--whisper-model", "-w", help="Local Whisper model size (tiny.en, base.en, small.en)."
    ),
    tts: bool = typer.Option(
        True, "--tts/--no-tts", help="Enable or disable local speech output."
    ),
):
    """Launch hands-free local voice interaction with Whisper STT and TTS."""
    print_banner()
    console.print("[dim]Initializing hands-free voice loop with local Whisper and TTS...[/dim]\n")

    from mileage.voice import SpeechToTextEngine, TextToSpeechEngine, VoiceSession

    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    config = workspace.get_config()

    ollama = OllamaClient(host=config.ollama.host, timeout_seconds=config.ollama.timeout_seconds)

    with console.status("[bold cyan]Auto-starting Ollama and readying local model...[/bold cyan]"):
        is_online, target_model = ollama.ensure_ready(preferred_model=model or config.ollama.default_model)

    tracker = MetricsTracker(workspace.metrics_path if workspace.is_initialized() else None)

    # Initialize local agent
    agent = LocalAgent(
        ollama_client=ollama,
        model_name=target_model,
        workspace=workspace,
        metrics_tracker=tracker,
        temperature=0.6,
    )
    agent.inject_workspace_context()

    try:
        stt_engine = SpeechToTextEngine(model_size=whisper_model)
        tts_engine = TextToSpeechEngine()

        session = VoiceSession(
            agent=agent,
            stt_engine=stt_engine,
            tts_engine=tts_engine,
            enable_tts=tts,
        )

        console.print(f"[bold cyan]Local Model:[/bold cyan] [green]{target_model}[/green]")
        console.print(f"[bold cyan]Whisper STT:[/bold cyan] [green]{whisper_model}[/green]")
        console.print(f"[bold cyan]Speech Output (TTS):[/bold cyan] {'[green]ON[/green]' if tts else '[yellow]MUTED[/yellow]'}")
        console.print("[dim]Press Ctrl+C or say 'stop' at any time to exit.[/dim]\n")

        session.run_live()

    except MileageError as me:
        render_error("Voice Session Failed", me.message, me.suggestion)
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Voice session closed by user.[/yellow]")
    except Exception as e:
        render_error("Voice Runtime Error", str(e), "Verify microphone permissions and sound device.")
        raise typer.Exit(code=1)


@app.command(name="code")
def code_cmd(
    goal: str = typer.Argument(..., help="The coding objective or task for the agent to complete."),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the default local Ollama model."
    ),
    max_iterations: int = typer.Option(
        10, "--max-iterations", "-i", help="Maximum execution loops before halting."
    ),
    test_cmd: Optional[str] = typer.Option(
        None, "--test-cmd", "-t", help="Custom project test command (e.g. 'pytest tests/')."
    ),
    auto_test: bool = typer.Option(
        True, "--auto-test/--no-auto-test", help="Automatically run project-native tests during execution."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Simulate execution without modifying files."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Display verbose action details in terminal."
    ),
):
    """Autonomous local coding agent: inspect → plan → edit → test → repair."""
    print_banner()

    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    config = workspace.get_config()

    ollama = OllamaClient(host=config.ollama.host, timeout_seconds=config.ollama.timeout_seconds)

    with console.status("[bold cyan]Auto-starting Ollama and readying local model...[/bold cyan]"):
        is_online, target_model = ollama.ensure_ready(preferred_model=model or config.ollama.default_model)

    if not is_online:
        render_error(
            "Ollama Offline",
            "Could not start or connect to local Ollama daemon.",
            "Ensure Ollama is installed. Run 'ollama serve' manually.",
        )
        raise typer.Exit(code=1)

    tracker = MetricsTracker(workspace.metrics_path if workspace.is_initialized() else None)

    console.print(f"[bold cyan]🎯 Goal:[/bold cyan] {goal}")
    console.print(f"[bold cyan]🤖 Model:[/bold cyan] [bold green]{target_model}[/bold green] | [dim]Max Retries: {max_iterations} | Auto-test: {auto_test}[/dim]\n")

    # Optional dashboard event bus integration
    try:
        from mileage.dashboard import DashboardEventBus, PipelineStage, BuildRecord
        bus = DashboardEventBus.get_instance()
        bus.publish_pipeline_update(
            stage=PipelineStage.BUILDING,
            goal=goal,
            model_name=target_model,
            iteration=1,
            max_iterations=max_iterations,
            message="Starting autonomous coding loop...",
        )
    except Exception:
        bus = None

    # Concise real-time terminal event callback
    def on_event(ev: AgentEvent) -> None:
        if bus:
            try:
                bus.publish_pipeline_update(
                    stage=PipelineStage.BUILDING,
                    goal=goal,
                    model_name=target_model,
                    iteration=ev.iteration,
                    max_iterations=max_iterations,
                    message=f"[{ev.phase.value}] {ev.message}",
                )
            except Exception:
                pass

        if ev.event_type == "step_start":
            console.print(f"[bold cyan]▶ Iteration {ev.iteration}/{max_iterations}[/bold cyan] [dim]({ev.phase.value})[/dim]")
        elif ev.event_type == "inspect":
            if verbose:
                console.print(f"  [dim]🔍 {ev.message}[/dim]")
        elif ev.event_type == "plan":
            console.print(f"  [cyan]📝 {ev.message}[/cyan]")
        elif ev.event_type == "edit":
            console.print(f"  [bold yellow]✏️ {ev.message}[/bold yellow]")
        elif ev.event_type == "test":
            console.print(f"  [dim]🧪 {ev.message}[/dim]")
        elif ev.event_type == "test_result":
            passed = ev.data.get("passed", False)
            badge = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
            console.print(f"  🧪 Tests: {badge} [dim]({ev.message})[/dim]")
        elif ev.event_type == "repair":
            console.print(f"  [bold red]🔧 {ev.message}[/bold red]")
        elif ev.event_type == "stagnation":
            console.print(f"  [bold yellow]⚠️ Stagnation: {ev.message}[/bold yellow]")
        elif ev.event_type == "complete":
            console.print(f"  [bold green]✓ {ev.message}[/bold green]")

    agent = CodingAgent(
        ollama_client=ollama,
        model_name=target_model,
        workspace=workspace,
        metrics_tracker=tracker,
        max_iterations=max_iterations,
        auto_test=auto_test,
        custom_test_cmd=test_cmd,
        dry_run=dry_run,
        on_event=on_event,
    )

    try:
        session = agent.execute_task(goal)
        console.print()
        render_coding_session(session)

        if bus:
            try:
                bus.record_build(
                    BuildRecord(
                        build_id=session.session_id,
                        goal=goal,
                        model_name=target_model,
                        status="complete" if session.status == AgentStatus.COMPLETED else "failed",
                        started_at=session.started_at,
                        completed_at=session.completed_at or "",
                        duration_ms=session.duration_ms,
                        tokens_used=session.total_tokens,
                        requirements_passed=1 if session.status == AgentStatus.COMPLETED else 0,
                        requirements_total=1,
                        escalated=False,
                    )
                )
                bus.publish_pipeline_update(
                    stage=PipelineStage.COMPLETE if session.status == AgentStatus.COMPLETED else PipelineStage.FAILED,
                    goal=goal,
                    model_name=target_model,
                    message=f"Build finished: {session.status.value}",
                )
            except Exception:
                pass

        if session.status != AgentStatus.COMPLETED:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        render_error("Coding Agent Error", str(e), "Check that local Ollama is active and model is loaded.")
        raise typer.Exit(code=1)


@app.command(name="agent")
def agent_cmd(
    goal: str = typer.Argument(..., help="The coding objective or task for the agent to complete."),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the default local Ollama model."
    ),
    max_iterations: int = typer.Option(
        10, "--max-iterations", "-i", help="Maximum execution loops before halting."
    ),
    test_cmd: Optional[str] = typer.Option(
        None, "--test-cmd", "-t", help="Custom project test command (e.g. 'pytest tests/')."
    ),
    auto_test: bool = typer.Option(
        True, "--auto-test/--no-auto-test", help="Automatically run project-native tests during execution."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Simulate execution without modifying files."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Display verbose action details in terminal."
    ),
):
    """Alias for 'mileage code'."""
    code_cmd(
        goal=goal,
        model=model,
        max_iterations=max_iterations,
        test_cmd=test_cmd,
        auto_test=auto_test,
        dry_run=dry_run,
        verbose=verbose,
    )


@app.command(name="evaluate")
def evaluate_cmd(
    plan_file: str = typer.Argument(..., help="Path to ActionPlan JSON file to evaluate against."),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the default local Ollama model."
    ),
    test_cmd: Optional[str] = typer.Option(
        None, "--test-cmd", "-t", help="Custom project test command."
    ),
    auto_escalate: bool = typer.Option(
        True, "--auto-escalate/--no-escalate",
        help="Automatically escalate to Copilot on overcapacity detection."
    ),
    open_vscode: bool = typer.Option(
        True, "--open-vscode/--no-vscode",
        help="Open VS Code automatically during escalation."
    ),
):
    """Evaluate implementation against an ActionPlan and optionally escalate to Copilot."""
    import json as _json
    from mileage.agents.evaluator import EvaluatorAgent
    from mileage.agents.evaluator_schemas import RequirementVerdict, EscalationReason
    from mileage.agents.escalation import CopilotEscalation
    from mileage.agents.planner_schemas import ActionPlan
    from mileage.ui.components import render_evaluation_report, render_escalation

    print_banner()

    # Load ActionPlan
    plan_path = Path(plan_file)
    if not plan_path.is_file():
        render_error("Plan Not Found", f"Cannot find plan file: {plan_file}")
        raise typer.Exit(code=1)

    try:
        plan_data = _json.loads(plan_path.read_text(encoding="utf-8"))
        plan = ActionPlan.model_validate(plan_data)
    except Exception as e:
        render_error("Invalid Plan", str(e), "Ensure the plan file is valid ActionPlan JSON.")
        raise typer.Exit(code=1)

    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    config = workspace.get_config()

    ollama = OllamaClient(host=config.ollama.host, timeout_seconds=config.ollama.timeout_seconds)

    with console.status("[bold cyan]Readying Ollama for evaluation...[/bold cyan]"):
        is_online, target_model = ollama.ensure_ready(preferred_model=model or config.ollama.default_model)

    if not is_online:
        render_error("Ollama Offline", "Could not connect to Ollama.", "Run 'ollama serve'.")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]📋 Evaluating:[/bold cyan] {plan.goal}")
    console.print(f"[bold cyan]🤖 Model:[/bold cyan] [bold green]{target_model}[/bold green]\n")

    try:
        from mileage.dashboard import DashboardEventBus, PipelineStage, BuildRecord
        bus = DashboardEventBus.get_instance()
        bus.publish_pipeline_update(
            stage=PipelineStage.EVALUATING,
            goal=plan.goal,
            model_name=target_model,
            message="Evaluating implementation against acceptance criteria...",
            requirements_passed=0,
            requirements_total=len(plan.requirements),
        )
    except Exception:
        bus = None

    evaluator = EvaluatorAgent(
        ollama_client=ollama,
        model_name=target_model,
        workspace=workspace,
    )

    try:
        report = evaluator.evaluate(plan, custom_test_cmd=test_cmd)
        render_evaluation_report(report)

        # Auto-escalate on overcapacity or FAIL
        if auto_escalate and (
            report.overcapacity_detected or report.overall_verdict == RequirementVerdict.FAIL
        ):
            console.print("[bold yellow]⚡ Escalating to GitHub Copilot...[/bold yellow]\n")

            reason = (
                EscalationReason.OVERCAPACITY
                if report.overcapacity_detected
                else EscalationReason.REPEATED_FAILURES
            )

            esc_engine = CopilotEscalation(workspace_root=workspace.root_dir)
            esc_record = esc_engine.escalate(
                plan=plan,
                report=report,
                reason=reason,
                open_vscode=open_vscode,
            )
            render_escalation(esc_record)

        if bus:
            try:
                is_escalated = auto_escalate and (
                    report.overcapacity_detected or report.overall_verdict == RequirementVerdict.FAIL
                )
                passed_cnt = sum(1 for r in report.requirements if r.verdict == RequirementVerdict.PASS)
                total_cnt = len(report.requirements)
                bus.record_build(
                    BuildRecord(
                        build_id=report.report_id,
                        goal=plan.goal,
                        model_name=target_model,
                        status="escalated" if is_escalated else ("complete" if report.overall_verdict == RequirementVerdict.PASS else "failed"),
                        started_at=report.evaluated_at,
                        completed_at=report.evaluated_at,
                        duration_ms=report.duration_ms,
                        tokens_used=report.tokens_used,
                        requirements_passed=passed_cnt,
                        requirements_total=total_cnt,
                        escalated=is_escalated,
                    )
                )
                bus.publish_pipeline_update(
                    stage=PipelineStage.ESCALATING if is_escalated else (PipelineStage.COMPLETE if report.overall_verdict == RequirementVerdict.PASS else PipelineStage.FAILED),
                    goal=plan.goal,
                    model_name=target_model,
                    requirements_passed=passed_cnt,
                    requirements_total=total_cnt,
                    message=f"Evaluation complete: {report.overall_verdict.value} ({passed_cnt}/{total_cnt} criteria passed)",
                )
            except Exception:
                pass

        if report.overall_verdict != RequirementVerdict.PASS:
            raise typer.Exit(code=1)

        render_success("All requirements PASS with evidence!", "Evaluation complete.")

    except typer.Exit:
        raise
    except Exception as e:
        render_error("Evaluator Error", str(e))
        raise typer.Exit(code=1)


@app.command(name="serve")
def serve_cmd(
    port: int = typer.Option(
        3000, "--port", "-p", help="Port to bind dashboard server (default: 3000)."
    ),
    host: str = typer.Option(
        "0.0.0.0", "--host", "-h", help="Host interface to bind to (default: 0.0.0.0 for LAN access)."
    ),
    open_browser: bool = typer.Option(
        False, "--open", help="Open dashboard in browser automatically."
    ),
):
    """Launch the m.AI.leage Command Center web dashboard with real-time SSE streaming."""
    import webbrowser
    import threading
    import time
    from mileage.dashboard.server import DashboardServer

    print_banner()
    workspace = WorkspaceManager.find_workspace() or WorkspaceManager()
    tracker = MetricsTracker(workspace.metrics_path if workspace.is_initialized() else None)

    static_dir = Path(__file__).parent / "dashboard" / "static"
    if not (static_dir / "index.html").exists():
        console.print("[yellow]⚠ Dashboard static assets not found at:[/yellow] [dim]" + str(static_dir) + "[/dim]")
        console.print("[dim]Serving API and SSE stream. Build dashboard via: cd dashboard && npm run build[/dim]\n")

    server = DashboardServer(
        port=port,
        host=host,
        static_dir=static_dir,
        metrics_tracker=tracker,
    )

    if open_browser:
        def _open():
            time.sleep(1.0)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.start(blocking=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard server stopped by user.[/yellow]")
        server.stop()


if __name__ == "__main__":
    app()
