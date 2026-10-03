"""Reusable Rich UI components for tables, status cards, and output."""

from typing import List, Optional
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from mileage.core.workspace import FileMetadata, WorkspaceStats
from mileage.metrics.schemas import MetricsSummary
from mileage.models.schemas import ModelInfo
from mileage.ui.console import console, err_console


def render_error(title: str, message: str, suggestion: Optional[str] = None) -> None:
    """Render a graceful, styled error panel."""
    content = Text()
    content.append(f"{message}\n", style="white")
    if suggestion:
        content.append(f"\n[!] Suggestion: {suggestion}", style="bold yellow")

    panel = Panel(
        content,
        title=f"[bold red][ERROR] {title}[/bold red]",
        border_style="red",
        padding=(1, 2),
    )
    err_console.print(panel)


def render_success(title: str, message: str) -> None:
    """Render a styled success notification."""
    panel = Panel(
        Text(message, style="white"),
        title=f"[bold green][OK] {title}[/bold green]",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)


def render_models_table(models: List[ModelInfo], active_model: Optional[str] = None) -> None:
    """Render a formatted table of local Ollama models."""
    if not models:
        console.print("[yellow]No local models found in Ollama.[/yellow]")
        console.print("[dim]Pull a model using: [bold]ollama pull llama3.2[/bold][/dim]")
        return

    table = Table(
        title="[bold magenta]Installed Local Models (Ollama)[/bold magenta]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Status", justify="center", width=8)
    table.add_column("Model Name", style="bold white")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Parameters", justify="center", style="yellow")
    table.add_column("Family", style="cyan")
    table.add_column("Format", style="dim")

    for m in models:
        is_active = (
            active_model
            and (m.name.lower() == active_model.lower() or m.name.split(":")[0] == active_model.split(":")[0])
        )
        status_badge = "[bold green]ACTIVE[/bold green]" if is_active else "[dim]READY[/dim]"
        param_size = m.details.parameter_size if m.details and m.details.parameter_size else "n/a"
        family = m.details.family if m.details and m.details.family else "unknown"
        fmt = m.details.format if m.details and m.details.format else "gguf"

        table.add_row(
            status_badge,
            m.name,
            m.size_human,
            param_size,
            family,
            fmt,
        )

    console.print(table)


def render_workspace_table(stats: WorkspaceStats, files: Optional[List[FileMetadata]] = None) -> None:
    """Render workspace summary statistics and top files."""
    table = Table(
        title=f"[bold cyan]Workspace: {stats.project_name}[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("Root Directory", stats.workspace_root)
    table.add_row("Total Files Tracked", f"{stats.total_files:,}")
    table.add_row("Total Workspace Size", f"{stats.total_bytes:,} bytes")
    table.add_row("Estimated Context Tokens", f"~{stats.total_estimated_tokens:,} tokens")

    # Extension breakdown
    ext_summary = ", ".join(f"{k}: {v}" for k, v in sorted(stats.extension_counts.items(), key=lambda x: -x[1])[:8])
    table.add_row("File Types", ext_summary or "None")

    console.print(table)

    if files:
        file_table = Table(
            title="[bold dim]Recent / Tracked Files[/bold dim]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        file_table.add_column("Relative Path", style="white")
        file_table.add_column("Size", justify="right", style="green")
        file_table.add_column("Est. Tokens", justify="right", style="yellow")
        file_table.add_column("Type", justify="center", style="dim")

        for f in files[:15]:
            file_table.add_row(
                f.relative_path,
                f"{f.size_bytes:,} B",
                f"~{f.estimated_tokens:,}",
                f.extension,
            )

        if len(files) > 15:
            file_table.add_row(f"... and {len(files) - 15} more files", "", "", "")

        console.print(file_table)


def render_metrics_table(summary: MetricsSummary) -> None:
    """Render local execution metrics table."""
    table = Table(
        title="[bold magenta]m.AI.leage Local Execution Telemetry[/bold magenta]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Telemetry Metric", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("Total Runs Executed", f"{summary.total_runs:,}")
    table.add_row("Successful Runs", f"{summary.successful_runs:,}")
    table.add_row("Failed Runs", f"{summary.failed_runs:,}")
    table.add_row("Success Rate", f"{summary.success_rate_percent}%")
    table.add_row("Total Tokens Processed", f"{summary.total_tokens:,}")
    table.add_row("  -> Prompt Tokens", f"{summary.total_prompt_tokens:,}")
    table.add_row("  -> Completion Tokens", f"{summary.total_completion_tokens:,}")
    table.add_row("Average Latency", f"{summary.average_latency_ms:.1f} ms")

    if summary.model_usage_counts:
        model_dist = ", ".join(f"{k}: {v}" for k, v in summary.model_usage_counts.items())
        table.add_row("Model Breakdown", model_dist)

    console.print(table)


def render_action_plan(plan: "ActionPlan", show_raw: bool = False) -> None:
    """Render a structured ActionPlan with rich panels, tables, and trees.

    Import is quoted to avoid circular imports — the function accepts any
    ActionPlan-like object with the expected attributes.
    """
    # ── Complexity badge ──
    complexity_colors = {
        "trivial": "green",
        "low": "cyan",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red",
    }
    comp_color = complexity_colors.get(plan.complexity.value, "white")

    # ── Header panel ──
    header = Text()
    header.append("🎯 Goal: ", style="bold white")
    header.append(f"{plan.goal}\n\n", style="white")
    if plan.context:
        header.append("📋 Context: ", style="bold dim")
        header.append(f"{plan.context}\n\n", style="dim")
    header.append("⚡ Complexity: ", style="bold white")
    header.append(f"[{plan.complexity.value.upper()}]", style=f"bold {comp_color}")
    header.append(f"    📥 Input: ", style="bold white")
    header.append(f"{plan.input_type.value}", style="cyan")
    header.append(f"    🧠 Model: ", style="bold white")
    header.append(f"{plan.model_used}", style="green")

    console.print(Panel(
        header,
        title=f"[bold magenta]📐 Action Plan: {plan.plan_id}[/bold magenta]",
        border_style="magenta",
        padding=(1, 2),
    ))

    # ── Requirements table ──
    if plan.requirements:
        req_table = Table(
            title="[bold cyan]📋 Requirements[/bold cyan]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        req_table.add_column("ID", style="bold yellow", width=6)
        req_table.add_column("Description", style="white")
        req_table.add_column("Priority", justify="center", width=10)

        priority_styles = {"must": "bold red", "should": "yellow", "could": "dim green"}
        for req in plan.requirements:
            pstyle = priority_styles.get(req.priority, "white")
            req_table.add_row(
                req.id,
                req.description,
                Text(req.priority.upper(), style=pstyle),
            )
        console.print(req_table)

    # ── Files tree ──
    if plan.files:
        tree = Tree("[bold green]📁 File References[/bold green]")
        role_icons = {"target": "🎯", "context": "📖", "output": "📝"}
        for f in plan.files:
            icon = role_icons.get(f.role, "📄")
            label = f"{icon} {f.path} [dim]({f.role})[/dim]"
            node = tree.add(label)
            if f.summary:
                node.add(f"[dim]{f.summary}[/dim]")
        console.print(Panel(tree, border_style="green", padding=(0, 1)))

    # ── Constraints ──
    if plan.constraints:
        constraint_text = Text()
        for i, c in enumerate(plan.constraints, 1):
            constraint_text.append(f"  ⛔ {i}. ", style="bold red")
            constraint_text.append(f"{c.description}", style="white")
            if c.reason:
                constraint_text.append(f" — {c.reason}", style="dim")
            constraint_text.append("\n")
        console.print(Panel(
            constraint_text,
            title="[bold red]🚧 Constraints[/bold red]",
            border_style="red",
            padding=(0, 1),
        ))

    # ── Acceptance Criteria ──
    if plan.acceptance_criteria:
        ac_table = Table(
            title="[bold green]✅ Acceptance Criteria[/bold green]",
            show_header=True,
            header_style="bold green",
            border_style="dim",
        )
        ac_table.add_column("ID", style="bold yellow", width=6)
        ac_table.add_column("Criterion", style="white")
        ac_table.add_column("Verification", style="dim")

        for ac in plan.acceptance_criteria:
            ac_table.add_row(ac.id, ac.description, ac.verification or "—")
        console.print(ac_table)

    # ── Image analysis ──
    if plan.image_analysis:
        console.print(Panel(
            Text(plan.image_analysis, style="white"),
            title="[bold blue]🖼️ Image Analysis[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        ))

    # ── Metadata footer ──
    meta = (
        f"[dim]Tokens: {plan.tokens_used:,}  |  "
        f"Latency: {plan.latency_ms:.0f}ms  |  "
        f"Created: {plan.created_at}[/dim]"
    )
    console.print(meta)

    # ── Raw JSON (optional) ──
    if show_raw:
        import json as json_mod
        console.print()
        console.print(Panel(
            Text(json_mod.dumps(plan.model_dump(mode="json"), indent=2), style="dim"),
            title="[dim]Raw JSON Output[/dim]",
            border_style="dim",
        ))


def render_coding_session(session) -> None:
    """Render a clean, concise summary card for an autonomous coding agent session."""
    from mileage.agents.coding_schemas import AgentStatus

    # Status color & icon
    status_map = {
        AgentStatus.COMPLETED: ("[bold green]COMPLETED[/bold green]", "green", "✓"),
        AgentStatus.STAGNATED: ("[bold yellow]STAGNATED (HALTED)[/bold yellow]", "yellow", "⚠"),
        AgentStatus.FAILED: ("[bold red]FAILED[/bold red]", "red", "✗"),
        AgentStatus.RUNNING: ("[bold cyan]RUNNING[/bold cyan]", "cyan", "▶"),
    }
    status_str, border_color, icon = status_map.get(
        session.status, (session.status.value.upper(), "white", "•")
    )

    header = Text()
    header.append(f"{icon} Status: ", style="bold white")
    header.append(f"{status_str}\n\n")
    header.append("🎯 Goal: ", style="bold white")
    header.append(f"{session.goal}\n", style="white")
    header.append("🤖 Model: ", style="bold white")
    header.append(f"{session.model_name}  ", style="green")
    header.append("🔄 Iterations: ", style="bold white")
    header.append(f"{session.total_iterations}  ", style="cyan")
    header.append("⏱️ Duration: ", style="bold white")
    header.append(f"{session.total_duration_ms / 1000:.1f}s  ", style="yellow")
    header.append("🔤 Tokens: ", style="bold white")
    header.append(f"{session.total_tokens:,} (~{session.total_prompt_tokens:,} in / ~{session.total_completion_tokens:,} out)\n", style="dim")

    if session.test_summary:
        header.append("\n🧪 Tests: ", style="bold white")
        test_color = "green" if session.tests_passed else "red"
        header.append(f"{session.test_summary}\n", style=f"bold {test_color}")

    if session.files_modified:
        header.append("\n📝 Files Modified:\n", style="bold white")
        for f in session.files_modified:
            header.append(f"  • {f}\n", style="cyan")

    if session.final_summary:
        header.append(f"\n📋 Summary: {session.final_summary}\n", style="dim")

    console.print(
        Panel(
            header,
            title="[bold cyan]🛠️ Local Coding Agent Session[/bold cyan]",
            border_style=border_color,
            padding=(1, 2),
        )
    )

    # If stagnation was triggered, render diagnostics panel
    if session.stagnation_report and session.stagnation_report.is_stagnant:
        stag = session.stagnation_report
        stag_text = Text()
        stag_text.append(f"Reason: {stag.reason.value if stag.reason else 'unknown'}\n", style="bold yellow")
        stag_text.append(f"Diagnostic: {stag.message}\n", style="white")
        if stag.details:
            stag_text.append(f"Details: {stag.details}\n", style="dim")
        if stag.recommendations:
            stag_text.append("\nActionable Recommendations:\n", style="bold white")
            for rec in stag.recommendations:
                stag_text.append(f"  → {rec}\n", style="yellow")

        console.print(
            Panel(
                stag_text,
                title="[bold yellow]⚠️ Loop & Stagnation Diagnostics[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    console.print(
        f"[dim]Session ID: {session.session_id} | Detailed logs: .mileage/logs/coding_agent_{session.session_id}.log[/dim]\n"
    )


def render_evaluation_report(report) -> None:
    """Render an EvaluationReport as a styled Rich panel with per-requirement verdicts."""
    verdict_map = {
        "pass": ("[bold green]PASS[/bold green]", "green", "✅"),
        "partial": ("[bold yellow]PARTIAL[/bold yellow]", "yellow", "🟡"),
        "fail": ("[bold red]FAIL[/bold red]", "red", "🔴"),
    }
    v_str, border_color, icon = verdict_map.get(
        report.overall_verdict.value, ("UNKNOWN", "white", "•")
    )

    header = Text()
    header.append(f"{icon} Overall: ", style="bold white")
    header.append(f"{report.overall_verdict.value.upper()}", style=f"bold {border_color}")
    header.append(f" (confidence: {report.overall_confidence:.2f})\n\n", style="dim")
    header.append("🎯 Goal: ", style="bold white")
    header.append(f"{report.goal}\n", style="white")
    header.append("📊 Summary: ", style="bold white")
    header.append(
        f"{report.pass_count} pass, {report.partial_count} partial, "
        f"{report.fail_count} fail / {report.total_requirements} total\n",
        style="cyan",
    )

    if report.test_summary:
        header.append("\n🧪 Tests: ", style="bold white")
        test_color = "green" if report.tests_passed else "red"
        header.append(f"{report.test_summary}\n", style=f"bold {test_color}")

    console.print(
        Panel(
            header,
            title="[bold cyan]📋 Evaluator Report[/bold cyan]",
            border_style=border_color,
            padding=(1, 2),
        )
    )

    # Requirements table
    if report.requirement_results:
        table = Table(title="Requirement Verdicts", show_lines=True, padding=(0, 1))
        table.add_column("ID", style="bold cyan", width=6)
        table.add_column("Description", style="white", max_width=40)
        table.add_column("Verdict", justify="center", width=10)
        table.add_column("Conf", justify="center", width=6)
        table.add_column("Evidence", style="dim", max_width=40)

        for r in report.requirement_results:
            v_display, _, v_icon = verdict_map.get(r.verdict.value, ("?", "white", "?"))
            table.add_row(
                r.requirement_id,
                r.description[:40],
                f"{v_icon} {r.verdict.value.upper()}",
                f"{r.confidence:.2f}",
                r.evidence[:40] + ("…" if len(r.evidence) > 40 else ""),
            )
        console.print(table)

    # Acceptance criteria table
    if report.acceptance_results:
        table = Table(title="Acceptance Criteria Verdicts", show_lines=True, padding=(0, 1))
        table.add_column("ID", style="bold cyan", width=6)
        table.add_column("Description", style="white", max_width=40)
        table.add_column("Verdict", justify="center", width=10)
        table.add_column("Conf", justify="center", width=6)
        table.add_column("Evidence", style="dim", max_width=40)

        for a in report.acceptance_results:
            v_display, _, v_icon = verdict_map.get(a.verdict.value, ("?", "white", "?"))
            table.add_row(
                a.criterion_id,
                a.description[:40],
                f"{v_icon} {a.verdict.value.upper()}",
                f"{a.confidence:.2f}",
                a.evidence[:40] + ("…" if len(a.evidence) > 40 else ""),
            )
        console.print(table)

    # Overcapacity warning
    if report.overcapacity_detected:
        oc_text = Text()
        oc_text.append("Local model overcapacity detected!\n\n", style="bold red")
        for sig in report.overcapacity_signals:
            oc_text.append(f"  ⚡ {sig}\n", style="yellow")
        console.print(
            Panel(
                oc_text,
                title="[bold red]🔥 Overcapacity Detected[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
        )

    console.print(
        f"[dim]Evaluation time: {report.evaluation_duration_ms:.0f}ms | "
        f"Tokens: {report.evaluation_tokens:,} | "
        f"LLM eval used: {report.llm_evaluation_used}[/dim]\n"
    )


def render_escalation(record) -> None:
    """Render an escalation record with Copilot prompt info."""
    content = Text()
    content.append("🚨 Escalated to GitHub Copilot\n\n", style="bold red")
    content.append("Reason: ", style="bold white")
    content.append(f"{record.reason.value}\n", style="yellow")
    content.append("Detail: ", style="bold white")
    content.append(f"{record.reason_detail}\n\n", style="dim")
    content.append("🎯 Goal: ", style="bold white")
    content.append(f"{record.goal}\n", style="white")
    content.append("🤖 Local Model: ", style="bold white")
    content.append(f"{record.model_name}\n", style="green")
    content.append("🔄 Local Attempts: ", style="bold white")
    content.append(f"{record.local_attempts}\n", style="cyan")

    if record.vscode_opened:
        content.append("\n✅ VS Code opened with workspace & prompt file\n", style="bold green")
    else:
        content.append("\n⚠️ VS Code could not be opened automatically\n", style="bold yellow")

    content.append("\n📄 Copilot Prompt File: ", style="bold white")
    content.append(f"{record.copilot_prompt_file}\n", style="cyan")
    content.append(
        "[dim]Open this file in VS Code, select all, and paste into Copilot Chat.[/dim]\n"
    )

    console.print(
        Panel(
            content,
            title="[bold magenta]🤖 Copilot Escalation[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )
