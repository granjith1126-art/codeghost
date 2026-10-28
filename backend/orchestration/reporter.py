"""Reporting utilities for CodeGhost Orchestration."""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.orchestration.state_store import StateStore, WorkflowState

console = Console()


class OrchestrationReporter:
    """Renders formatted status and benchmark summaries."""

    def __init__(self, state_store: StateStore):
        self.state_store = state_store

    def print_status(self) -> None:
        """Display current workflow state, phase statuses, and budget usage in terminal."""
        state = self.state_store.state

        console.print()
        console.rule(f"[bold cyan]CodeGhost Orchestration Status: {state.milestone}[/bold cyan]")
        console.print()

        info_table = Table(show_header=False, box=None)
        info_table.add_column("Key", style="bold")
        info_table.add_column("Value")
        info_table.add_row("Workflow Status", f"[bold green]{state.status}[/bold green]")
        info_table.add_row("Started At", state.started_at)
        info_table.add_row("Last Updated", state.updated_at)
        info_table.add_row("OpenAI API Requests", str(state.api_requests_count))
        info_table.add_row("Tokens Consumed", f"{state.prompt_tokens_used + state.completion_tokens_used:,} (Prompt: {state.prompt_tokens_used:,}, Comp: {state.completion_tokens_used:,})")
        info_table.add_row("Estimated API Cost", f"${state.estimated_cost_usd:.4f}")
        console.print(Panel(info_table, title="Session Overview", border_style="cyan"))
        console.print()

        # Phase table
        phase_table = Table(title="Milestone 8 Phase Progression", show_header=True, header_style="bold magenta")
        phase_table.add_column("Phase", style="cyan")
        phase_table.add_column("Title", style="bold")
        phase_table.add_column("Status", style="bold")
        phase_table.add_column("Review", style="italic")
        phase_table.add_column("Artifacts", style="dim")

        for pid, p in state.phases.items():
            status_style = (
                "[green]PASSED[/green]" if p.status == "PASSED"
                else "[yellow]PASSED (LIMITATIONS)[/yellow]" if p.status == "PASSED_WITH_LIMITATIONS"
                else "[cyan]IN PROGRESS[/cyan]" if p.status == "IN_PROGRESS"
                else "[red]FAILED[/red]" if p.status == "FAILED"
                else "[magenta]BLOCKED[/magenta]" if p.status == "BLOCKED"
                else "[dim]PENDING[/dim]"
            )
            review_display = (
                "[green]APPROVED[/green]" if getattr(p, "review_verdict", None) == "APPROVED"
                else f"[yellow]{p.review_verdict}[/yellow]" if getattr(p, "review_verdict", None)
                else "-"
            )
            art_summary = ", ".join([a.split("/")[-1] for a in p.artifacts]) if p.artifacts else "-"
            phase_table.add_row(pid.replace("_", " ").upper(), p.title, status_style, review_display, art_summary)

        console.print(phase_table)
        console.print()
