"""CodeGhost CLI: Ingestion, Historical Bug Extraction, Vector Indexing & RAG Retrieval."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Ensure project root is in sys.path when executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure Windows console encoding does not crash on unicode characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

from backend.history.extractor import HistoryExtractor
from backend.ingestion.extractor import RepositoryExtractor
from backend.models.schemas import (
    BugFixPair,
    CodeUnit,
    HistoryExtractionSummary,
    IngestionResult,
)
from backend.autofix.generator import FixGenerator
from backend.autofix.sandbox import TestSandbox
from backend.autofix.schemas import FixCandidate, FixStrategy, VerifiedFix
from backend.pr.creator import PRCreator
from backend.pr.formatter import PRFormatter
from backend.pr.schemas import PRDraft, PRResult
from backend.reasoning.explainer import ReasoningEngine
from backend.reasoning.schemas import ComplexityFactor, HistoricalPrecedent, RiskExplanation
from backend.retrieval.embedder import FastEmbedEmbedder
from backend.retrieval.indexer import BugFixIndexer
from backend.retrieval.retriever import FunctionRetriever
from backend.retrieval.schemas import (
    EvaluationResult,
    IndexingStats,
    RepresentationType,
    RetrievalQuery,
    RetrievedMatch,
)
from backend.retrieval.vector_store import QdrantVectorStore
from backend.risk.baseline_model import BaselineRiskModel
from backend.risk.dataset_builder import RiskDatasetBuilder
from backend.risk.evaluator import RiskEvaluator
from backend.risk.llm_scorer import MockLLMScorer, QwenLLMScorer
from backend.risk.schemas import (
    DatasetSplits,
    RiskCategory,
    RiskEvaluationMetrics,
    RiskExample,
    RiskPrediction,
    RiskThresholds,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich output formatting."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def print_ingestion_report(result: IngestionResult, preview_count: int = 5) -> None:
    """Print summary report of the Milestone 1 ingestion metrics."""
    console.print()
    console.rule("[bold cyan]CodeGhost Repository Ingestion Report[/bold cyan]")
    console.print()

    table = Table(title="Ingestion Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Repository", result.repository_url)
    table.add_row("Commit SHA", result.commit_hash)
    table.add_row("Local Path", result.local_path)
    table.add_row("Files Discovered", str(result.files_discovered))
    table.add_row("Files Successfully Parsed", str(result.files_parsed))
    table.add_row("Unsupported / Ignored Files", str(result.unsupported_files))
    table.add_row("Functions Extracted", str(result.functions_extracted))
    table.add_row("Classes Extracted", str(result.classes_extracted))
    table.add_row(
        "Parsing Errors",
        f"[red]{len(result.errors)}[/red]" if result.errors else "[green]0[/green]",
    )
    table.add_row("JSONL Output File", result.output_file or "N/A")

    console.print(table)
    console.print()

    if result.languages_detected:
        lang_table = Table(title="Detected Languages", show_header=True, header_style="bold blue")
        lang_table.add_column("Language", style="cyan")
        lang_table.add_column("Files", style="yellow", justify="right")

        for lang, count in sorted(result.languages_detected.items(), key=lambda x: x[1], reverse=True):
            lang_table.add_row(lang.capitalize(), str(count))

        console.print(lang_table)
        console.print()

    if result.errors:
        err_table = Table(title="Parsing Errors & Warnings", show_header=True, header_style="bold red")
        err_table.add_column("File Path", style="yellow")
        err_table.add_column("Error Type", style="red")
        err_table.add_column("Details")

        for err in result.errors[:10]:
            err_table.add_row(err.file_path, err.error_type, err.message)

        if len(result.errors) > 10:
            err_table.add_row("...", f"and {len(result.errors) - 10} more errors", "...")

        console.print(err_table)
        console.print()


def print_history_report(summary: HistoryExtractionSummary, preview_count: int = 10) -> None:
    """Print summary report of the Milestone 2 historical bug extraction."""
    console.print()
    console.rule("[bold cyan]CodeGhost Historical Bug Extraction Report[/bold cyan]")
    console.print()

    table = Table(title="Commit History & Bug-Fix Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Repository", summary.repository_url)
    table.add_row("Commits Scanned", str(summary.commits_scanned))
    table.add_row("Bug-Fix Candidates", f"[bold green]{summary.bug_fix_candidates}[/bold green]")
    table.add_row("Uncertain Commits", f"[yellow]{summary.uncertain_commits}[/yellow]")
    table.add_row("Non-Bug Commits", str(summary.non_bug_commits))
    table.add_row("Files Changed", str(summary.files_changed))
    table.add_row("Functions Changed", str(summary.functions_changed))
    table.add_row("Bug-Fix Pairs Extracted", f"[bold cyan]{summary.bug_fix_pairs_extracted}[/bold cyan]")
    if summary.non_semantic_pairs_filtered:
        table.add_row("Non-Semantic Filtered", f"[yellow]{summary.non_semantic_pairs_filtered}[/yellow]")
    table.add_row(
        "Extraction Failures",
        f"[red]{summary.extraction_failures}[/red]" if summary.extraction_failures else "[green]0[/green]",
    )
    table.add_row("Raw Commits File", summary.raw_commits_file or "N/A")
    table.add_row("Processed Bug-Fix Pairs", summary.processed_pairs_file or "N/A")

    console.print(table)
    console.print()


def print_indexing_report(stats: IndexingStats) -> None:
    """Print summary report of Milestone 3 vector indexing."""
    console.print()
    console.rule("[bold cyan]CodeGhost Vector Indexing Report[/bold cyan]")
    console.print()

    table = Table(title="Vector Indexing Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Collection Name", stats.collection_name)
    table.add_row("Dataset Records", str(stats.dataset_records))
    table.add_row("Valid Records", str(stats.valid_records))
    table.add_row("Vectors Embedded", str(stats.embedded))
    table.add_row("Vectors Indexed in Qdrant", f"[bold green]{stats.indexed}[/bold green]")
    table.add_row("Indexing Failures", f"[red]{stats.failed}[/red]" if stats.failed else "[green]0[/green]")
    table.add_row("Vector Dimensionality", str(stats.vector_dimension))
    table.add_row("Elapsed Time", f"{stats.elapsed_seconds:.2f}s")

    console.print(table)
    console.print()


def print_retrieval_report(matches: List[RetrievedMatch], query_title: str) -> None:
    """Print top-k retrieval results with rich side-by-side code blocks."""
    console.print()
    console.rule(f"[bold cyan]Top {len(matches)} Historical Matches for '{query_title}'[/bold cyan]")
    console.print()

    if not matches:
        console.print("[yellow]No historically similar bug-fix examples found matching the criteria.[/yellow]")
        return

    for i, m in enumerate(matches, 1):
        console.print(
            f"[bold magenta]Match #{i}:[/bold magenta] [bold yellow]{m.function_name}[/bold yellow] "
            f"in [cyan]{m.file_path}[/cyan] (Score: [bold green]{m.similarity_score:.4f}[/bold green] | Commit: [blue]{m.commit_sha[:8]}[/blue])"
        )
        console.print(f"[italic]Commit Message:[/italic] {m.commit_message.splitlines()[0]}")
        console.print(f"[italic]Classification:[/italic] {m.detector_classification} ({m.detector_score:.2f})")

        # Side by side buggy vs fixed code
        buggy_syntax = Syntax(m.buggy_code, "python", theme="monokai", line_numbers=True, word_wrap=True)
        fixed_syntax = Syntax(m.fixed_code, "python", theme="monokai", line_numbers=True, word_wrap=True)

        b_panel = Panel(buggy_syntax, title="[bold red]Historical Buggy Code[/bold red]", border_style="red")
        f_panel = Panel(fixed_syntax, title="[bold green]Historical Fixed Code[/bold green]", border_style="green")
        console.print(Columns([b_panel, f_panel], equal=True))
        console.print()


def print_evaluation_report(results: List[EvaluationResult]) -> None:
    """Print evaluation benchmark comparison table."""
    console.print()
    console.rule("[bold cyan]CodeGhost Retrieval Evaluation Benchmark[/bold cyan]")
    console.print()

    table = Table(title="Retrieval Metrics & Baseline Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Method / Representation", style="bold cyan")
    table.add_column("Queries", justify="right")
    table.add_column("Recall@1", justify="right")
    table.add_column("Recall@3", justify="right")
    table.add_column("Recall@5", justify="right", style="bold green")
    table.add_column("Recall@10", justify="right")
    table.add_column("MRR", justify="right", style="bold yellow")
    table.add_column("Latency", justify="right")

    for r in results:
        table.add_row(
            r.method_name,
            str(r.total_queries),
            f"{r.recall_at_1:.1f}%",
            f"{r.recall_at_3:.1f}%",
            f"{r.recall_at_5:.1f}%",
            f"{r.recall_at_10:.1f}%",
            f"{r.mrr:.4f}",
            f"{r.avg_latency_ms:.1f} ms",
        )

    console.print(table)
    console.print()


def print_risk_dataset_report(splits: DatasetSplits) -> None:
    """Print summary report of the Milestone 4 dataset construction and temporal splits."""
    console.print()
    console.rule("[bold cyan]CodeGhost Risk Scoring: Dataset Construction & Temporal Split[/bold cyan]")
    console.print()

    table = Table(title="Dataset Partitioning (Temporal Backtest Split)", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="bold cyan")
    table.add_column("Total Samples", justify="right")
    table.add_column("Positives (Buggy)", justify="right", style="red")
    table.add_column("Negatives (Stable)", justify="right", style="green")
    table.add_column("Class Balance", justify="right")
    table.add_column("Temporal Cutoff", justify="right", style="yellow")

    for name, sample_list in [("Train (Oldest 70%)", splits.train), ("Validation (Next 15%)", splits.val), ("Test (Latest 15%)", splits.test)]:
        total = len(sample_list)
        pos = sum(1 for s in sample_list if s.label == 1)
        neg = total - pos
        ratio = f"{pos / total:.1%}" if total > 0 else "0.0%"
        cutoff = sample_list[-1].timestamp[:10] if sample_list else "N/A"
        table.add_row(name, str(total), str(pos), str(neg), ratio, cutoff)

    total_all = len(splits.train) + len(splits.val) + len(splits.test)
    pos_all = sum(1 for s in (splits.train + splits.val + splits.test) if s.label == 1)
    neg_all = total_all - pos_all
    table.add_row(
        "[bold]Total Dataset[/bold]",
        f"[bold]{total_all}[/bold]",
        f"[bold red]{pos_all}[/bold red]",
        f"[bold green]{neg_all}[/bold green]",
        f"[bold]{pos_all/total_all:.1%}[/bold]" if total_all else "0.0%",
        "Temporal Ascending",
    )

    console.print(table)
    console.print()


def print_risk_metrics_report(metrics: RiskEvaluationMetrics) -> None:
    """Print comprehensive evaluation metrics and confusion matrix for risk models."""
    console.print()
    console.rule(f"[bold cyan]Risk Evaluation Metrics: {metrics.model_name} ({metrics.split_name.upper()} SET)[/bold cyan]")
    console.print()

    # Metrics table
    table = Table(title=f"Performance Metrics (Threshold: {metrics.threshold})", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Model Name", metrics.model_name)
    table.add_row("Evaluated Split", metrics.split_name)
    table.add_row("Total Test Samples", str(metrics.total_samples))
    table.add_row("Positives / Negatives", f"{metrics.positive_samples} / {metrics.negative_samples} (Balance: {metrics.class_balance_ratio:.1%})")
    table.add_row("Precision", f"{metrics.precision:.4f}")
    table.add_row("Recall", f"{metrics.recall:.4f}")
    table.add_row("F1-Score", f"[bold green]{metrics.f1:.4f}[/bold green]")
    table.add_row("PR-AUC", f"[bold yellow]{metrics.pr_auc:.4f}[/bold yellow]" if metrics.pr_auc is not None else "N/A")
    table.add_row("ROC-AUC", f"{metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "N/A")
    table.add_row("False Positive Rate (FPR)", f"[red]{metrics.false_positive_rate:.4f}[/red]")
    table.add_row("Average Inference Latency", f"{metrics.latency_ms:.2f} ms")

    console.print(table)
    console.print()

    # Confusion matrix
    if metrics.confusion_matrix and len(metrics.confusion_matrix) == 2:
        cm_table = Table(title="Confusion Matrix", show_header=True, header_style="bold blue")
        cm_table.add_column("Actual \\ Predicted", style="bold")
        cm_table.add_column("Predicted Safe (0)", justify="right")
        cm_table.add_column("Predicted Risky (1)", justify="right")

        tn, fp = metrics.confusion_matrix[0]
        fn, tp = metrics.confusion_matrix[1]

        cm_table.add_row("Actual Safe (0)", f"[green]TN: {tn}[/green]", f"[red]FP: {fp}[/red]")
        cm_table.add_row("Actual Risky (1)", f"[red]FN: {fn}[/red]", f"[bold green]TP: {tp}[/bold green]")
        console.print(cm_table)
        console.print()

    # Categorical Score Distribution
    dist_table = Table(title="Risk Score Distribution (1-10 Scale)", show_header=True, header_style="bold yellow")
    dist_table.add_column("Category", style="bold")
    dist_table.add_column("Score Range", style="italic")
    dist_table.add_column("Predicted Count", justify="right")

    dist = metrics.score_distribution
    dist_table.add_row("[green]LOW[/green]", "1.0 - 3.9", str(dist.get("LOW", 0)))
    dist_table.add_row("[yellow]MEDIUM[/yellow]", "4.0 - 6.9", str(dist.get("MEDIUM", 0)))
    dist_table.add_row("[orange3]HIGH[/orange3]", "7.0 - 8.9", str(dist.get("HIGH", 0)))
    dist_table.add_row("[bold red]CRITICAL[/bold red]", "9.0 - 10.0", str(dist.get("CRITICAL", 0)))
    console.print(dist_table)
    console.print()


def print_risk_prediction_report(pred: RiskPrediction) -> None:
    """Print detailed risk prediction card for a single function."""
    console.print()
    console.rule(f"[bold cyan]Risk Prediction for '{pred.function_name}'[/bold cyan]")
    console.print()

    cat_colors = {
        "LOW": "green",
        "MEDIUM": "yellow",
        "HIGH": "orange3",
        "CRITICAL": "bold red",
    }
    cat_str = pred.risk_category.value if hasattr(pred.risk_category, "value") else str(pred.risk_category)
    color = cat_colors.get(cat_str, "white")

    console.print(f"[bold]Target Function:[/bold] [yellow]{pred.function_name}[/yellow] in [cyan]{pred.file_path}[/cyan]")
    console.print(f"[bold]Evaluating Model:[/bold] {pred.model_name}")
    console.print(f"[bold]Risk Probability:[/bold] [cyan]{pred.risk_probability:.4f}[/cyan] ({pred.risk_probability*100:.1f}%)")
    console.print(f"[bold]Numerical Risk Score:[/bold] [{color}]{pred.risk_score:.1f} / 10.0[/{color}]")
    console.print(f"[bold]Risk Category:[/bold] [{color}]{cat_str}[/{color}]")
    console.print(f"[bold]Binary Classification:[/bold] [{'red' if pred.is_risky else 'green'}]{'PREDICTED RISKY' if pred.is_risky else 'PREDICTED SAFE'}[/{'red' if pred.is_risky else 'green'}]")
    if pred.reasoning_summary:
        console.print(f"[italic]{pred.reasoning_summary}[/italic]")
    console.print()

    if pred.top_retrieved_matches:
        match_table = Table(title="Top Retrieved Historical Bug Evidence (Milestone 3)", show_header=True, header_style="bold magenta")
        match_table.add_column("#", style="bold")
        match_table.add_column("Function", style="yellow")
        match_table.add_column("File", style="cyan")
        match_table.add_column("Similarity", justify="right", style="green")
        match_table.add_column("Commit", style="blue")
        match_table.add_column("Commit Message")

        for idx, m in enumerate(pred.top_retrieved_matches[:3], 1):
            msg_line = m.commit_message.splitlines()[0][:45] if m.commit_message else "N/A"
            match_table.add_row(
                str(idx),
                m.function_name,
                m.file_path,
                f"{m.similarity_score:.4f}",
                m.commit_sha[:8],
                msg_line,
            )
        console.print(match_table)
        console.print()

    if pred.features_used:
        feat_table = Table(title="Key Feature Values Used in Prediction", show_header=True, header_style="bold blue")
        feat_table.add_column("Feature", style="bold")
        feat_table.add_column("Value", justify="right", style="green")
        for k, v in list(pred.features_used.items())[:8]:
            feat_table.add_row(k, f"{v:.4f}")
        console.print(feat_table)
        console.print()

    console.print(Panel(
        "[bold yellow]Notice:[/bold yellow] Similarity to historical bugs is evidence, NOT proof that this function contains a bug.\n"
        "Predictions reflect statistical risk based on historical change patterns and code complexity.",
        border_style="yellow",
    ))
    console.print()


def print_explanation_report(exp: RiskExplanation) -> None:
    """Display rich explanation report for Milestone 5."""
    console.print()
    console.rule(f"[bold cyan]CodeGhost Explainability & Reasoning: {exp.function_name}[/bold cyan]")
    console.print()

    cat_str = getattr(exp.risk_category, "value", str(exp.risk_category))
    color = "red" if exp.risk_score >= 8.5 else "orange3" if exp.risk_score >= 7.0 else "yellow" if exp.risk_score >= 4.0 else "green"

    summary_panel = Panel(
        f"[bold]Summary:[/bold] {exp.summary}\n\n"
        f"[bold]Risk Assessment:[/bold] [{color}]{exp.risk_score:.1f} / 10.0 ({cat_str})[/{color}] | "
        f"[bold]Provider:[/bold] [cyan]{exp.provider}[/cyan] ({exp.model_name})",
        title=f"Assessment Overview: {exp.function_name}",
        border_style=color,
    )
    console.print(summary_panel)
    console.print()

    if exp.complexity_factors:
        comp_table = Table(title="AST & Structural Complexity Drivers", show_header=True, header_style="bold magenta")
        comp_table.add_column("Driver", style="bold")
        comp_table.add_column("Value", justify="right", style="cyan")
        comp_table.add_column("Impact", style="yellow")
        comp_table.add_column("Diagnostic Description")
        for f in exp.complexity_factors:
            comp_table.add_row(f.name, str(f.value), f.risk_impact, f.description)
        console.print(comp_table)
        console.print()

    if exp.historical_precedents:
        hist_table = Table(title="Historical Precedents & Analogous Bugs", show_header=True, header_style="bold blue")
        hist_table.add_column("Commit", style="cyan")
        hist_table.add_column("Similarity", justify="right", style="green")
        hist_table.add_column("Function", style="yellow")
        hist_table.add_column("Commit Message & Relevance")
        for p in exp.historical_precedents[:3]:
            hist_table.add_row(
                p.commit_sha[:8],
                f"{p.similarity_score:.1%}",
                p.function_name,
                f"[bold]{p.commit_message}[/bold]\n[dim]{p.relevance_reason}[/dim]",
            )
        console.print(hist_table)
        console.print()

    if exp.vulnerable_patterns:
        pat_panel = Panel(
            "\n".join([f"* {p}" for p in exp.vulnerable_patterns]),
            title="Identified Code Hazards & Vulnerability Patterns",
            border_style="red",
        )
        console.print(pat_panel)
        console.print()

    if exp.mitigation_strategies:
        mit_panel = Panel(
            "\n".join([f"[+] {m}" for m in exp.mitigation_strategies]),
            title="Actionable Mitigation Strategies",
            border_style="green",
        )
        console.print(mit_panel)
        console.print()


def print_fixes_report(verified_fixes: List[VerifiedFix]) -> None:
    """Display rich automated fixes report for Milestone 6."""
    console.print()
    console.rule("[bold cyan]CodeGhost Automated Fix Candidates & Sandbox Verification[/bold cyan]")
    console.print()

    if not verified_fixes:
        console.print("[yellow]No viable fix candidates could be synthesized for this pattern.[/yellow]")
        return

    table = Table(title="Synthesized Fix Candidates", show_header=True, header_style="bold magenta")
    table.add_column("Fix ID", style="bold")
    table.add_column("Strategy", style="cyan")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Syntax", style="blue")
    table.add_column("Sandbox Verdict", style="bold")
    table.add_column("Description")

    for vf in verified_fixes:
        c = vf.candidate
        sb = vf.sandbox_result
        strat_str = c.strategy.value if hasattr(c.strategy, "value") else str(c.strategy)
        status_str = "[green]PASSED[/green]" if vf.is_verified else "[red]FAILED[/red]"
        syntax_str = "[green]VALID[/green]" if vf.syntax_valid else "[red]SYNTAX ERROR[/red]"
        table.add_row(
            c.fix_id,
            strat_str,
            f"{c.confidence:.0%}",
            syntax_str,
            status_str,
            c.description,
        )
    console.print(table)
    console.print()

    # Show top diff
    top_fix = verified_fixes[0]
    syntax = Syntax(top_fix.candidate.diff, "diff", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=f"Top Verified Fix Diff: {top_fix.candidate.fix_id} ({top_fix.verification_summary})", border_style="green" if top_fix.is_verified else "yellow"))
    console.print()


def print_pr_report(pr_res: PRResult) -> None:
    """Display rich GitHub PR report for Milestone 7."""
    console.print()
    console.rule("[bold cyan]CodeGhost GitHub Pull Request Dispatch[/bold cyan]")
    console.print()

    draft = pr_res.draft
    status_text = "[bold green]LIVE PR CREATED[/bold green]" if pr_res.pr_url else "[bold yellow]DRY-RUN READY[/bold yellow]"

    panel_content = [
        f"[bold]Status:[/bold] {status_text}",
        f"[bold]Summary:[/bold] {pr_res.message}",
    ]
    if pr_res.pr_url:
        panel_content.append(f"[bold]Pull Request URL:[/bold] [link={pr_res.pr_url}]{pr_res.pr_url}[/link]")
    if pr_res.patch_file_path:
        panel_content.append(f"[bold]Serialized Patch File:[/bold] [cyan]{pr_res.patch_file_path}[/cyan]")
    if draft:
        panel_content.extend([
            f"[bold]Title:[/bold] {draft.title}",
            f"[bold]Target Branch:[/bold] {draft.base_branch} ← [cyan]{draft.branch_name}[/cyan]",
            f"[bold]Risk Score:[/bold] {draft.risk_score} ({draft.risk_category})",
        ])

    console.print(Panel("\n".join(panel_content), title="Pull Request Deployment Summary", border_style="cyan"))
    console.print()


def main(argv: Optional[list] = None) -> int:
    """CLI entrypoint supporting Ingestion, History Mining, Vector Indexing, and RAG Retrieval."""
    parser = argparse.ArgumentParser(
        description="CodeGhost - AST Ingestion, Bug Mining, and Function Retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "repo_url_pos",
        nargs="?",
        default=None,
        help="Public GitHub repository URL or local directory path",
    )
    parser.add_argument(
        "--repo",
        dest="repo_url_flag",
        default=None,
        help="Public GitHub repository URL or local directory path (named option)",
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "history", "index", "retrieve", "evaluate", "risk", "explain", "fix", "pr", "scan", "all"],
        default=None,
        help="Execution mode: 'ingest', 'history', 'index', 'retrieve', 'evaluate', 'risk', 'explain', 'fix', 'pr', 'scan', or 'all'",
    )
    parser.add_argument(
        "--provider",
        choices=["deterministic", "gemini", "openai"],
        default="deterministic",
        help="Reasoning provider for explainability and fixes",
    )
    parser.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        help="Submit live pull request to GitHub API (requires GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Run in dry-run mode without external mutations (default: True)",
    )
    parser.set_defaults(dry_run=True)
    parser.add_argument(
        "--github-token",
        default=None,
        help="GitHub personal access token for live PR creation",
    )
    parser.add_argument(
        "--base-branch",
        default="main",
        help="Target base branch for Pull Requests",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "llm", "compare"],
        default="baseline",
        help="Risk model type: 'baseline', 'llm', or 'compare'",
    )
    parser.add_argument(
        "--baseline-type",
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
        default="random_forest",
        help="Classical ML baseline algorithm",
    )
    parser.add_argument(
        "--risk-splits-dir",
        default="backend/data/processed/risk_splits",
        help="Directory to save/load train/val/test splits for risk scoring",
    )
    parser.add_argument(
        "--units-file",
        default=None,
        help="Path to code_units.jsonl for negative sampling",
    )
    parser.add_argument(
        "--model-save-path",
        default="backend/data/models/baseline_model.joblib",
        help="Path to save or load trained risk model",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Classification threshold for positive (risky) prediction",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate optimal decision threshold on validation split",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Force rebuild of risk dataset splits from raw pairs and units",
    )
    parser.add_argument(
        "--collection",
        default="codeghost_bug_history",
        help="Qdrant vector collection name",
    )
    parser.add_argument(
        "--representation",
        choices=["code_only", "code_and_metadata", "code_and_context"],
        default="code_and_metadata",
        help="Code representation strategy to use",
    )
    parser.add_argument(
        "--function",
        default=None,
        help="Target function name to query in retrieve/risk mode (e.g. 'Flask.run')",
    )
    parser.add_argument(
        "--pairs-file",
        default=None,
        help="Path to bug_fix_pairs.jsonl dataset",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=None,
        help="Maximum commits to scan for historical bug extraction",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest historical matches to retrieve",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Lower date bound for commit scanning",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Upper date bound for commit scanning",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Base storage directory",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Path for output JSONL file (ingest mode)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for processed bug-fix pairs JSONL (history mode)",
    )
    parser.add_argument(
        "--branch",
        "-b",
        default=None,
        help="Specific branch or tag to inspect",
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=None,
        help="Clone depth",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Force clean re-cloning if already present",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of examples to preview in console",
    )
    parser.add_argument(
        "--process-uncertain",
        action="store_true",
        help="Also attempt bug-fix extraction on UNCERTAIN commits",
    )
    parser.add_argument(
        "--filter-non-semantic",
        action="store_true",
        help="Filter out non-semantic candidate pairs (comments, docstrings, formatting) during history extraction",
    )
    parser.add_argument(
        "--cross-repo",
        action="store_true",
        help="Disable repository isolation filter during retrieval",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    repo_url = args.repo_url_flag or args.repo_url_pos
    mode = args.mode

    # Auto-detect default mode
    if mode is None:
        if args.function is not None:
            mode = "retrieve"
        elif args.max_commits is not None:
            mode = "history"
        elif repo_url:
            mode = "ingest"
        else:
            mode = "index"

    storage_dir = Path(args.output_dir) if args.output_dir else None
    pairs_file = (
        Path(args.pairs_file).resolve()
        if args.pairs_file
        else Path(__file__).resolve().parent / "data" / "processed" / "bug_fix_pairs.jsonl"
    )

    # MODE 1: Ingestion (Milestone 1)
    if mode in ("ingest", "all"):
        if not repo_url:
            console.print("[bold red]Error:[/bold red] Repository URL is required for ingestion.")
            return 1
        console.print(f"[bold cyan]Initializing CodeGhost Ingestion (Milestone 1)...[/bold cyan]")
        extractor = RepositoryExtractor(base_storage_dir=storage_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("[cyan]Ingesting repository...", total=100)

            def progress_update(current: int, total: int, current_file: str):
                progress.update(
                    task_id,
                    total=total,
                    completed=current,
                    description=f"[cyan]Parsing ({current}/{total}): [yellow]{current_file[:30]}[/yellow]...",
                )

            try:
                ingest_res = extractor.ingest(
                    repo_url_or_path=repo_url,
                    output_jsonl_path=Path(args.output_jsonl) if args.output_jsonl else None,
                    branch=args.branch,
                    depth=args.depth,
                    force_clean=args.clean,
                    progress_callback=progress_update,
                )
                progress.update(task_id, completed=100, description="[bold green]Ingestion complete!")
            except Exception as e:
                console.print(f"[bold red]Ingestion failed: {e}[/bold red]")
                if args.verbose:
                    console.print_exception()
                return 1

        print_ingestion_report(ingest_res, preview_count=args.preview)

    # MODE 2: Historical Bug Extraction (Milestone 2)
    if mode in ("history", "all"):
        if not repo_url:
            console.print("[bold red]Error:[/bold red] Repository URL is required for history extraction.")
            return 1
        console.print(f"[bold cyan]Initializing CodeGhost Historical Bug Extraction (Milestone 2)...[/bold cyan]")
        history_extractor = HistoryExtractor(base_storage_dir=storage_dir)
        max_commits = args.max_commits if args.max_commits is not None else 500

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("[cyan]Mining commit history...", total=max_commits)

            def history_update(current: int, total: int, msg: str):
                progress.update(
                    task_id,
                    total=total,
                    completed=current,
                    description=f"[cyan]Scanning ({current}/{total}): [yellow]{msg[:35]}[/yellow]...",
                )

            try:
                hist_res = history_extractor.extract_history(
                    repo_url_or_path=repo_url,
                    max_commits=max_commits,
                    since=args.since,
                    until=args.until,
                    branch=args.branch,
                    processed_pairs_path=Path(args.output) if args.output else None,
                    process_uncertain=args.process_uncertain,
                    filter_non_semantic=args.filter_non_semantic,
                    progress_callback=history_update,
                )
                progress.update(task_id, completed=max_commits, description="[bold green]Mining complete!")
            except Exception as e:
                console.print(f"[bold red]Historical extraction failed: {e}[/bold red]")
                if args.verbose:
                    console.print_exception()
                return 1

        print_history_report(hist_res, preview_count=args.preview)

    # MODE 3: Vector Indexing (Milestone 3)
    if mode in ("index", "all"):
        console.print(f"[bold cyan]Initializing Vector Indexing into Qdrant (Milestone 3)...[/bold cyan]")
        console.print(f"Source Dataset: [green]{pairs_file}[/green]")
        console.print(f"Collection: [yellow]{args.collection}[/yellow] | Strategy: [magenta]{args.representation}[/magenta]")

        v_store = QdrantVectorStore()
        embedder = FastEmbedEmbedder()
        indexer = BugFixIndexer(
            vector_store=v_store,
            embedder=embedder,
            collection_name=args.collection,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("[cyan]Embedding and indexing bug-fix pairs...", total=100)

            def index_update(current: int, total: int, msg: str):
                progress.update(
                    task_id,
                    total=total,
                    completed=current,
                    description=f"[cyan]{msg}...",
                )

            try:
                stats = indexer.index_file(
                    jsonl_path=pairs_file,
                    representation_type=args.representation,
                    recreate_collection=True,
                    progress_callback=index_update,
                )
                progress.update(task_id, completed=stats.valid_records, description="[bold green]Indexing complete!")
            except Exception as e:
                console.print(f"[bold red]Vector indexing failed: {e}[/bold red]")
                if args.verbose:
                    console.print_exception()
                return 1

        print_indexing_report(stats)

    # MODE 4: Function Retrieval (Milestone 3)
    if mode in ("retrieve", "all"):
        console.print(f"[bold cyan]Initializing Function Retrieval (Milestone 3)...[/bold cyan]")

        # Find target query function
        target_fn = args.function
        query_code = None
        query_repo = None if args.cross_repo else repo_url
        query_file = None
        query_class = None

        if target_fn:
            # Search in bug_fix_pairs.jsonl or raw units
            if pairs_file.exists():
                with open(pairs_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            p = BugFixPair.model_validate_json(line)
                            if target_fn.lower() in p.function_name.lower():
                                query_code = p.buggy_code
                                query_repo = p.repository if not args.cross_repo else None
                                query_file = p.file_path
                                query_class = p.class_name
                                target_fn = p.function_name
                                break

        if not query_code:
            # Fallback default function query if not found
            query_code = "def run(self, host=None, port=None, debug=None, **options):\n    pass"
            target_fn = target_fn or "Flask.run"
            query_repo = repo_url if not args.cross_repo else None

        v_store = QdrantVectorStore()
        embedder = FastEmbedEmbedder()
        retriever = FunctionRetriever(
            vector_store=v_store,
            embedder=embedder,
            collection_name=args.collection,
        )

        query = RetrievalQuery(
            code=query_code,
            repository=query_repo,
            function_name=target_fn,
            file_path=query_file,
            class_name=query_class,
            representation_type=args.representation,
        )

        console.print(f"Query Function: [bold yellow]{target_fn}[/bold yellow]")
        if query_repo:
            console.print(f"Repository Isolation: [green]{query_repo}[/green]")
        else:
            console.print("[yellow]Repository Isolation: Disabled (Cross-Repository Search)[/yellow]")

        matches = retriever.retrieve(query=query, top_k=args.top_k)
        print_retrieval_report(matches, target_fn)

    # MODE 5: Retrieval Evaluation (Milestone 3)
    if mode in ("evaluate", "all"):
        console.print(f"[bold cyan]Running Retrieval Evaluation Benchmark (Milestone 3)...[/bold cyan]")
        from evaluation.retrieval_eval import run_representation_experiment

        if not pairs_file.exists():
            console.print(f"[bold red]Error:[/bold red] Pairs dataset not found at {pairs_file}")
            return 1

        results = run_representation_experiment(pairs_file=pairs_file)
        print_evaluation_report(results)

    # MODE 6: Risk Scoring & Model Evaluation (Milestone 4)
    if mode in ("risk", "all"):
        console.print(f"[bold cyan]Initializing CodeGhost Risk Scoring (Milestone 4)...[/bold cyan]")
        splits_dir = Path(args.risk_splits_dir).resolve()
        model_save_path = Path(args.model_save_path).resolve()
        units_path = Path(args.units_file).resolve() if args.units_file else ROOT_DIR / "backend" / "data" / "raw" / "pallets_flask_units.jsonl"

        v_store = QdrantVectorStore()
        embedder = FastEmbedEmbedder()
        retriever = FunctionRetriever(
            vector_store=v_store,
            embedder=embedder,
            collection_name=args.collection,
        )

        builder = RiskDatasetBuilder(retriever=retriever, collection_name=args.collection)

        # 1. Dataset Construction & Splitting
        need_rebuild = args.rebuild_dataset or not (splits_dir / "test.jsonl").exists()
        if need_rebuild:
            console.print("[cyan]Constructing labeled risk dataset with temporal backtest splitting...[/cyan]")
            pos_examples = builder.build_positive_examples(pairs_file)
            neg_examples = builder.build_negative_examples(
                pairs=pos_examples,
                units_path=units_path if units_path.exists() else None,
                target_count=len(pos_examples),
            )
            all_examples = pos_examples + neg_examples

            console.print(f"[cyan]Querying vector database for historical bug context ({len(all_examples)} samples)...[/cyan]")
            all_examples = builder.populate_retrieval_matches(all_examples, top_k=args.top_k)

            splits = builder.temporal_split(all_examples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
            builder.save_splits(splits, splits_dir)
            console.print(f"[green]Saved train/val/test splits to {splits_dir}[/green]")
        else:
            console.print(f"[cyan]Loading existing dataset splits from {splits_dir}...[/cyan]")
            splits = builder.load_splits(splits_dir)

        print_risk_dataset_report(splits)

        # 2. Train and Evaluate Models
        evaluator = RiskEvaluator()

        if args.model in ("baseline", "compare"):
            console.print(f"[bold cyan]Training Model A: Classical Baseline ({args.baseline_type})...[/bold cyan]")
            baseline = BaselineRiskModel(model_type=args.baseline_type)
            baseline.fit_examples(splits.train)
            if args.calibrate:
                opt_thresh = baseline.calibrate_threshold(splits.val, criterion="f1")
                console.print(f"[green]Calibrated optimal threshold on validation set: {opt_thresh:.3f}[/green]")
            baseline.save(model_save_path)
            console.print(f"[green]Model saved to {model_save_path}[/green]")

            effective_threshold = (
                baseline.optimal_threshold if args.calibrate else args.decision_threshold
            )
            console.print(f"[cyan]Evaluating Model A on TEST split (threshold: {effective_threshold})...[/cyan]")
            test_metrics = evaluator.evaluate_model(
                baseline,
                splits.test,
                split_name="test",
                decision_threshold=effective_threshold,
            )
            print_risk_metrics_report(test_metrics)

        if args.model in ("llm", "compare"):
            console.print(f"[bold cyan]Evaluating Model B: Qwen2.5-Coder (Offline / Mock LLM Mode)...[/bold cyan]")
            mock_llm = MockLLMScorer()
            y_true = []
            y_prob = []
            for ex in splits.test:
                pred = mock_llm.predict(
                    code=ex.code,
                    function_name=ex.function_name,
                    file_path=ex.file_path,
                    retrieved_matches=ex.retrieved_matches,
                    language=ex.language,
                )
                y_true.append(ex.label)
                y_prob.append(pred.risk_probability)

            llm_metrics = evaluator.evaluate_predictions(
                y_true=np.array(y_true, dtype=np.int32),
                y_prob=np.array(y_prob, dtype=np.float32),
                model_name="Qwen2.5-Coder (Simulated)",
                split_name="test",
                decision_threshold=args.decision_threshold,
            )
            print_risk_metrics_report(llm_metrics)

        # 3. Predict on a specific function if requested
        target_fn = args.function
        if target_fn:
            console.print(f"[bold cyan]Running Risk Prediction on Function: '{target_fn}'...[/bold cyan]")
            # Query code lookup
            fn_code = None
            fn_file = "src/flask/app.py"
            if pairs_file.exists():
                with open(pairs_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            p = BugFixPair.model_validate_json(line)
                            if target_fn.lower() in p.function_name.lower():
                                fn_code = p.buggy_code
                                fn_file = p.file_path
                                target_fn = p.function_name
                                break
            if not fn_code:
                fn_code = "def run(self, host=None, port=None, debug=None, **options):\n    pass"

            matches = retriever.retrieve(
                query=RetrievalQuery(
                    code=fn_code,
                    repository=repo_url,
                    function_name=target_fn,
                    file_path=fn_file,
                ),
                top_k=3,
            )

            # Use baseline or LLM
            if args.model == "llm":
                scorer = MockLLMScorer()
                pred = scorer.predict(
                    code=fn_code,
                    function_name=target_fn,
                    file_path=fn_file,
                    retrieved_matches=matches,
                )
            else:
                baseline = BaselineRiskModel.load(model_save_path)
                pred = baseline.predict_function(
                    code=fn_code,
                    function_name=target_fn,
                    file_path=fn_file,
                    retrieved_matches=matches,
                )

            print_risk_prediction_report(pred)

    # MODE 7, 8, 9, 10: Explainability, Fix Generation, PR Dispatch, End-to-End Scan
    if mode in ("explain", "fix", "pr", "scan"):
        target_fn = args.function or "Flask.run"
        console.print(f"[bold cyan]CodeGhost Autonomous Workflow ({mode.upper()}) for Function: '{target_fn}'...[/bold cyan]")

        # 1. Resolve function code and file
        fn_code = None
        fn_file = "src/flask/app.py"
        if pairs_file.exists():
            with open(pairs_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p = BugFixPair.model_validate_json(line)
                        if target_fn.lower() in p.function_name.lower():
                            fn_code = p.buggy_code
                            fn_file = p.file_path
                            target_fn = p.function_name
                            break
        if not fn_code:
            fn_code = (
                "def run(self, host=None, port=None, debug=None, **options):\n"
                "    server_name = options.get('SERVER_NAME')\n"
                "    if server_name:\n"
                "        host, _, port = server_name.partition(':')\n"
                "        port = int(port)\n"
                "    return serve(host, port, **options)\n"
            )

        # 2. Retrieve historical context
        v_store = QdrantVectorStore()
        embedder = FastEmbedEmbedder()
        retriever = FunctionRetriever(
            vector_store=v_store,
            embedder=embedder,
            collection_name=args.collection,
        )
        matches = retriever.retrieve(
            query=RetrievalQuery(
                code=fn_code,
                repository=repo_url,
                function_name=target_fn,
                file_path=fn_file,
            ),
            top_k=args.top_k,
        )

        # 3. Predict risk
        model_save_path = Path(args.model_save_path).resolve()
        if model_save_path.exists() and args.model != "llm":
            baseline = BaselineRiskModel.load(model_save_path)
            pred = baseline.predict_function(
                code=fn_code,
                function_name=target_fn,
                file_path=fn_file,
                retrieved_matches=matches,
            )
        else:
            scorer = MockLLMScorer()
            pred = scorer.predict(
                code=fn_code,
                function_name=target_fn,
                file_path=fn_file,
                retrieved_matches=matches,
            )

        print_risk_prediction_report(pred)

        # 4. Generate Reasoning & Explanation (Milestone 5)
        reasoning_engine = ReasoningEngine(provider=args.provider)
        explanation = reasoning_engine.explain(
            prediction=pred,
            code=fn_code,
            file_path=fn_file,
            language="python",
        )
        print_explanation_report(explanation)

        # 5. Automated Fix Generation & Sandbox Execution (Milestone 6)
        verified_fixes = []
        if mode in ("fix", "pr", "scan"):
            fix_gen = FixGenerator(provider=args.provider)
            candidates = fix_gen.generate_fixes(
                function_name=target_fn,
                code=fn_code,
                file_path=fn_file,
                explanation=explanation,
            )

            sandbox = TestSandbox()
            verified_fixes = [sandbox.verify_fix(c) for c in candidates]
            print_fixes_report(verified_fixes)

        # 6. GitHub PR Creation (Milestone 7)
        if mode in ("pr", "scan"):
            if not verified_fixes:
                console.print("[yellow]No viable fix candidates available to package for PR.[/yellow]")
            else:
                top_fix = verified_fixes[0]
                creator = PRCreator()
                pr_result = creator.create_pr(
                    verified_fix=top_fix,
                    explanation=explanation,
                    repo_url_or_path=repo_url or "https://github.com/pallets/flask",
                    base_branch=args.base_branch,
                    dry_run=args.dry_run,
                    github_token=args.github_token,
                )
                print_pr_report(pr_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
