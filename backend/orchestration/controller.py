"""Main Orchestration Controller and Command-Line Interface for CodeGhost.

Provides automated orchestration, task planning, phase execution,
resumption, and reporting across Milestone 8 Phases 2 through 10.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.orchestration.config import OrchestrationConfig
from backend.orchestration.executor import LocalDirectExecutor
from backend.orchestration.logging_utils import get_orchestration_logger
from backend.orchestration.phase_runner import PhaseRunner
from backend.orchestration.planner import OpenAIPlanner
from backend.orchestration.reporter import OrchestrationReporter
from backend.orchestration.state_store import StateStore
from backend.orchestration.verification import TaskVerifier


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CodeGhost Autonomous AI Orchestrator")
    parser.add_argument("--status", action="store_true", help="Display current workflow state and phase statuses")
    parser.add_argument("--milestone", default="8", help="Milestone to execute (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan without mutating disk or running paid APIs")
    parser.add_argument("--resume", action="store_true", help="Resume workflow from the earliest incomplete phase")
    parser.add_argument("--phase", default=None, help="Execute a specific phase (e.g. 'phase_2')")
    parser.add_argument("--report", action="store_true", help="Display summary of completed benchmark reports")

    args = parser.parse_args(argv)

    config = OrchestrationConfig()
    if str(args.milestone) == "9":
        state_file = config.reports_dir / "orchestration_state_m9.json"
    else:
        state_file = config.state_file

    logger = get_orchestration_logger(config=config)
    state_store = StateStore(state_file)
    reporter = OrchestrationReporter(state_store)
    executor = LocalDirectExecutor(config)
    verifier = TaskVerifier(config, executor)
    planner = OpenAIPlanner(config, state_store)
    runner = PhaseRunner(config, executor, planner, state_store, verifier)

    # 1. Status Command
    if args.status:
        reporter.print_status()
        return 0

    # 2. Report Command
    if args.report:
        rep_name = "milestone_9_report.md" if str(args.milestone) == "9" else "milestone_8_final_report.md"
        final_rep = config.reports_dir / rep_name
        if final_rep.exists():
            with open(final_rep, "r", encoding="utf-8") as f:
                print(f.read())
        else:
            print(f"Report not yet generated: {rep_name}. Run --milestone {args.milestone} first.")
        return 0

    if str(args.milestone) == "9":
        all_milestone_phases = ["phase_9_1", "phase_9_2", "phase_9_3", "phase_9_4"]
    else:
        all_milestone_phases = [f"phase_{i}" for i in range(2, 11)]

    # 3. Dry-Run Check
    if args.dry_run:
        print("[DRY-RUN] Validating configuration, paths, and task planner...")
        print(f"OpenAI Key Configured: {config.has_api_key} ({config.masked_api_key})")
        print(f"OpenAI Model: {config.openai_model}")
        print(f"Project Root: {config.project_root}")
        
        target_phases = [args.phase] if args.phase else all_milestone_phases
        for pid in target_phases:
            plan = planner.plan_phase(pid, f"Dry-run plan for {pid}")
            print(f"\nPlan for {pid} ({len(plan)} tasks):")
            for t in plan:
                print(f"  - [{t.task_id}] {t.description}")
        return 0

    # 4. Phase or Sequential Execution
    if args.phase:
        phases_to_run = [args.phase]
    elif args.resume:
        next_p = state_store.get_next_incomplete_phase()
        if not next_p:
            print("All phases are already complete! Run --status to inspect results.")
            return 0
        if next_p in all_milestone_phases:
            start_idx = all_milestone_phases.index(next_p)
            phases_to_run = all_milestone_phases[start_idx:]
        else:
            phases_to_run = [next_p]
    else:
        phases_to_run = all_milestone_phases

    print(f"Starting CodeGhost Autonomous Orchestrator: Milestone {args.milestone}")
    print(f"Executing Phases: {', '.join(phases_to_run)}")
    
    for pid in phases_to_run:
        print(f"\n>>> Running {pid.upper()}...")
        success = runner.run_phase(pid, dry_run=False)
        if not success:
            print(f"\n[ERROR] {pid} encountered a failure. Halting autonomous progression.")
            reporter.print_status()
            return 1
        print(f">>> {pid.upper()} COMPLETED SUCCESSFULLY.")

    if not state_store.get_next_incomplete_phase():
        state_store.state.status = "COMPLETED"
        state_store.save()

    reporter.print_status()
    print("\n[SUCCESS] Milestone 8 Autonomous Validation Completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
