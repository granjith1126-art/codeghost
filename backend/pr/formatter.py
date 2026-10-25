"""Pull Request Markdown Formatter for CodeGhost (Milestone 7).

Assembles clear, informative GitHub PR descriptions linking historical bug
precedents, AST complexity drivers, verified sandbox results, and human review checklists.
"""

import hashlib
import os
import re
from typing import Optional

from backend.autofix.schemas import VerifiedFix
from backend.pr.schemas import PRDraft
from backend.reasoning.schemas import RiskExplanation


class PRFormatter:
    """Formats rich GitHub Pull Request payloads from verified fixes and risk explanations."""

    def format_draft(
        self,
        verified_fix: VerifiedFix,
        explanation: RiskExplanation,
        repo_identifier: str = "repository",
        base_branch: str = "main",
    ) -> PRDraft:
        """Construct a complete PRDraft model.

        Args:
            verified_fix: VerifiedFix containing fix candidate and sandbox results.
            explanation: RiskExplanation from Milestone 5.
            repo_identifier: Repository name or URL.
            base_branch: Base branch to merge into.

        Returns:
            PRDraft with title, branch, conventional commit message, and markdown body.
        """
        cand = verified_fix.candidate
        clean_func = re.sub(r"[^a-zA-Z0-9_]", "_", cand.function_name.lower())
        short_hash = hashlib.sha256(cand.fixed_code.encode("utf-8")).hexdigest()[:8]
        branch_name = f"codeghost/fix-{clean_func}-{short_hash}"

        cat_str = getattr(explanation.risk_category, "value", str(explanation.risk_category))
        file_base = os.path.basename(cand.file_path).replace(".py", "")
        title = f"fix({file_base}): mitigate {cat_str.lower()} risk in `{cand.function_name}`"
        commit_message = (
            f"fix({file_base}): resolve {cand.strategy.value if hasattr(cand.strategy, 'value') else str(cand.strategy)} vulnerability in {cand.function_name}\n\n"
            f"Automated fix generated and verified by CodeGhost.\n"
            f"Risk Score: {explanation.risk_score:.1f}/10.0 ({cat_str})"
        )

        body_markdown = self._build_body(verified_fix, explanation)

        return PRDraft(
            title=title,
            branch_name=branch_name,
            commit_message=commit_message,
            body_markdown=body_markdown,
            diff=cand.diff,
            target_repo=repo_identifier,
            base_branch=base_branch,
            risk_score=explanation.risk_score,
            risk_category=cat_str,
            is_verified=verified_fix.is_verified,
            metadata={
                "fix_id": cand.fix_id,
                "strategy": cand.strategy.value if hasattr(cand.strategy, "value") else str(cand.strategy),
                "confidence": cand.confidence,
            },
        )

    def _build_body(self, verified_fix: VerifiedFix, explanation: RiskExplanation) -> str:
        """Construct the full markdown PR description."""
        cand = verified_fix.candidate
        sb = verified_fix.sandbox_result
        cat_str = getattr(explanation.risk_category, "value", str(explanation.risk_category))
        strat_str = cand.strategy.value if hasattr(cand.strategy, "value") else str(cand.strategy)

        # Section 1: Executive Summary & Risk
        badge_color = "red" if explanation.risk_score >= 8.5 else "orange" if explanation.risk_score >= 7.0 else "yellow"
        body = [
            f"# 🛡️ CodeGhost Automated Fix: `{cand.function_name}`",
            "",
            f"> **Risk Assessment**: **{explanation.risk_score:.1f} / 10.0** (`{cat_str}`)  ",
            f"> **Target File**: `{cand.file_path}`  ",
            f"> **Fix Strategy**: `{strat_str}` (Confidence: {cand.confidence:.0%})  ",
            f"> **Sandbox Status**: {'✅ **VERIFIED (All Tests Passed)**' if verified_fix.is_verified else '⚠️ **UNVERIFIED (Review Required)**'}",
            "",
            "## 📌 Executive Summary",
            explanation.summary,
            "",
            "## 🔍 Risk & Structural Analysis",
            explanation.detailed_analysis,
            "",
        ]

        # Section 2: Complexity Drivers
        if explanation.complexity_factors:
            body.append("### AST Complexity Drivers")
            body.append("| Metric | Observed Value | Risk Impact | Diagnostic Description |")
            body.append("| :--- | :--- | :--- | :--- |")
            for f in explanation.complexity_factors:
                body.append(f"| **{f.name}** | `{f.value}` | `{f.risk_impact}` | {f.description} |")
            body.append("")

        # Section 3: Historical Bug Precedents
        if explanation.historical_precedents:
            body.append("## 🧬 Historical Repository Precedents")
            body.append(
                "CodeGhost identified the following historically fixed defects in this repository "
                "sharing similar syntax or failure patterns:"
            )
            body.append("")
            for idx, p in enumerate(explanation.historical_precedents[:3], 1):
                body.append(f"#### {idx}. Commit [`{p.commit_sha[:8]}`] — *\"{p.commit_message}\"*")
                body.append(f"- **Similarity Alignment**: `{p.similarity_score:.1%}`")
                body.append(f"- **Relevance**: {p.relevance_reason}")
                if p.diff_summary:
                    body.append(f"```diff\n{p.diff_summary[:300]}\n```")
                body.append("")

        # Section 4: Proposed Fix & Diff
        body.extend([
            "## 🛠️ Proposed Solution & Diff",
            f"{cand.description}",
            "",
            "```diff",
            cand.diff,
            "```",
            "",
        ])

        # Section 5: Sandbox Verification
        body.append("## 🧪 Isolated Sandbox Verification")
        if sb:
            body.append(f"- **Outcome**: {'✅ Tests Passed' if sb.success else '❌ Tests Failed'}")
            body.append(f"- **Total Tests Executed**: `{sb.tests_run}`")
            body.append(f"- **Tests Passed**: `{sb.tests_passed}`")
            body.append(f"- **Tests Failed / Errored**: `{sb.tests_failed}` failed, `{sb.tests_errored}` errored")
            body.append(f"- **Duration**: `{sb.duration_seconds:.2f}s`")
            if sb.stdout:
                body.append(f"\n<details><summary>Sandbox Test Output Log</summary>\n\n```text\n{sb.stdout[:1500]}\n```\n</details>\n")
        else:
            body.append("- **Outcome**: Static AST syntax verified without regressions.")
        body.append("")

        # Section 6: Human-in-the-Loop Review Checklist
        body.extend([
            "## 📋 Human Reviewer Checklist",
            "- [ ] **Intent Check**: Does the proposed fix preserve the original intended behavior?",
            "- [ ] **Boundary Conditions**: Are edge cases, null arguments, and empty collections handled appropriately?",
            "- [ ] **API Compatibility**: Does this change maintain backwards compatibility with existing callers?",
            "- [ ] **Test Coverage**: Does the repository have sufficient automated tests for this path?",
            "",
            "---",
            "*Automated pull request draft prepared by [CodeGhost](https://github.com/granjith1126-art/codeghost).*",
        ])

        return "\n".join(body)
