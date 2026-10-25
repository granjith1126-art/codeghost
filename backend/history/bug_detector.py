"""Heuristic Bug-Fix Candidate Detector.

Analyzes commit messages using weighted lexical patterns, keyword signals,
and context boundaries to classify commits into:
- BUG_FIX_CANDIDATE
- UNCERTAIN
- NON_BUG_FIX

NOTE: This classification is a heuristic filter and NOT ground truth.
All evidence and matched signals are preserved.
"""

import logging
import re
from typing import List, Tuple

from backend.models.schemas import BugDetectionEvidence, BugFixClassification

logger = logging.getLogger(__name__)

# Strong bug fix signals (weight: +0.6 to +0.8)
STRONG_BUG_PATTERNS: List[Tuple[str, float, str]] = [
    (r"\b(?:fix|fixes|fixed|fixing|hotfix|bugfix)\b", 0.70, "fix_keyword"),
    (r"\b(?:bug|bugs)\b", 0.65, "bug_keyword"),
    (r"\b(?:crash|crashes|crashed|crashing|segfault)\b", 0.75, "crash_keyword"),
    (r"\b(?:regression|regressed)\b", 0.75, "regression_keyword"),
    (r"\b(?:defect|defects)\b", 0.65, "defect_keyword"),
    (r"\b(?:patch|patches|patched)\b", 0.55, "patch_keyword"),
    (r"\b(?:resolve|resolves|resolved|resolving)\b", 0.55, "resolve_keyword"),
    (r"\b(?:fixes|closes|resolves)\s*#\d+\b", 0.85, "fixes_issue_ref"),
    (r"\bgh-\d+\b", 0.50, "gh_issue_ref"),
]

# Secondary / Contextual signals (weight: +0.2 to +0.4)
CONTEXT_BUG_PATTERNS: List[Tuple[str, float, str]] = [
    (r"\b(?:incorrect|wrong|invalid|unexpected)\b", 0.35, "incorrect_behavior"),
    (r"\b(?:broken|break|breaks)\b", 0.35, "broken_keyword"),
    (r"\b(?:error|errors)\b", 0.30, "error_keyword"),
    (r"\b(?:failure|fail|fails|failed)\b", 0.30, "failure_keyword"),
    (r"\b(?:leak|leaks|leaked|overflow)\b", 0.40, "leak_keyword"),
    (r"\b(?:infinite loop|deadlock)\b", 0.50, "deadlock_keyword"),
    (r"\b(?:workaround|typo|handle null|handle none|off-by-one)\b", 0.35, "workaround_or_typo"),
    (r"\b(?:prevent|prevents|avoid|avoids)\b", 0.25, "prevention_keyword"),
]

# Negative signals indicating features, docs, chores, refactors, or releases
NEGATIVE_PATTERNS: List[Tuple[str, float, str]] = [
    (r"^(?:feat|feature)(?:\([^)]+\))?:", -0.40, "feat_prefix"),
    (r"^(?:docs?|documentation)(?:\([^)]+\))?:", -0.50, "docs_prefix"),
    (r"^(?:chore|ci|build|style|test|tests)(?:\([^)]+\))?:", -0.45, "chore_prefix"),
    (r"^(?:refactor)(?:\([^)]+\))?:", -0.30, "refactor_prefix"),
    (r"\b(?:merge\s+branch|merge\s+pull\s+request)\b", -0.35, "merge_message"),
    (r"\b(?:bump\s+version|release\s+\d+\.\d+|version\s+\d+\.\d+)\b", -0.50, "release_version"),
    (r"\b(?:add|adds|added|adding)\s+new\s+(?:feature|support|module)\b", -0.30, "new_feature"),
]


class BugDetector:
    """Classifies commit messages to identify potential bug-fix candidates."""

    def __init__(
        self,
        candidate_threshold: float = 0.60,
        uncertain_threshold: float = 0.30,
    ):
        self.candidate_threshold = candidate_threshold
        self.uncertain_threshold = uncertain_threshold

        # Precompile regexes
        self._strong = [
            (re.compile(pat, re.IGNORECASE | re.MULTILINE), weight, label)
            for pat, weight, label in STRONG_BUG_PATTERNS
        ]
        self._context = [
            (re.compile(pat, re.IGNORECASE | re.MULTILINE), weight, label)
            for pat, weight, label in CONTEXT_BUG_PATTERNS
        ]
        self._negative = [
            (re.compile(pat, re.IGNORECASE | re.MULTILINE), weight, label)
            for pat, weight, label in NEGATIVE_PATTERNS
        ]

    def analyze(self, commit_message: str) -> BugDetectionEvidence:
        """
        Analyze a commit message and produce evidence and classification.

        Args:
            commit_message: The full commit message.

        Returns:
            BugDetectionEvidence with score, classification, and matched signals.
        """
        if not commit_message or not commit_message.strip():
            return BugDetectionEvidence(
                score=0.0,
                classification=BugFixClassification.NON_BUG_FIX,
                matched_signals=["empty_commit_message"],
                explanation="Commit message is empty or whitespace only",
            )

        text = commit_message.strip()
        matched_signals: List[str] = []
        raw_score = 0.0

        # Check strong signals
        for regex, weight, label in self._strong:
            match = regex.search(text)
            if match:
                matched_signals.append(f"{label}:'{match.group(0)}'")
                raw_score += weight

        # Check contextual signals
        for regex, weight, label in self._context:
            match = regex.search(text)
            if match:
                matched_signals.append(f"{label}:'{match.group(0)}'")
                raw_score += weight

        # Check negative signals
        for regex, weight, label in self._negative:
            match = regex.search(text)
            if match:
                matched_signals.append(f"{label}:'{match.group(0)}'")
                raw_score += weight

        # Normalization: clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, round(raw_score, 3)))

        # Determine classification
        if score >= self.candidate_threshold:
            classification = BugFixClassification.BUG_FIX_CANDIDATE
        elif score >= self.uncertain_threshold:
            classification = BugFixClassification.UNCERTAIN
        else:
            classification = BugFixClassification.NON_BUG_FIX

        explanation = (
            f"Score: {score:.2f} (raw: {raw_score:.2f}) -> {classification.value} "
            f"based on {len(matched_signals)} matched signal(s)"
        )

        return BugDetectionEvidence(
            score=score,
            classification=classification,
            matched_signals=matched_signals,
            explanation=explanation,
        )
