"""Tests for the heuristic BugDetector."""

import pytest
from backend.history.bug_detector import BugDetector
from backend.models.schemas import BugFixClassification


@pytest.fixture
def detector():
    return BugDetector(candidate_threshold=0.60, uncertain_threshold=0.30)


def test_strong_bug_fix_messages(detector: BugDetector):
    examples = [
        "Fix crash when receiving empty request payload",
        "Fixed critical bug in auth session validation",
        "Bugfix for regression in url routing logic",
        "Resolves #1234: prevent deadlock during teardown",
        "Patched security defect in cookie deserializer",
    ]
    for msg in examples:
        evidence = detector.analyze(msg)
        assert evidence.classification == BugFixClassification.BUG_FIX_CANDIDATE, f"Failed for: {msg}"
        assert evidence.score >= 0.60
        assert len(evidence.matched_signals) > 0


def test_non_bug_fix_messages(detector: BugDetector):
    examples = [
        "docs: update installation guide and contributing docs",
        "feat: add new CLI command for exporting logs",
        "chore: bump version to 2.1.0 and update dependencies",
        "style: reformat according to PEP8 guidelines",
        "Merge branch 'main' of github.com:org/repo",
    ]
    for msg in examples:
        evidence = detector.analyze(msg)
        assert evidence.classification == BugFixClassification.NON_BUG_FIX, f"Failed for: {msg}"
        assert evidence.score < 0.60


def test_uncertain_messages(detector: BugDetector):
    # Contextual keywords without strong 'fix' keyword
    examples = [
        "handle unexpected error status code in response",
        "avoid potential leak when closing sockets",
        "workaround for broken third-party library behavior",
    ]
    for msg in examples:
        evidence = detector.analyze(msg)
        assert evidence.classification in (
            BugFixClassification.UNCERTAIN,
            BugFixClassification.BUG_FIX_CANDIDATE,
        ), f"Failed for: {msg}"
        assert evidence.score >= 0.25


def test_empty_and_whitespace_messages(detector: BugDetector):
    for empty in ["", "   ", "\n\t\n"]:
        evidence = detector.analyze(empty)
        assert evidence.classification == BugFixClassification.NON_BUG_FIX
        assert evidence.score == 0.0
        assert "empty_commit_message" in evidence.matched_signals
