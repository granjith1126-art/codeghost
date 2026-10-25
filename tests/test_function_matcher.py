"""Tests for FunctionMatcher: AST-aware diff mapping to functions and classes."""

import pytest
from backend.history.function_matcher import FunctionMatcher
from backend.models.schemas import (
    BugDetectionEvidence,
    BugFixClassification,
    CommitFileDiff,
    CommitRecord,
)


@pytest.fixture
def matcher():
    return FunctionMatcher()


def test_match_function_in_class(matcher: FunctionMatcher):
    parent_content = '''
class Calculator:
    """A math calculator."""
    def add(self, a: int, b: int) -> int:
        # Bug: subtracting instead of adding
        return a - b

    def multiply(self, a: int, b: int) -> int:
        return a * b
'''

    fix_content = '''
class Calculator:
    """A math calculator."""
    def add(self, a: int, b: int) -> int:
        # Fixed: correct addition
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b
'''

    commit = CommitRecord(
        repository="test_repo",
        commit_sha="commit_fix_123",
        parent_shas=["commit_buggy_000"],
        author_name="Alice",
        author_email="alice@example.com",
        timestamp="2026-01-01T12:00:00Z",
        commit_message="Fix bug in Calculator.add logic",
    )

    diff = CommitFileDiff(
        file_path="calc.py",
        change_type="M",
        additions=2,
        deletions=2,
        diff_text="@@ -5,2 +5,2 @@\n-# Bug: subtracting instead of adding\n-        return a - b\n+# Fixed: correct addition\n+        return a + b\n",
        old_line_ranges=[(5, 6)],
        new_line_ranges=[(5, 6)],
    )

    evidence = BugDetectionEvidence(
        score=0.75,
        classification=BugFixClassification.BUG_FIX_CANDIDATE,
        matched_signals=["fix_keyword:'Fix'", "bug_keyword:'bug'"],
        explanation="Score 0.75",
    )

    pairs = matcher.match_commit_changes(
        commit_record=commit,
        file_diff=diff,
        old_content=parent_content,
        new_content=fix_content,
        detection_evidence=evidence,
    )

    assert len(pairs) == 1
    pair = pairs[0]

    # Verify the smallest relevant unit was matched: Calculator.add (not Calculator or multiply)
    assert pair.function_name == "Calculator.add"
    assert pair.class_name == "Calculator"
    assert "return a - b" in pair.buggy_code
    assert "return a + b" in pair.fixed_code
    assert "+        return a + b" in pair.diff
    assert "-        return a - b" in pair.diff
    assert pair.detector_classification == "BUG_FIX_CANDIDATE"
    assert pair.additions >= 1
    assert pair.deletions >= 1


def test_match_standalone_function(matcher: FunctionMatcher):
    parent_content = '''
def process_data(items):
    if not items:
        return None
    return items[1]  # IndexError bug
'''

    fix_content = '''
def process_data(items):
    if not items:
        return None
    return items[0]  # Fixed
'''

    commit = CommitRecord(
        repository="test_repo",
        commit_sha="commit_fix_456",
        parent_shas=["commit_buggy_111"],
        author_name="Bob",
        author_email="bob@example.com",
        timestamp="2026-01-02T12:00:00Z",
        commit_message="Fix crash in process_data when items has 1 element",
    )

    diff = CommitFileDiff(
        file_path="processor.py",
        change_type="M",
        additions=1,
        deletions=1,
        diff_text="@@ -5 +5 @@\n-    return items[1]  # IndexError bug\n+    return items[0]  # Fixed\n",
        old_line_ranges=[(5, 5)],
        new_line_ranges=[(5, 5)],
    )

    evidence = BugDetectionEvidence(
        score=0.80,
        classification=BugFixClassification.BUG_FIX_CANDIDATE,
        matched_signals=["fix_keyword:'Fix'", "crash_keyword:'crash'"],
        explanation="Score 0.80",
    )

    pairs = matcher.match_commit_changes(
        commit_record=commit,
        file_diff=diff,
        old_content=parent_content,
        new_content=fix_content,
        detection_evidence=evidence,
    )

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.function_name == "process_data"
    assert pair.class_name is None
    assert "return items[1]" in pair.buggy_code
    assert "return items[0]" in pair.fixed_code
