"""Unit tests for RiskDatasetBuilder and temporal splitting."""

import pytest
from backend.models.schemas import BugFixPair
from backend.risk.dataset_builder import RiskDatasetBuilder, parse_iso_timestamp
from backend.risk.schemas import RiskExample


def test_parse_iso_timestamp():
    ts = "2026-08-11T19:48:40+00:00"
    epoch = parse_iso_timestamp(ts)
    assert epoch > 1700000000.0


def test_build_positive_examples(tmp_path):
    pairs_file = tmp_path / "pairs.jsonl"
    pair = BugFixPair(
        repository="https://github.com/pallets/flask",
        commit_sha="c111111111111111111111111111111111111111",
        parent_sha="c00000000000000000000000000000000000000",
        commit_message="Fix parsing of IPv6 address",
        author="Tester",
        timestamp="2026-08-10T12:00:00+00:00",
        file_path="src/flask/app.py",
        language="python",
        code_unit_type="method",
        class_name="Flask",
        function_name="Flask.run",
        buggy_code="def run(): host.partition(':')",
        fixed_code="def run(): urlsplit(host)",
        diff="--- a\n+++ b",
        detector_score=0.8,
        detector_signals=["fix_keyword"],
        detector_classification="BUG_FIX_CANDIDATE",
    )
    with open(pairs_file, "w", encoding="utf-8") as f:
        f.write(pair.model_dump_json() + "\n")

    builder = RiskDatasetBuilder()
    examples = builder.build_positive_examples(pairs_file)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.label == 1
    assert ex.function_name == "Flask.run"
    assert ex.code == "def run(): host.partition(':')"
    assert ex.commit_sha == pair.commit_sha


def test_temporal_split():
    builder = RiskDatasetBuilder()
    examples = []
    # Create 10 examples with strictly increasing timestamps
    for i in range(10):
        ex = RiskExample(
            example_id=f"ex_{i}",
            repository="repo",
            commit_sha=f"commit_{i:02d}",
            function_name=f"func_{i}",
            file_path=f"file_{i}.py",
            language="python",
            code="def foo(): pass",
            timestamp=f"2026-01-{i+1:02d}T00:00:00+00:00",
            timestamp_epoch=float(1000 + i * 100),
            label=1 if i % 2 == 0 else 0,
            label_rationale="test",
        )
        examples.append(ex)

    splits = builder.temporal_split(examples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    assert len(splits.train) + len(splits.val) + len(splits.test) == 10
    assert len(splits.train) >= 6
    assert len(splits.val) >= 1
    assert len(splits.test) >= 1

    # Verify temporal ordering: max train timestamp < min val timestamp < min test timestamp
    max_train_epoch = max(ex.timestamp_epoch for ex in splits.train)
    min_val_epoch = min(ex.timestamp_epoch for ex in splits.val)
    max_val_epoch = max(ex.timestamp_epoch for ex in splits.val)
    min_test_epoch = min(ex.timestamp_epoch for ex in splits.test)

    assert max_train_epoch <= min_val_epoch
    assert max_val_epoch <= min_test_epoch


def test_save_and_load_splits(tmp_path):
    builder = RiskDatasetBuilder()
    ex = RiskExample(
        example_id="ex_test",
        repository="repo",
        commit_sha="sha123",
        function_name="func",
        file_path="f.py",
        language="python",
        code="def func(): return 42",
        timestamp="2026-01-01T00:00:00+00:00",
        timestamp_epoch=1000.0,
        label=0,
        label_rationale="test",
    )
    splits = builder.temporal_split([ex, ex, ex, ex, ex, ex], train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    builder.save_splits(splits, tmp_path)

    loaded = builder.load_splits(tmp_path)
    assert len(loaded.train) == len(splits.train)
    assert len(loaded.val) == len(splits.val)
    assert len(loaded.test) == len(splits.test)
