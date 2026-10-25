"""Evaluation script comparing the Classical ML Baseline vs Fine-Tuned Qwen2.5-Coder."""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.risk.baseline_model import BaselineRiskModel
from backend.risk.dataset_builder import RiskDatasetBuilder
from backend.risk.evaluator import RiskEvaluator
from backend.risk.llm_scorer import MockLLMScorer, QwenLLMScorer
from backend.risk.schemas import RiskEvaluationMetrics, RiskExample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_llm_on_examples(
    scorer,
    test_examples: List[RiskExample],
    evaluator: RiskEvaluator,
    model_name: str = "Qwen2.5-Coder-7B",
) -> RiskEvaluationMetrics:
    """Run inference with LLM scorer over test examples and compute metrics."""
    y_true = []
    y_prob = []

    for ex in test_examples:
        pred = scorer.predict(
            code=ex.code,
            function_name=ex.function_name,
            file_path=ex.file_path,
            retrieved_matches=ex.retrieved_matches,
            language=ex.language,
            model_name=model_name,
        )
        y_true.append(ex.label)
        y_prob.append(pred.risk_probability)

    return evaluator.evaluate_predictions(
        y_true=np.array(y_true, dtype=np.int32),
        y_prob=np.array(y_prob, dtype=np.float32),
        model_name=model_name,
        split_name="test",
    )


def compare_models(
    baseline_model_path: Path,
    test_examples: List[RiskExample],
    adapter_path: Optional[Path] = None,
    use_mock_llm: bool = True,
) -> Dict[str, RiskEvaluationMetrics]:
    """Evaluate both Baseline and LLM on the identical test split."""
    evaluator = RiskEvaluator()

    # 1. Baseline Evaluation
    logger.info("Evaluating Classical ML Baseline on test split (%d samples)...", len(test_examples))
    baseline = BaselineRiskModel.load(baseline_model_path)
    baseline_metrics = evaluator.evaluate_model(baseline, test_examples, split_name="test")

    # 2. LLM Evaluation
    if use_mock_llm or adapter_path is None or not Path(adapter_path).exists():
        logger.info("Evaluating LLM using MockLLMScorer (Offline / CPU mode)...")
        llm_scorer = MockLLMScorer()
        llm_name = "Qwen2.5-Coder (Simulated)"
    else:
        logger.info("Evaluating fine-tuned Qwen model with adapter '%s'...", adapter_path)
        llm_scorer = QwenLLMScorer(adapter_path=str(adapter_path))
        llm_name = "Qwen2.5-Coder (Fine-Tuned)"

    llm_metrics = evaluate_llm_on_examples(llm_scorer, test_examples, evaluator, model_name=llm_name)

    return {
        "baseline": baseline_metrics,
        "llm": llm_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare Classical Baseline vs Qwen LLM")
    parser.add_argument("--baseline-model", default="backend/data/models/baseline_rf.joblib", help="Path to saved baseline model")
    parser.add_argument("--splits-dir", default="backend/data/processed/risk_splits", help="Directory containing test.jsonl")
    parser.add_argument("--adapter-path", default="training/checkpoints/qwen_lora", help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--use-mock", action="store_true", default=True, help="Use mock LLM for testing without GPU")
    args = parser.parse_args()

    builder = RiskDatasetBuilder()
    splits = builder.load_splits(args.splits_dir)

    if not splits.test:
        print(f"No test examples found in {args.splits_dir}")
        return 1

    results = compare_models(
        baseline_model_path=Path(args.baseline_model),
        test_examples=splits.test,
        adapter_path=Path(args.adapter_path),
        use_mock_llm=args.use_mock,
    )

    print("\n=================== MODEL COMPARISON REPORT ===================")
    print(f"{'Model':<30} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'PR-AUC':<10} | {'ROC-AUC':<10}")
    print("-" * 90)
    for key, m in results.items():
        pr_str = f"{m.pr_auc:.4f}" if m.pr_auc is not None else "N/A"
        roc_str = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "N/A"
        print(f"{m.model_name:<30} | {m.precision:<10.4f} | {m.recall:<10.4f} | {m.f1:<10.4f} | {pr_str:<10} | {roc_str:<10}")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
