"""Converts RiskExample dataset splits into chat instruction-tuning datasets for Qwen2.5-Coder."""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.risk.baseline_model import map_probability_to_score
from backend.risk.llm_scorer import MockLLMScorer
from backend.risk.schemas import RiskCategory, RiskExample, RiskThresholds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_example_to_chat(
    example: RiskExample,
    scorer_helper: MockLLMScorer,
    thresholds: RiskThresholds,
) -> Dict[str, Any]:
    """Format a single RiskExample into an OpenAI/Qwen chat messages dictionary."""
    prompt = scorer_helper.build_prompt(
        code=example.code,
        function_name=example.function_name,
        file_path=example.file_path,
        retrieved_matches=example.retrieved_matches,
        language=example.language,
    )

    # Determine ground-truth target probability and score
    # For positives (label=1), probability is high (e.g. 0.85 - 0.95)
    # For negatives (label=0), probability is low (e.g. 0.05 - 0.15)
    if example.label == 1:
        prob = 0.88
        summary = f"Function has historical vulnerabilities matching prior fixes in commit {example.commit_sha[:8]}."
    else:
        prob = 0.12
        summary = "Function demonstrates stable control flow and no matches to historical regression patterns."

    score = map_probability_to_score(prob)
    category = thresholds.categorize(score).value

    completion_json = json.dumps({
        "risk_probability": prob,
        "risk_score": score,
        "classification": category,
        "reasoning_summary": summary,
    }, indent=2)

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are CodeGhost Risk Evaluator. Analyze code and historical context to predict risk.",
            },
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": completion_json,
            },
        ]
    }


def prepare_chat_dataset(
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, int]:
    """Convert train, val, and test splits into SFT format."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scorer = MockLLMScorer()
    thresholds = RiskThresholds()
    stats: Dict[str, int] = {}

    for split_name in ("train", "val", "test"):
        in_file = input_dir / f"{split_name}.jsonl"
        out_file = output_dir / f"sft_{split_name}.jsonl"

        if not in_file.exists():
            logger.warning("Split file %s does not exist, skipping.", in_file)
            continue

        count = 0
        with open(in_file, "r", encoding="utf-8") as f_in, open(out_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                ex = RiskExample.model_validate_json(line.strip())
                chat_entry = convert_example_to_chat(ex, scorer, thresholds)
                f_out.write(json.dumps(chat_entry) + "\n")
                count += 1

        stats[split_name] = count
        logger.info("Generated %s with %d chat examples.", out_file, count)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare chat instruction dataset for Qwen LoRA training")
    parser.add_argument("--input-dir", default="backend/data/processed/risk_splits", help="Input directory containing risk splits")
    parser.add_argument("--output-dir", default="training/data", help="Output directory for formatted SFT datasets")
    args = parser.parse_args()

    stats = prepare_chat_dataset(Path(args.input_dir), Path(args.output_dir))
    print(f"Dataset preparation complete: {stats}")


if __name__ == "__main__":
    main()
