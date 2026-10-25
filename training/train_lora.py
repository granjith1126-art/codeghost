"""Fine-tuning pipeline for Qwen2.5-Coder using LoRA / QLoRA with optional W&B tracking.

Compatible with Google Colab / Kaggle (GPU) and includes a --dry-run mode for CPU / testing.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import yaml

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_yaml(file_path: Path) -> dict:
    """Load configuration dictionary from YAML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_training(
    train_config_path: Path,
    lora_config_path: Path,
    dry_run: bool = False,
    override_model_id: str = None,
    override_epochs: int = None,
):
    """Execute LoRA / QLoRA training run."""
    train_cfg = load_yaml(train_config_path)
    lora_cfg = load_yaml(lora_config_path)

    base_model_id = override_model_id or train_cfg.get("base_model_id", "Qwen/Qwen2.5-Coder-7B-Instruct")
    num_epochs = override_epochs or train_cfg.get("num_train_epochs", 3)
    dataset_dir = Path(train_cfg.get("dataset_dir", "backend/data/processed/risk_splits"))
    output_dir = Path(train_cfg.get("output_dir", "training/checkpoints/qwen_lora"))

    logger.info("=== CodeGhost Qwen LoRA Training Configuration ===")
    logger.info("Base Model: %s", base_model_id)
    logger.info("Dataset Directory: %s", dataset_dir)
    logger.info("Output Checkpoints: %s", output_dir)
    logger.info("Epochs: %d, LR: %s, LoRA r: %s", num_epochs, train_cfg.get("learning_rate"), lora_cfg.get("r"))
    logger.info("Dry run mode: %s", dry_run)

    # Validate dataset existence
    sft_train = dataset_dir / "sft_train.jsonl"
    sft_val = dataset_dir / "sft_val.jsonl"
    alt_train = Path("training/data/sft_train.jsonl")
    alt_val = Path("training/data/sft_val.jsonl")

    train_file = sft_train if sft_train.exists() else alt_train
    val_file = sft_val if sft_val.exists() else alt_val

    if dry_run:
        logger.info("[Dry Run] Validating dataset paths and configurations...")
        if not train_file.exists():
            logger.warning("[Dry Run] SFT train file not found at %s or %s (run prepare_dataset.py first).", sft_train, alt_train)
        else:
            with open(train_file, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            logger.info("[Dry Run] SFT train file validated (%d samples).", lines)
        logger.info("[Dry Run] Configuration and pipeline syntax validated successfully. Exiting dry run.")
        return 0

    # Real training requires torch, transformers, and peft
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
        from trl import SFTTrainer
    except ImportError as e:
        logger.error("Missing deep learning dependencies for full training: %s", e)
        logger.error("To run full training, use a GPU environment (e.g. Google Colab / Kaggle) or install torch, transformers, peft, trl, bitsandbytes.")
        return 1

    # Optional Weights & Biases
    use_wandb = train_cfg.get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=train_cfg.get("wandb_project", "codeghost-risk-scoring"),
                config={**train_cfg, **lora_cfg},
            )
            logger.info("W&B tracking initialized.")
        except Exception as e:
            logger.warning("Could not initialize W&B (%s). Continuing with local logging.", e)

    # Setup quantization config if CUDA is available
    bnb_config = None
    if train_cfg.get("load_in_4bit", True) and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if train_cfg.get("bf16", False) else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer and base model
    logger.info("Loading tokenizer and base model from '%s'...", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Setup LoRA
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load datasets
    dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(val_file)})

    # SFT Trainer arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        eval_strategy="steps",
        eval_steps=int(train_cfg.get("eval_steps", 20)),
        save_steps=int(train_cfg.get("save_steps", 50)),
        save_total_limit=2,
        seed=int(train_cfg.get("seed", 42)),
        fp16=not train_cfg.get("bf16", False) and torch.cuda.is_available(),
        bf16=train_cfg.get("bf16", False) and torch.cuda.is_available(),
        report_to="wandb" if use_wandb else "none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_text_field="messages",
        max_seq_length=int(train_cfg.get("max_seq_length", 2048)),
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    # Save final model
    logger.info("Saving fine-tuned adapter to %s...", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Training complete!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Coder with LoRA / QLoRA")
    parser.add_argument("--train-config", default="training/configs/train_config.yaml", help="Path to training config YAML")
    parser.add_argument("--lora-config", default="training/configs/lora_config.yaml", help="Path to LoRA config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and datasets without starting training")
    parser.add_argument("--model-id", default=None, help="Override base model ID")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()

    return run_training(
        train_config_path=Path(args.train_config),
        lora_config_path=Path(args.lora_config),
        dry_run=args.dry_run,
        override_model_id=args.model_id,
        override_epochs=args.epochs,
    )


if __name__ == "__main__":
    sys.exit(main())
