# CodeGhost Qwen2.5-Coder LoRA Training Pipeline

This directory contains the standalone LoRA/QLoRA fine-tuning and evaluation pipeline for **Model B (Qwen2.5-Coder-7B)**.

---

## Directory Overview

```text
training/
├── prepare_dataset.py       # Converts RiskExample splits into chat instruction-tuning JSONL
├── train_lora.py            # SFTTrainer + 4-bit QLoRA script with optional W&B tracking
├── evaluate.py              # Evaluates fine-tuned adapter vs classical baseline
├── configs/
│   ├── lora_config.yaml     # LoRA hyperparameters (r=16, alpha=32, target_modules)
│   └── train_config.yaml    # Training hyperparameters (learning_rate, epochs, batch_size)
└── README.md                # Documentation & execution guide
```

---

## Quickstart (Local Dry-Run)

Validate pipeline configuration and dataset formatting on CPU without downloading a 15GB model:

```bash
# 1. Prepare chat dataset from risk splits
python training/prepare_dataset.py --input-dir backend/data/processed/risk_splits --output-dir training/data

# 2. Run dry-run validation
python training/train_lora.py --dry-run
```

---

## Google Colab / Kaggle GPU Execution

For full fine-tuning with 4-bit QLoRA on a free T4 or A100 GPU:

### 1. Install GPU Dependencies
```bash
pip install torch transformers peft bitsandbytes trl datasets accelerate pyyaml wandb
```

### 2. Run Training
```bash
python training/train_lora.py \
  --train-config training/configs/train_config.yaml \
  --lora-config training/configs/lora_config.yaml
```

Checkpoints will be saved to `training/checkpoints/qwen_lora/`.

### 3. Evaluate & Compare Against Baseline
```bash
python training/evaluate.py \
  --baseline-model backend/data/models/baseline_rf.joblib \
  --adapter-path training/checkpoints/qwen_lora
```
