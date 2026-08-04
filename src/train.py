"""QLoRA fine-tuning of Qwen/Qwen2.5-1.5B-Instruct for Bengali medical dialogue.

Usage
-----
Smoke test (run this first in Kaggle notebook):
    python src/train.py --mode smoke --experiment-id smoke-test-v1

Short training run (after smoke test is approved):
    python src/train.py --mode short --experiment-id short-run-v1

Full training run (after short run is approved):
    python src/train.py --mode full --experiment-id full-run-v1

Resume from checkpoint:
    python src/train.py --mode short --experiment-id short-run-v1 --resume

AGENTS.md rules enforced here:
- Never reads test.csv
- Always uses fixed split_v1 from data/processed/split_v1.json
- Logs every run to experiments/experiment_log.csv before training begins
- Prints exact parameter counts before training
- Saves to models/<experiment-id>/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

# ── Paths (same ROOT convention as retrieval_baselines.py) ──────────────────
ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
SPLIT_JSON = ROOT / "data" / "processed" / "split_v1.json"
MODELS_DIR = ROOT / "models"
EXPERIMENT_LOG = ROOT / "experiments" / "experiment_log.csv"

# ── Model / training constants ────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "main"

# LoRA hyper-parameters (from model_research.md)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Tokenisation
MAX_SEQ_LEN = 512      # covers p95 of Bengali prompts (to be confirmed via fertility test)
RESPONSE_TEMPLATE = "\n### Response:\n"  # delimiter between prompt and response in training

# Prompt template — instruction format
SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ বাংলাদেশি চিকিৎসক। "
    "রোগীর প্রশ্নের উত্তর বাংলায় সংক্ষিপ্ত ও স্পষ্টভাবে দিন।"
)

# Training budgets per mode
MODE_CONFIG: dict[str, dict[str, Any]] = {
    "smoke": {
        "train_frac": 0.05,          # 5% of train rows (~4,904 rows)
        "num_train_epochs": 1,
        "max_steps": 100,            # hard cap — whichever limit is hit first
        "save_steps": 50,
        "logging_steps": 10,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,  # effective batch = 16
    },
    "short": {
        "train_frac": 1.0,           # full train set (~98,074 rows)
        "num_train_epochs": 1,
        "max_steps": -1,             # full epoch
        "save_steps": 500,
        "logging_steps": 50,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
    },
    "full": {
        "train_frac": 1.0,
        "num_train_epochs": 3,
        "max_steps": -1,
        "save_steps": 1000,
        "logging_steps": 100,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — Environment checks
# ─────────────────────────────────────────────────────────────────────────────

def check_environment() -> None:
    """Print environment info and abort if GPU is not available."""
    log.info("=" * 60)
    log.info("STEP 0 — Environment Check")
    log.info("=" * 60)

    # torch
    log.info("torch version      : %s", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    log.info("CUDA available     : %s", cuda_ok)
    if cuda_ok:
        log.info("GPU name           : %s", torch.cuda.get_device_name(0))
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU memory (total) : %.1f GB", mem_gb)
    else:
        log.error(
            "No GPU detected. QLoRA training requires a CUDA GPU. "
            "This script is designed for Kaggle notebook GPU — not local CPU. Exiting."
        )
        sys.exit(1)

    # transformers
    try:
        import transformers
        log.info("transformers ver   : %s", transformers.__version__)
    except ImportError:
        log.error("transformers not installed. Run: pip install transformers")
        sys.exit(1)

    # peft
    try:
        import peft
        log.info("peft version       : %s", peft.__version__)
    except ImportError:
        log.error("peft not installed. Run: pip install peft")
        sys.exit(1)

    # bitsandbytes
    try:
        import bitsandbytes as bnb
        log.info("bitsandbytes ver   : %s", bnb.__version__)
    except ImportError:
        log.error("bitsandbytes not installed. Run: pip install bitsandbytes")
        sys.exit(1)

    # trl (for SFTTrainer)
    try:
        import trl
        log.info("trl version        : %s", trl.__version__)
    except ImportError:
        log.error("trl not installed. Run: pip install trl")
        sys.exit(1)

    log.info("Environment check passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_split() -> tuple[list[int], list[int]]:
    """Load the fixed split_v1 row IDs. Never re-splits."""
    log.info("Loading split from %s", SPLIT_JSON)
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)
    train_ids = set(split["train_row_ids"])
    val_ids = set(split["validation_row_ids"])
    log.info("Split loaded: %d train rows, %d val rows", len(train_ids), len(val_ids))
    return train_ids, val_ids


def load_data(train_ids: set[int], val_ids: set[int], train_frac: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and return (train_df, val_df) using the fixed split.

    train_frac: fraction of train rows to use (< 1.0 for smoke tests).
    AGENTS.md: never reads test.csv.
    """
    log.info("Loading %s", TRAIN_CSV)
    df = pd.read_csv(TRAIN_CSV, dtype={"id": int})

    # Assign integer row index as merge key (0-based position)
    df["_row_idx"] = df.index

    train_df = df[df["_row_idx"].isin(train_ids)].copy()
    val_df = df[df["_row_idx"].isin(val_ids)].copy()

    # Subset for smoke test
    if train_frac < 1.0:
        n = max(1, int(len(train_df) * train_frac))
        train_df = train_df.sample(n=n, random_state=42).reset_index(drop=True)
        log.info("Smoke subset: using %d / %d train rows", n, len(train_ids))

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    log.info("Data loaded: %d train, %d val", len(train_df), len(val_df))
    return train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_prompt(row: pd.Series, tokenizer: Any, include_response: bool = True) -> str:
    """Format a single row into a chat-template string.

    Uses Qwen2.5's apply_chat_template so the model sees the same
    format it was instruction-tuned with.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": str(row["input"])},
    ]

    if include_response:
        messages.append({"role": "assistant", "content": str(row["output"])})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # for inference
        )
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer fertility report
# ─────────────────────────────────────────────────────────────────────────────

def report_tokenizer_fertility(tokenizer: Any, train_df: pd.DataFrame) -> None:
    """Print token length statistics for 100 random samples."""
    log.info("Tokenizer fertility report (100 random samples):")
    sample = train_df.sample(min(100, len(train_df)), random_state=42)

    prompt_lens, resp_lens, total_lens = [], [], []
    for _, row in sample.iterrows():
        p_text = str(row["input"])
        r_text = str(row["output"])
        p_len = len(tokenizer.encode(p_text))
        r_len = len(tokenizer.encode(r_text))
        prompt_lens.append(p_len)
        resp_lens.append(r_len)
        total_lens.append(p_len + r_len)

    def _stats(lst: list[int], name: str) -> None:
        import statistics
        log.info(
            "  %s — mean=%.0f, median=%.0f, p95=%.0f, max=%d",
            name,
            statistics.mean(lst),
            statistics.median(lst),
            sorted(lst)[int(0.95 * len(lst))],
            max(lst),
        )

    _stats(prompt_lens, "Prompt tokens    ")
    _stats(resp_lens,   "Response tokens  ")
    _stats(total_lens,  "Total tokens     ")

    p95_total = sorted(total_lens)[int(0.95 * len(total_lens))]
    if p95_total > MAX_SEQ_LEN:
        log.warning(
            "p95 total tokens (%d) exceeds MAX_SEQ_LEN (%d). "
            "Consider increasing --max-seq-len.",
            p95_total, MAX_SEQ_LEN,
        )
    log.info("")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter count report
# ─────────────────────────────────────────────────────────────────────────────

def report_param_count(model: Any) -> dict[str, int]:
    """Print and return exact parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    log.info("=" * 60)
    log.info("Parameter count")
    log.info("  Total         : %s (%d)", f"{total/1e9:.3f}B", total)
    log.info("  Trainable     : %s (%d)", f"{trainable/1e6:.1f}M", trainable)
    log.info("  Frozen        : %s (%d)", f"{frozen/1e9:.3f}B", frozen)
    log.info("  Trainable %%   : %.2f%%", 100 * trainable / total)
    log.info("=" * 60)

    # Hard-fail if somehow over 3B
    if total > 3_000_000_000:
        raise RuntimeError(
            f"AGENTS.md violation: model has {total/1e9:.2f}B parameters, "
            "exceeding the 3B inference constraint."
        )
    log.info("")
    return {"total": total, "trainable": trainable}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment log
# ─────────────────────────────────────────────────────────────────────────────

def append_experiment_log(row: dict[str, Any]) -> None:
    """Append one row to experiments/experiment_log.csv (AGENTS.md requirement)."""
    fieldnames = [
        "experiment_id", "git_commit", "dataset_version", "split_version",
        "model_name", "model_revision", "exact_total_parameters",
        "trainable_parameters", "tokenizer", "training_config",
        "generation_settings", "validation_metrics", "manual_safety_findings",
        "kaggle_notebook_version", "decision_next_step",
    ]
    write_header = not EXPERIMENT_LOG.exists() or EXPERIMENT_LOG.stat().st_size < 10
    with open(EXPERIMENT_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    log.info("Experiment logged to %s", EXPERIMENT_LOG)


# ─────────────────────────────────────────────────────────────────────────────
# Model & tokenizer loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(output_dir: Path) -> tuple[Any, Any]:
    """Load Qwen2.5-1.5B-Instruct in 4-bit NF4 with LoRA adapters."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info("Loading tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    # Qwen2.5 has no pad token by default — use eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # needed for causal LM training

    log.info("Loading model: %s (4-bit NF4 quantization)", MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False          # required for gradient checkpointing
    model.config.pretraining_tp = 1         # disable tensor parallelism (single GPU)
    model.enable_input_require_grads()      # needed when using gradient checkpointing with PEFT

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def build_hf_dataset(df: pd.DataFrame, tokenizer: Any, max_seq_len: int) -> Any:
    """Convert a DataFrame to a HuggingFace Dataset of tokenised sequences."""
    from datasets import Dataset

    texts = [format_prompt(row, tokenizer, include_response=True) for _, row in df.iterrows()]

    def tokenize(batch: dict) -> dict:
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Generation samples (Step 5 of smoke test)
# ─────────────────────────────────────────────────────────────────────────────

def generate_samples(
    model: Any,
    tokenizer: Any,
    val_df: pd.DataFrame,
    n: int = 5,
    max_new_tokens: int = 256,
) -> list[dict[str, str]]:
    """Generate n sample responses from the validation set for manual review."""
    model.eval()
    results = []
    sample = val_df.sample(n=min(n, len(val_df)), random_state=42)

    for _, row in sample.iterrows():
        prompt_text = format_prompt(row, tokenizer, include_response=False)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy for reproducibility in smoke test
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append({
            "patient_prompt": str(row["input"]),
            "reference_response": str(row["output"]),
            "generated_response": generated,
        })
        print("\n" + "─" * 60)
        print(f"[PATIENT]:    {row['input'][:200]}")
        print(f"[REFERENCE]:  {row['output'][:200]}")
        print(f"[GENERATED]:  {generated[:200]}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Bengali medical dialogue")
    parser.add_argument(
        "--mode",
        choices=["smoke", "short", "full"],
        default="smoke",
        help="Training mode: smoke (100 steps, 5%% data), short (1 epoch, full data), full (3 epochs)",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment ID for logging (default: <mode>-<timestamp>)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in models/<experiment-id>/",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Maximum sequence length (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--skip-samples",
        action="store_true",
        help="Skip the generation sample step (Step 5). Not recommended for smoke test.",
    )
    args = parser.parse_args()

    # ── Step 0: Environment ──────────────────────────────────────────────────
    check_environment()

    # ── Resolve experiment ID ────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    experiment_id = args.experiment_id or f"{args.mode}-{ts}"
    output_dir = MODELS_DIR / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Experiment ID : %s", experiment_id)
    log.info("Output dir    : %s", output_dir)

    cfg = MODE_CONFIG[args.mode]
    log.info("Mode          : %s", args.mode)
    log.info("Train budget  : frac=%.2f, epochs=%d, max_steps=%d",
             cfg["train_frac"], cfg["num_train_epochs"], cfg["max_steps"])

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    log.info("\nSTEP 1 — Loading data")
    train_ids, val_ids = load_split()
    train_df, val_df = load_data(train_ids, val_ids, train_frac=cfg["train_frac"])

    # ── Step 2: Model & tokenizer ─────────────────────────────────────────────
    log.info("\nSTEP 2 — Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(output_dir)
    param_counts = report_param_count(model)
    report_tokenizer_fertility(tokenizer, train_df)

    # ── Pre-training experiment log entry (AGENTS.md requirement) ─────────────
    training_cfg_str = (
        f"mode={args.mode};r={LORA_R};alpha={LORA_ALPHA};dropout={LORA_DROPOUT};"
        f"targets={'|'.join(LORA_TARGET_MODULES)};"
        f"max_seq_len={args.max_seq_len};"
        f"batch={cfg['per_device_train_batch_size']}x{cfg['gradient_accumulation_steps']};"
        f"max_steps={cfg['max_steps']};epochs={cfg['num_train_epochs']};"
        f"train_rows={len(train_df)}"
    )
    append_experiment_log({
        "experiment_id": experiment_id,
        "dataset_version": "train_rows=108954;sha256=3ea68ae5e025ec82700a67716d3cbcecb37869e10569045b85c3b1734b525b80",
        "split_version": "split_v1",
        "model_name": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "exact_total_parameters": f"{param_counts['total']/1e9:.3f}B",
        "trainable_parameters": f"{param_counts['trainable']/1e6:.1f}M",
        "tokenizer": "Qwen2.5 BPE v151646",
        "training_config": training_cfg_str,
        "decision_next_step": "training started",
    })

    # ── Step 3: Dataset ───────────────────────────────────────────────────────
    log.info("\nSTEP 3 — Building HuggingFace datasets")
    from transformers import DataCollatorForSeq2Seq
    train_ds = build_hf_dataset(train_df, tokenizer, args.max_seq_len)
    val_ds = build_hf_dataset(val_df.head(500), tokenizer, args.max_seq_len)  # val subset for eval during training
    log.info("Train dataset: %d examples", len(train_ds))
    log.info("Val dataset  : %d examples (capped at 500 for in-training eval)", len(val_ds))

    # ── Step 4: Training ──────────────────────────────────────────────────────
    log.info("\nSTEP 4 — Training")
    from transformers import Trainer, TrainingArguments

    resume_from = None
    if args.resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            log.info("Resuming from checkpoint: %s", resume_from)
        else:
            log.warning("--resume specified but no checkpoints found in %s", output_dir)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",      # QLoRA-compatible optimiser
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,                      # T4 supports bf16
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        logging_steps=cfg["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["save_steps"],
        load_best_model_at_end=False,   # avoid reloading quantised model (can cause OOM)
        report_to="none",               # no WandB / no internet required
        seed=42,
        dataloader_num_workers=0,       # Kaggle multiprocessing issues
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,        # mask pad tokens in loss
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - t0
    log.info("Training complete in %.1f min", elapsed / 60)

    # Save final adapter
    final_adapter_dir = output_dir / "final_adapter"
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    log.info("Saved final adapter to %s", final_adapter_dir)

    # ── Step 5: Generation samples for manual review ───────────────────────────
    if not args.skip_samples:
        log.info("\nSTEP 5 — Generating samples for manual review (n=5)")
        samples = generate_samples(model, tokenizer, val_df, n=5, max_new_tokens=256)

        samples_path = output_dir / "smoke_samples.json"
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        log.info("\nSamples saved to %s", samples_path)
        log.info("REVIEW THESE BEFORE PROCEEDING TO FULL TRAINING.")

    # ── Update experiment log with training outcome ────────────────────────────
    append_experiment_log({
        "experiment_id": f"{experiment_id}-COMPLETE",
        "dataset_version": "train_rows=108954;sha256=3ea68ae5e025ec82700a67716d3cbcecb37869e10569045b85c3b1734b525b80",
        "split_version": "split_v1",
        "model_name": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "exact_total_parameters": f"{param_counts['total']/1e9:.3f}B",
        "trainable_parameters": f"{param_counts['trainable']/1e6:.1f}M",
        "tokenizer": "Qwen2.5 BPE v151646",
        "training_config": training_cfg_str,
        "decision_next_step": f"training finished in {elapsed/60:.1f}min; review samples before next step",
    })

    log.info("\n%s", "=" * 60)
    log.info("RUN COMPLETE: %s", experiment_id)
    log.info("Checkpoint dir : %s", output_dir)
    log.info("Next step: review %s/smoke_samples.json then proceed to short run", output_dir)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
