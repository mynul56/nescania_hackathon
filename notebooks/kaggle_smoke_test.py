"""
Kaggle Notebook — Bengali Medical Dialogue QLoRA Smoke Test
===========================================================
PASTE EACH SECTION INTO A SEPARATE KAGGLE CODE CELL.

BEFORE RUNNING:
1. In the Kaggle notebook, click "+ Add Input" → search for the
   competition dataset (nescania_hackathon or nascenia-ai-hackathon)
   and attach it. The train.csv will then be at /kaggle/input/<dataset>/train.csv
2. Enable GPU: Settings → Accelerator → GPU T4 x2  (or P100)
3. Internet ON (needed to download Qwen2.5-1.5B-Instruct weights)
4. Run cells in order: CELL 1 → 2 → 3 → 4 → 5

After CELL 5 prints the 5 sample generations, copy them and share
with your assistant for review before proceeding to the short run.
"""

# ===========================================================================
# CELL 1 — Install missing packages (torch/transformers already on Kaggle)
# ===========================================================================

import subprocess, sys

for pkg in [
    "peft>=0.10.0",
    "trl>=0.8.6",
    "bitsandbytes>=0.43.0",
    "accelerate>=0.27.0",
    "datasets>=2.18.0",
]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch, transformers, peft, trl, bitsandbytes as bnb
print(f"torch        = {torch.__version__}")
print(f"CUDA         = {torch.cuda.is_available()}")
print(f"GPU          = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"transformers = {transformers.__version__}")
print(f"peft         = {peft.__version__}")
print(f"trl          = {trl.__version__}")
print(f"bitsandbytes = {bnb.__version__}")

if not torch.cuda.is_available():
    raise RuntimeError("No GPU! Enable GPU in notebook Settings → Accelerator.")

# ===========================================================================
# CELL 2 — Locate the data and build the fixed split
# ===========================================================================

import os, json, re, hashlib
from pathlib import Path
import pandas as pd

# ── Locate train.csv ─────────────────────────────────────────────────────────
# Try common Kaggle input locations
_candidates = list(Path("/kaggle/input").rglob("train.csv")) if Path("/kaggle/input").exists() else []
if not _candidates:
    raise FileNotFoundError(
        "Could not find train.csv under /kaggle/input. "
        "Did you attach the competition dataset? "
        "Click '+ Add Input' in the Kaggle notebook sidebar."
    )
TRAIN_CSV = _candidates[0]
print(f"Found train.csv at: {TRAIN_CSV}")

# ── Working dirs ─────────────────────────────────────────────────────────────
WORK = Path("/kaggle/working")
(WORK / "data/processed").mkdir(parents=True, exist_ok=True)
(WORK / "models").mkdir(parents=True, exist_ok=True)
(WORK / "experiments").mkdir(parents=True, exist_ok=True)
(WORK / "outputs").mkdir(parents=True, exist_ok=True)

SPLIT_JSON  = WORK / "data/processed/split_v1.json"
EXP_LOG     = WORK / "experiments/experiment_log.csv"
MODELS_DIR  = WORK / "models"

# ── SHA-256 of train.csv ──────────────────────────────────────────────────────
_sha = hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest()
print(f"train.csv SHA-256: {_sha}")
EXPECTED_SHA = "3ea68ae5e025ec82700a67716d3cbcecb37869e10569045b85c3b1734b525b80"
if _sha != EXPECTED_SHA:
    print(f"WARNING: SHA mismatch. Expected {EXPECTED_SHA}, got {_sha}. Proceeding anyway.")

# ── Load & inspect ────────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV, dtype={"id": str})
print(f"train.csv shape  : {df.shape}")
print(f"Columns          : {list(df.columns)}")
print(df.head(2).to_string())

# ── Check if we have a pre-built split or need to build one ───────────────────
# Try to find split_v1.json from a Kaggle dataset upload, otherwise build one
_split_candidates = list(Path("/kaggle/input").rglob("split_v1.json")) if Path("/kaggle/input").exists() else []

if _split_candidates:
    import shutil
    shutil.copy(_split_candidates[0], SPLIT_JSON)
    print(f"Copied split_v1.json from {_split_candidates[0]}")
else:
    print("split_v1.json not found in inputs — building a deterministic 90/10 split...")
    import numpy as np
    rng = np.random.default_rng(42)
    n = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * 0.90)
    train_row_ids = idx[:cut].tolist()
    val_row_ids   = idx[cut:].tolist()
    split_data = {
        "seed": 42,
        "split_version": "split_v1_kaggle_fallback",
        "note": "Fallback deterministic 90/10 split — use project split_v1.json when available",
        "train_row_ids": train_row_ids,
        "validation_row_ids": val_row_ids,
        "row_counts": {"train": len(train_row_ids), "validation": len(val_row_ids)},
    }
    SPLIT_JSON.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
    print(f"Built fallback split: {len(train_row_ids)} train / {len(val_row_ids)} val")

with open(SPLIT_JSON) as f:
    split = json.load(f)
print(f"Split: {split['row_counts']}")

# ===========================================================================
# CELL 3 — Configure constants and build smoke-test datasets
# ===========================================================================

import random
import statistics

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REV     = "main"
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
MAX_SEQ_LEN   = 512
SEED          = 42
SMOKE_FRAC    = 0.05    # 5% of train for smoke test
MAX_STEPS     = 100     # hard cap
SAVE_STEPS    = 50
LOG_STEPS     = 10
BATCH_SIZE    = 4
GRAD_ACCUM    = 4       # effective batch = 16

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ বাংলাদেশি চিকিৎসক। "
    "রোগীর প্রশ্নের উত্তর বাংলায় সংক্ষিপ্ত ও স্পষ্টভাবে দিন।"
)

# ── Build train/val DataFrames ─────────────────────────────────────────────────
df_all = pd.read_csv(TRAIN_CSV, dtype={"id": str})
df_all["_row_idx"] = df_all.index

train_ids = set(split["train_row_ids"])
val_ids   = set(split["validation_row_ids"])

train_df = df_all[df_all["_row_idx"].isin(train_ids)].copy().reset_index(drop=True)
val_df   = df_all[df_all["_row_idx"].isin(val_ids)].copy().reset_index(drop=True)

# Smoke subset (5%)
n_smoke = max(1, int(len(train_df) * SMOKE_FRAC))
smoke_df = train_df.sample(n=n_smoke, random_state=SEED).reset_index(drop=True)
print(f"Smoke train rows : {len(smoke_df)}")
print(f"Validation rows  : {len(val_df)}")

# ── Load tokenizer and report fertility ───────────────────────────────────────
from transformers import AutoTokenizer

print(f"\nLoading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
print(f"Vocabulary size  : {tokenizer.vocab_size}")

# Fertility report on 100 random samples
sample100 = smoke_df.sample(min(100, len(smoke_df)), random_state=SEED)
p_lens = [len(tokenizer.encode(str(r["input"])))  for _,r in sample100.iterrows()]
r_lens = [len(tokenizer.encode(str(r["output"]))) for _,r in sample100.iterrows()]
t_lens = [p+r for p,r in zip(p_lens, r_lens)]

def _pct(lst, q): return sorted(lst)[int(q*len(lst))]

print(f"\nFertility (100 samples):")
print(f"  Prompt  tokens — mean={statistics.mean(p_lens):.0f}, p50={_pct(p_lens,.50):.0f}, p95={_pct(p_lens,.95):.0f}, max={max(p_lens)}")
print(f"  Response tokens — mean={statistics.mean(r_lens):.0f}, p50={_pct(r_lens,.50):.0f}, p95={_pct(r_lens,.95):.0f}, max={max(r_lens)}")
print(f"  Total   tokens — mean={statistics.mean(t_lens):.0f}, p50={_pct(t_lens,.50):.0f}, p95={_pct(t_lens,.95):.0f}, max={max(t_lens)}")
if _pct(t_lens, .95) > MAX_SEQ_LEN:
    print(f"  WARNING: p95 total ({_pct(t_lens,.95)}) > MAX_SEQ_LEN ({MAX_SEQ_LEN}). Consider increasing.")
else:
    print(f"  OK: p95 total fits within MAX_SEQ_LEN={MAX_SEQ_LEN}")

# ===========================================================================
# CELL 4 — Load model, apply QLoRA, build dataset, TRAIN (smoke test)
# ===========================================================================

import csv as csv_mod
import time
from datetime import datetime, timezone
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

EXPERIMENT_ID = "smoke-test-v1"
out_dir = MODELS_DIR / EXPERIMENT_ID
out_dir.mkdir(parents=True, exist_ok=True)

# ── QLoRA model ───────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} in 4-bit NF4...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REV,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.enable_input_require_grads()

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS, bias="none", inference_mode=False,
)
model = get_peft_model(model, lora_cfg)

# Parameter count
total_p     = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params     : {total_p/1e9:.3f}B")
print(f"Trainable params : {trainable_p/1e6:.1f}M  ({100*trainable_p/total_p:.2f}%)")
assert total_p <= 3_000_000_000, f"AGENTS.md violation: {total_p/1e9:.2f}B > 3B cap!"

# ── Dataset ───────────────────────────────────────────────────────────────────
def make_text(row):
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": str(row["input"])},
        {"role": "assistant", "content": str(row["output"])},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False)
    out["labels"] = out["input_ids"].copy()
    return out

print("\nBuilding train dataset...")
train_texts = [make_text(r) for _,r in smoke_df.iterrows()]
train_ds = Dataset.from_dict({"text": train_texts})
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"Train examples   : {len(train_ds)}")

val_texts = [make_text(r) for _,r in val_df.head(200).iterrows()]
val_ds = Dataset.from_dict({"text": val_texts})
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"Val examples     : {len(val_ds)} (capped at 200 for in-training eval)")

# ── Log experiment start ──────────────────────────────────────────────────────
cfg_str = (
    f"mode=smoke;r={LORA_R};alpha={LORA_ALPHA};dropout={LORA_DROPOUT};"
    f"max_seq_len={MAX_SEQ_LEN};batch={BATCH_SIZE}x{GRAD_ACCUM};"
    f"max_steps={MAX_STEPS};train_rows={len(smoke_df)}"
)
fieldnames = [
    "experiment_id","git_commit","dataset_version","split_version","model_name",
    "model_revision","exact_total_parameters","trainable_parameters","tokenizer",
    "training_config","generation_settings","validation_metrics",
    "manual_safety_findings","kaggle_notebook_version","decision_next_step",
]
write_hdr = not EXP_LOG.exists()
with open(EXP_LOG, "a", newline="", encoding="utf-8") as f:
    w = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if write_hdr: w.writeheader()
    w.writerow({
        "experiment_id": EXPERIMENT_ID,
        "dataset_version": f"train_rows=108954;sha256={_sha}",
        "split_version": "split_v1",
        "model_name": MODEL_ID, "model_revision": MODEL_REV,
        "exact_total_parameters": f"{total_p/1e9:.3f}B",
        "trainable_parameters": f"{trainable_p/1e6:.1f}M",
        "tokenizer": "Qwen2.5 BPE v151646",
        "training_config": cfg_str,
        "decision_next_step": "smoke training started",
    })
print("Experiment logged.")

# ── TrainingArguments ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(out_dir),
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False, bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    logging_steps=LOG_STEPS,
    evaluation_strategy="steps",
    eval_steps=SAVE_STEPS,
    load_best_model_at_end=False,
    report_to="none",
    seed=SEED,
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

collator = DataCollatorForSeq2Seq(
    tokenizer, model=model,
    label_pad_token_id=-100, pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    tokenizer=tokenizer, data_collator=collator,
)

print(f"\nStarting smoke training: max_steps={MAX_STEPS}, save every {SAVE_STEPS} steps")
print("=" * 60)
t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0

print("=" * 60)
print(f"Training complete: {elapsed/60:.1f} min")
print(f"Final train loss : {result.training_loss:.4f}")

# Save adapter
adapter_dir = out_dir / "final_adapter"
model.save_pretrained(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))
print(f"Adapter saved to : {adapter_dir}")

# ===========================================================================
# CELL 5 — Generate 5 samples for manual review (Step 5)
# ===========================================================================

print("\n" + "="*60)
print("STEP 5 — Manual review samples (n=5)")
print("Review these BEFORE proceeding to short/full training.")
print("="*60)

model.eval()
samples_5 = val_df.sample(n=5, random_state=SEED)
review_results = []

for i, (_, row) in enumerate(samples_5.iterrows(), 1):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": str(row["input"])},
    ]
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_toks = out_ids[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_toks, skip_special_tokens=True)

    entry = {
        "sample_num":          i,
        "patient_prompt":      str(row["input"]),
        "reference_response":  str(row["output"]),
        "generated_response":  generated,
        "gen_len_tokens":      len(new_toks),
        "ref_len_tokens":      len(tokenizer.encode(str(row["output"]))),
    }
    review_results.append(entry)

    print(f"\n{'─'*60}")
    print(f"[{i}/5] PATIENT:   {str(row['input'])[:300]}")
    print(f"       REFERENCE: {str(row['output'])[:300]}")
    print(f"       GENERATED: {generated[:300]}")
    print(f"       gen_tokens={entry['gen_len_tokens']}, ref_tokens={entry['ref_len_tokens']}")

# Save to JSON
import json as _json
samples_path = out_dir / "smoke_samples.json"
samples_path.write_text(_json.dumps(review_results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n{'='*60}")
print(f"Samples saved: {samples_path}")
print(f"Training loss: {result.training_loss:.4f}")
print(f"Checkpoint  : {out_dir}")
print(f"{'='*60}")

# Log completion
with open(EXP_LOG, "a", newline="", encoding="utf-8") as f:
    w = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writerow({
        "experiment_id": f"{EXPERIMENT_ID}-COMPLETE",
        "dataset_version": f"train_rows=108954;sha256={_sha}",
        "split_version": "split_v1",
        "model_name": MODEL_ID, "model_revision": MODEL_REV,
        "exact_total_parameters": f"{total_p/1e9:.3f}B",
        "trainable_parameters": f"{trainable_p/1e6:.1f}M",
        "tokenizer": "Qwen2.5 BPE v151646",
        "training_config": cfg_str,
        "validation_metrics": f"train_loss={result.training_loss:.4f};steps={MAX_STEPS}",
        "decision_next_step": "AWAITING HUMAN REVIEW of smoke_samples.json before short run",
    })
print("Completion logged to experiment_log.csv")
print("\nNEXT: Copy the 5 samples above and share with your assistant for review.")
print("DO NOT start the short run until samples are approved.")
