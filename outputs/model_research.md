# Model Research — Bengali Medical Dialogue Generation

**Date:** 2026-08-04  
**Dataset:** 108,954 train rows; split_v1 → 98,074 train / 10,880 validation  
**Task:** Bengali doctor-response generation from Bengali patient prompt  
**Hard constraint:** ≤ 3B total parameters at inference (ALL parameter counts are active/dense at inference)

---

## Retrieval Baseline Scores

These are the scores to beat with a fine-tuned model:

| Baseline | Token F1 | ROUGE-L |
|---|---|---|
| Exact Match | ~0.0495 | — |
| TF-IDF | 0.1834 | — |
| BM25 | 0.1831 | 0.1140 |

---

## Candidate Model Evaluation

### DISQUALIFIED: Phi-3.5-mini-instruct

- **Model:** `microsoft/Phi-3.5-mini-instruct`
- **Exact parameters (model card):** 3.8B dense parameters
- **Verdict:** OVER 3B CAP — ELIMINATED.

---

### Candidate 1: Qwen/Qwen2.5-1.5B-Instruct — PRIMARY

**Parameter count (official HF model card):**
- Total parameters: 1.54B
- Non-embedding parameters: 1.31B
- Dense model (all parameters active at inference)
- Under 3B cap by a large margin

**Checkpoint:** `Qwen2.5-1.5B-Instruct` specifically (family has 0.5B/1.5B/3B/7B/14B/32B/72B variants)

**Bengali support:**
- Pretrained on 18T tokens, 29+ languages
- Bengali in multilingual corpus (not primary language)
- Community fine-tunes on HF confirm functional Bengali capability
- BPE tokenizer with 151,646 vocabulary; expected 2-3x fertility vs English

**License:** Apache 2.0 (no gating, commercial friendly)  
**Context:** 32K tokens  
**Decision: PRIMARY CANDIDATE**

---

### Candidate 2: hishab/titulm-gemma-2-2b-v1.1 — ALTERNATE

**Parameter count (HF model card):** 2.6B total — under 3B cap

**Checkpoint:** Continual-pretrained Gemma-2-2B on 4.4B Bengali tokens by Hishab

**Bengali support:** Excellent — purpose-built for Bengali, outperforms base Gemma-2-2B on Bengali benchmarks

**Context:** 4096 tokens  
**License:** Gemma license (GATED — requires HF token and access approval)  
**Decision: STRONG ALTERNATE / ENSEMBLE CANDIDATE**

---

### Candidate 3: meta-llama/Llama-3.2-1B-Instruct — REJECTED

**Parameters:** 1.23B  
**Bengali support:** NOT officially supported (official languages: EN/DE/FR/IT/PT/HI/ES/TH)  
**Decision: REJECTED — no documented Bengali pretraining coverage**

---

### Candidate 4: google/mt5-base — FALLBACK ONLY

**Parameters:** ~580M (encoder-decoder architecture)  
**Bengali support:** Yes — explicitly in mT5 pretraining (101 languages)  
**Decision: FALLBACK ONLY** — different architecture (Seq2SeqTrainer), not instruction-tuned

---

## Primary Selection

| Attribute | Value |
|---|---|
| Primary model | Qwen/Qwen2.5-1.5B-Instruct |
| Exact parameters | 1.54B total / 1.31B non-embedding |
| License | Apache 2.0 |
| Training approach | QLoRA (4-bit NF4 + LoRA adapters via PEFT) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA rank | r=16, alpha=32, dropout=0.05 |
| Alternate | hishab/titulm-gemma-2-2b-v1.1 (if HF token available in Kaggle) |

---

## Tokenizer Fertility Test (to run in Kaggle notebook)

```python
from transformers import AutoTokenizer
import pandas as pd

df = pd.read_csv("../input/train.csv")
sample = df.sample(100, random_state=42)

for model_id in ["Qwen/Qwen2.5-1.5B-Instruct"]:
    tok = AutoTokenizer.from_pretrained(model_id)
    prompt_tokens = sample["input"].apply(lambda x: len(tok.encode(x)))
    resp_tokens  = sample["output"].apply(lambda x: len(tok.encode(x)))
    print(f"{model_id}")
    print(f"  Prompt — mean: {prompt_tokens.mean():.1f}, p95: {prompt_tokens.quantile(0.95):.0f}")
    print(f"  Response — mean: {resp_tokens.mean():.1f}, p95: {resp_tokens.quantile(0.95):.0f}")
    print(f"  Max seq len (p95 prompt+resp): {(prompt_tokens+resp_tokens).quantile(0.95):.0f}")
```

NOTE: Cannot run locally — model too large to download. Must run in Kaggle notebook.

---

## Next Steps

1. Log this to experiment_log.csv
2. Run tokenizer fertility test in Kaggle notebook
3. Build src/train.py with QLoRA pipeline
4. Run smoke-test-v1 (5% data subset, confirm GPU/loss/generation)
5. Review generations before any full training run
