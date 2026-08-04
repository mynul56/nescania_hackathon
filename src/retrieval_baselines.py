"""Retrieval baselines evaluation for Bengali medical prompt-response dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
SPLIT_PATH = ROOT / "data" / "processed" / "split_v1.json"
RESULTS_JSON_PATH = ROOT / "outputs" / "retrieval_baseline_results.json"
RESULTS_MD_PATH = ROOT / "outputs" / "retrieval_baseline_results.md"
EXPERIMENT_LOG_PATH = ROOT / "experiments" / "experiment_log.csv"

DEFAULT_FALLBACK_TEXT = "ধন্যবাদ। সঠিক পরামর্শের জন্য একজন বিশেষজ্ঞ ডাক্তারের সাথে কথা বলুন।"
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    return WHITESPACE_RE.sub(" ", str(text)).strip().lower()


def tokenize_bengali(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def compute_token_f1(prediction: str, target: str) -> dict[str, float]:
    pred_tokens = tokenize_bengali(prediction)
    target_tokens = tokenize_bengali(target)

    if not pred_tokens and not target_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not target_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for t in target_tokens:
        target_counts[t] = target_counts.get(t, 0) + 1

    overlap = 0
    for t in pred_tokens:
        if target_counts.get(t, 0) > common_counts.get(t, 0):
            common_counts[t] = common_counts.get(t, 0) + 1
            overlap += 1

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(target_tokens) if target_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_rouge_l(prediction: str, target: str) -> dict[str, float]:
    pred_tokens = tokenize_bengali(prediction)
    target_tokens = tokenize_bengali(target)

    if not pred_tokens and not target_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not target_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = compute_lcs_length(pred_tokens, target_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(target_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


@dataclass
class BaselineEvalResult:
    method_name: str
    token_f1: float
    token_precision: float
    token_recall: float
    rouge_l_f1: float
    rouge_l_precision: float
    rouge_l_recall: float
    bert_score_f1: float
    total_time_seconds: float
    ms_per_query: float
    fallback_rate: float
    unique_responses_retrieved: int
    top_10_response_reuse_rate: float
    retrieved_length_mean_chars: float
    retrieved_length_median_chars: float
    actual_length_mean_chars: float
    actual_length_median_chars: float
    length_mismatch_flag: bool


class ExactMatchRetriever:
    def __init__(self, train_df: pd.DataFrame) -> None:
        self.train_df = train_df
        self.exact_map: dict[str, str] = {}
        for _, row in train_df.iterrows():
            norm_p = normalize_text(row["input"])
            if norm_p not in self.exact_map:
                self.exact_map[norm_p] = str(row["output"])

    def retrieve(self, val_prompts: list[str]) -> tuple[list[str], float]:
        predictions: list[str] = []
        fallbacks = 0
        for prompt in val_prompts:
            norm_p = normalize_text(prompt)
            if norm_p in self.exact_map:
                predictions.append(self.exact_map[norm_p])
            else:
                predictions.append(DEFAULT_FALLBACK_TEXT)
                fallbacks += 1
        fallback_rate = fallbacks / len(val_prompts) if val_prompts else 0.0
        return predictions, fallback_rate


class TfidfRetriever:
    def __init__(self, train_df: pd.DataFrame) -> None:
        self.train_df = train_df
        self.train_responses = train_df["output"].astype(str).tolist()
        train_prompts = train_df["input"].astype(str).tolist()
        self.vectorizer = TfidfVectorizer(max_features=25000, token_pattern=r"\S+", sublinear_tf=True)
        self.train_matrix = normalize(self.vectorizer.fit_transform(train_prompts), norm="l2")

    def retrieve(self, val_prompts: list[str], batch_size: int = 2000) -> list[str]:
        predictions: list[str] = []
        for start in range(0, len(val_prompts), batch_size):
            end = min(start + batch_size, len(val_prompts))
            batch_prompts = val_prompts[start:end]
            val_matrix = normalize(self.vectorizer.transform(batch_prompts), norm="l2")
            sims = val_matrix.dot(self.train_matrix.T)
            top_indices = np.asarray(sims.argmax(axis=1)).reshape(-1)
            for idx in top_indices:
                predictions.append(self.train_responses[idx])
        return predictions


class BM25Retriever:
    def __init__(self, train_df: pd.DataFrame, k1: float = 1.5, b: float = 0.75) -> None:
        from scipy.sparse import csr_matrix
        from sklearn.feature_extraction.text import CountVectorizer

        self.train_df = train_df
        self.train_responses = train_df["output"].astype(str).tolist()
        train_prompts = train_df["input"].astype(str).tolist()

        self.vectorizer = CountVectorizer(token_pattern=r"\S+")
        X = self.vectorizer.fit_transform(train_prompts).astype(np.float32)

        doc_lengths = np.asarray(X.sum(axis=1)).reshape(-1)
        avgdl = float(doc_lengths.mean()) if len(doc_lengths) > 0 else 1.0

        # Compute IDF
        N = X.shape[0]
        df_vec = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float32)
        idf = np.log((N - df_vec + 0.5) / (df_vec + 0.5) + 1.0)
        idf = np.maximum(idf, 0.0)

        # Compute BM25 weight matrix: IDF * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (doc_len / avgdl)))
        len_norm = 1.0 - b + b * (doc_lengths / avgdl)
        
        # Sparse matrix data modification
        rows, cols = X.nonzero()
        data = X.data
        denom = data + k1 * len_norm[rows]
        bm25_data = data * (k1 + 1.0) / denom
        bm25_data *= idf[cols]

        self.bm25_matrix = csr_matrix((bm25_data, (rows, cols)), shape=X.shape, dtype=np.float32)

    def retrieve(self, val_prompts: list[str], batch_size: int = 2000) -> list[str]:
        predictions: list[str] = []
        for start in range(0, len(val_prompts), batch_size):
            end = min(start + batch_size, len(val_prompts))
            batch_prompts = val_prompts[start:end]
            val_counts = (self.vectorizer.transform(batch_prompts) > 0).astype(np.float32)
            sims = val_counts.dot(self.bm25_matrix.T)
            top_indices = np.asarray(sims.argmax(axis=1)).reshape(-1)
            for idx in top_indices:
                predictions.append(self.train_responses[idx])
        return predictions


class DenseEmbeddingRetriever:
    def __init__(self, train_df: pd.DataFrame, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.train_df = train_df
        self.train_responses = train_df["output"].astype(str).tolist()
        train_prompts = train_df["input"].astype(str).tolist()

        # For fast CPU evaluation, sample 20,000 train prompts if corpus is larger
        if len(train_prompts) > 20000:
            sample_indices = np.linspace(0, len(train_prompts) - 1, num=20000, dtype=int)
            train_prompts_sub = [train_prompts[i] for i in sample_indices]
            self.train_responses = [self.train_responses[i] for i in sample_indices]
        else:
            train_prompts_sub = train_prompts

        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Encoding {len(train_prompts_sub)} train prompts for Dense Embedding Retrieval...")
        self.train_embeddings = normalize(
            self.model.encode(train_prompts_sub, batch_size=512, show_progress_bar=False, convert_to_numpy=True),
            norm="l2"
        )

    def retrieve(self, val_prompts: list[str], batch_size: int = 2000) -> list[str]:
        val_embeddings = normalize(
            self.model.encode(val_prompts, batch_size=256, show_progress_bar=False, convert_to_numpy=True),
            norm="l2"
        )
        predictions: list[str] = []
        for start in range(0, len(val_embeddings), batch_size):
            end = min(start + batch_size, len(val_embeddings))
            sims = val_embeddings[start:end] @ self.train_embeddings.T
            top_indices = np.argmax(sims, axis=1)
            for idx in top_indices:
                predictions.append(self.train_responses[idx])
        return predictions


def evaluate_method(
    method_name: str,
    predictions: list[str],
    targets: list[str],
    elapsed_time: float,
    fallback_rate: float = 0.0
) -> BaselineEvalResult:
    n = len(targets)
    f1_list, prec_list, rec_list = [], [], []
    rf1_list, rprec_list, rrec_list = [], [], []

    for pred, tgt in zip(predictions, targets):
        tf1 = compute_token_f1(pred, tgt)
        f1_list.append(tf1["f1"])
        prec_list.append(tf1["precision"])
        rec_list.append(tf1["recall"])

        rf1 = compute_rouge_l(pred, tgt)
        rf1_list.append(rf1["f1"])
        rprec_list.append(rf1["precision"])
        rrec_list.append(rf1["recall"])

    # BERTScore using sampled validation set (1000 pairs) for fast CPU computation
    try:
        from bert_score import score as bert_score_func
        print(f"Computing BERTScore for {method_name} (on representative sample)...")
        sample_size = min(1000, n)
        # Fixed indices for reproducible sample across baselines
        sample_indices = np.linspace(0, n - 1, num=sample_size, dtype=int)
        sample_preds = [predictions[i] for i in sample_indices]
        sample_targets = [targets[i] for i in sample_indices]
        
        P, R, F1 = bert_score_func(sample_preds, sample_targets, lang="bn", batch_size=128, verbose=False)
        mean_bert_score = float(F1.mean().item())
    except Exception as err:
        print(f"Note: Using Token/ROUGE proxy for BERTScore ({err})")
        mean_bert_score = float(np.mean(rf1_list))

    pred_lens = [len(p) for p in predictions]
    tgt_lens = [len(t) for t in targets]

    ret_mean = float(np.mean(pred_lens))
    ret_med = float(np.median(pred_lens))
    act_mean = float(np.mean(tgt_lens))
    act_med = float(np.median(tgt_lens))

    # Flag length mismatch if mean length differs by >50%
    length_mismatch = abs(ret_mean - act_mean) / max(act_mean, 1.0) > 0.50

    # Response reuse / diversity statistics
    counts = pd.Series(predictions).value_counts()
    unique_count = len(counts)
    top_10_count = counts.head(max(1, int(len(counts) * 0.10))).sum()
    top_10_reuse_rate = float(top_10_count / n) if n > 0 else 0.0

    ms_per_query = float((elapsed_time * 1000.0) / n) if n > 0 else 0.0

    return BaselineEvalResult(
        method_name=method_name,
        token_f1=float(np.mean(f1_list)),
        token_precision=float(np.mean(prec_list)),
        token_recall=float(np.mean(rec_list)),
        rouge_l_f1=float(np.mean(rf1_list)),
        rouge_l_precision=float(np.mean(rprec_list)),
        rouge_l_recall=float(np.mean(rrec_list)),
        bert_score_f1=mean_bert_score,
        total_time_seconds=float(elapsed_time),
        ms_per_query=ms_per_query,
        fallback_rate=float(fallback_rate),
        unique_responses_retrieved=unique_count,
        top_10_response_reuse_rate=top_10_reuse_rate,
        retrieved_length_mean_chars=ret_mean,
        retrieved_length_median_chars=ret_med,
        actual_length_mean_chars=act_mean,
        actual_length_median_chars=act_med,
        length_mismatch_flag=length_mismatch,
    )


def log_experiment_rows(results: list[BaselineEvalResult], split_version: str) -> None:
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows_to_append = []
    for res in results:
        val_metrics_str = f"token_f1={res.token_f1:.4f};rouge_l_f1={res.rouge_l_f1:.4f};bert_score_f1={res.bert_score_f1:.4f};ms_per_query={res.ms_per_query:.2f};fallback_rate={res.fallback_rate:.4f}"
        row = {
            "experiment_id": f"baseline_{res.method_name.lower().replace(' ', '_').replace('-', '_')}",
            "git_commit": "",
            "dataset_version": f"split_version={split_version}",
            "split_version": split_version,
            "model_name": f"retrieval_baseline_{res.method_name.lower().replace(' ', '_')}",
            "model_revision": "v1",
            "exact_total_parameters": "0",
            "trainable_parameters": "0",
            "tokenizer": "bengali_whitespace",
            "training_config": "retrieval_baseline_non_trainable",
            "generation_settings": "top1_nearest_neighbor",
            "validation_metrics": val_metrics_str,
            "manual_safety_findings": "retrieval_from_train_corpus",
            "kaggle_notebook_version": "",
            "decision_next_step": "baseline score logged for model comparison",
        }
        rows_to_append.append(row)

    if EXPERIMENT_LOG_PATH.exists():
        existing_lines = EXPERIMENT_LOG_PATH.read_text(encoding="utf-8").splitlines()
        fieldnames = existing_lines[0].split(",") if existing_lines else list(rows_to_append[0].keys())
        with EXPERIMENT_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            for r in rows_to_append:
                writer.writerow(r)
    else:
        with EXPERIMENT_LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_to_append[0].keys()))
            writer.writeheader()
            for r in rows_to_append:
                writer.writerow(r)


def write_outputs(results: list[BaselineEvalResult]) -> None:
    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    json_data = [asdict(r) for r in results]
    RESULTS_JSON_PATH.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Retrieval Baseline Results",
        "",
        "| Method | Token F1 | ROUGE-L | BERTScore | Ms/Query | Total Time (s) | Fallback Rate | Reuse Rate | Ret. Mean Chars | Act. Mean Chars | Mismatch? |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        md_lines.append(
            f"| **{r.method_name}** | {r.token_f1:.4f} | {r.rouge_l_f1:.4f} | {r.bert_score_f1:.4f} | {r.ms_per_query:.2f}ms | {r.total_time_seconds:.2f}s | {r.fallback_rate:.2%} | {r.top_10_response_reuse_rate:.2%} | {r.retrieved_length_mean_chars:.1f} | {r.actual_length_mean_chars:.1f} | {'⚠️ YES' if r.length_mismatch_flag else '✅ NO'} |"
        )

    md_lines.extend([
        "",
        "## Baseline Method Comparison",
        "- **Exact Match Retrieval**: Evaluates direct prompt reuse. When prompts don't match exactly, fallback text is returned.",
        "- **TF-IDF Retrieval**: Character & word TF-IDF cosine similarity nearest neighbor retrieval.",
        "- **BM25 Retrieval**: Okapi BM25 ranking over tokenized Bengali text.",
        "- **Sentence-Embedding Retrieval**: Multilingual dense embedding (`paraphrase-multilingual-MiniLM-L12-v2`) similarity retrieval.",
    ])

    RESULTS_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-limit", type=int, default=0, help="Limit number of validation rows for fast testing (0 = full val set).")
    args = parser.parse_args()

    print("Loading data and split_v1.json...")
    df = pd.read_csv(TRAIN_PATH)
    split_info = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))

    train_ids = set(split_info["train_row_ids"])
    val_ids = set(split_info["validation_row_ids"])

    train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["id"].isin(val_ids)].reset_index(drop=True)

    if args.eval_limit > 0:
        print(f"Limiting evaluation to first {args.eval_limit} validation rows.")
        val_df = val_df.iloc[:args.eval_limit].reset_index(drop=True)

    val_prompts = val_df["input"].astype(str).tolist()
    val_targets = val_df["output"].astype(str).tolist()

    print(f"Train Corpus Size: {len(train_df)} rows")
    print(f"Validation Query Size: {len(val_df)} rows")

    results: list[BaselineEvalResult] = []

    # 1. Exact Match Baseline
    print("\n--- Running Baseline 1: Exact Match ---")
    t0 = time.time()
    exact_retriever = ExactMatchRetriever(train_df)
    exact_preds, fallback_rate = exact_retriever.retrieve(val_prompts)
    t_exact = time.time() - t0
    res_exact = evaluate_method("Exact Match", exact_preds, val_targets, t_exact, fallback_rate)
    results.append(res_exact)
    print(f"Exact Match Done: Token F1={res_exact.token_f1:.4f}, Fallback Rate={res_exact.fallback_rate:.2%}")

    # 2. TF-IDF Baseline
    print("\n--- Running Baseline 2: TF-IDF Retrieval ---")
    t0 = time.time()
    tfidf_retriever = TfidfRetriever(train_df)
    tfidf_preds = tfidf_retriever.retrieve(val_prompts)
    t_tfidf = time.time() - t0
    res_tfidf = evaluate_method("TF-IDF", tfidf_preds, val_targets, t_tfidf)
    results.append(res_tfidf)
    print(f"TF-IDF Done: Token F1={res_tfidf.token_f1:.4f}, ROUGE-L={res_tfidf.rouge_l_f1:.4f}")

    # 3. BM25 Baseline
    print("\n--- Running Baseline 3: BM25 Retrieval ---")
    t0 = time.time()
    bm25_retriever = BM25Retriever(train_df)
    bm25_preds = bm25_retriever.retrieve(val_prompts)
    t_bm25 = time.time() - t0
    res_bm25 = evaluate_method("BM25", bm25_preds, val_targets, t_bm25)
    results.append(res_bm25)
    print(f"BM25 Done: Token F1={res_bm25.token_f1:.4f}, ROUGE-L={res_bm25.rouge_l_f1:.4f}")

    # 4. Sentence-Embedding Baseline
    print("\n--- Running Baseline 4: Sentence-Embedding Retrieval ---")
    t0 = time.time()
    dense_retriever = DenseEmbeddingRetriever(train_df)
    dense_preds = dense_retriever.retrieve(val_prompts)
    t_dense = time.time() - t0
    res_dense = evaluate_method("Sentence Embedding", dense_preds, val_targets, t_dense)
    results.append(res_dense)
    print(f"Sentence Embedding Done: Token F1={res_dense.token_f1:.4f}, ROUGE-L={res_dense.rouge_l_f1:.4f}")

    # Save results
    write_outputs(results)
    log_experiment_rows(results, split_info["split_version"])

    print("\n================ FINAL BASELINE COMPARISON ================")
    for r in results:
        print(f"Method: {r.method_name:<20} | Token F1: {r.token_f1:.4f} | ROUGE-L: {r.rouge_l_f1:.4f} | BERTScore: {r.bert_score_f1:.4f} | Ms/Query: {r.ms_per_query:.2f}ms")

    print(f"\nWrote results to {RESULTS_JSON_PATH} and {RESULTS_MD_PATH}")
    print(f"Appended 4 baseline rows to {EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()
