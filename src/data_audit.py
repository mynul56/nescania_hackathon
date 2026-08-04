"""Data audit entry point for the Bengali medical dialogue dataset."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
CONFIG_PATH = ROOT / "config" / "competition_info.yaml"
OUTPUT_DIR = ROOT / "outputs"
REPORT_MD_PATH = OUTPUT_DIR / "data_audit.md"
REPORT_JSON_PATH = OUTPUT_DIR / "data_audit.json"
STATUS_PATH = OUTPUT_DIR / "data_audit_status.txt"

BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"[0-9]")
WHITESPACE_RE = re.compile(r"\s+")

RISK_KEYWORDS = [
    "জরুরি",
    "emergency",
    "রক্তপাত",
    "bleeding",
    "বুক ব্যথা",
    "chest pain",
    "শ্বাসকষ্ট",
    "difficulty breathing",
    "অজ্ঞান",
    "faint",
    "স্ট্রোক",
    "stroke",
    "heart attack",
    "হার্ট অ্যাটাক",
    "convulsion",
    "খিঁচুনি",
    "suicid",
    "আত্মহত্যা",
    "poison",
    "বিষ",
    "pregnant",
    "pregnancy",
    "গর্ভবতী",
    "জ্বর",
    "pain",
    "ব্যথা",
]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip().lower()


def normalize_series(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str)
    codes, uniques = pd.factorize(values, sort=False)
    normalized_uniques = [normalize_text(value) for value in uniques]
    normalized = np.asarray(normalized_uniques, dtype=object)[codes]
    return pd.Series(normalized, index=series.index)


def safe_utf8_row_count(series: pd.Series) -> dict[str, int]:
    encodable = 0
    non_encodable = 0
    for value in series.fillna("").astype(str):
        try:
            value.encode("utf-8")
            encodable += 1
        except UnicodeEncodeError:
            non_encodable += 1
    return {
        "utf8_encodable_rows": encodable,
        "non_encodable_rows": non_encodable,
        "total_rows": int(len(series)),
    }


def text_quality(series: pd.Series) -> dict[str, Any]:
    values = series.fillna("").astype(str)
    char_counts = values.str.len()
    word_counts = values.str.split().map(len)
    total_chars = char_counts.replace(0, np.nan)
    bangla_counts = values.map(lambda item: len(BANGLA_RE.findall(item)))
    latin_counts = values.map(lambda item: len(LATIN_RE.findall(item)))
    digit_counts = values.map(lambda item: len(DIGIT_RE.findall(item)))

    return {
        "row_count": int(len(values)),
        "unique_values": int(values.nunique(dropna=False)),
        "char_stats": {
            "min": int(char_counts.min()),
            "mean": float(char_counts.mean()),
            "median": float(char_counts.median()),
            "p95": float(char_counts.quantile(0.95)),
            "max": int(char_counts.max()),
        },
        "word_stats": {
            "min": int(word_counts.min()),
            "mean": float(word_counts.mean()),
            "median": float(word_counts.median()),
            "p95": float(word_counts.quantile(0.95)),
            "max": int(word_counts.max()),
        },
        "script_quality": {
            "avg_bangla_script_ratio": float((bangla_counts / total_chars).fillna(0).mean()),
            "avg_latin_ratio": float((latin_counts / total_chars).fillna(0).mean()),
            "avg_digit_ratio": float((digit_counts / total_chars).fillna(0).mean()),
            "rows_with_any_bangla": int((bangla_counts > 0).sum()),
            "rows_with_no_bangla": int((bangla_counts == 0).sum()),
        },
        "utf8_sanity": safe_utf8_row_count(series),
    }


def top_tokens(series: pd.Series, limit: int = 50) -> list[tuple[str, int]]:
    token_counts: Counter[str] = Counter()
    for text in series.fillna("").astype(str):
        token_counts.update(re.findall(r"[\u0980-\u09FF]+|[A-Za-z]+|\d+", text.lower()))
    return token_counts.most_common(limit)


def infer_text_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, str, list[str], list[str]]:
    train_object_columns = [column for column in train.columns if train[column].dtype == "object"]
    test_object_columns = [column for column in test.columns if test[column].dtype == "object"]

    name_prompt_candidates = [
        column
        for column in train.columns
        if any(token in column.lower() for token in ["prompt", "question", "query", "input", "symptom", "patient"])
    ]
    name_response_candidates = [
        column
        for column in train.columns
        if any(token in column.lower() for token in ["response", "answer", "reply", "output", "doctor"])
    ]

    prompt_column = name_prompt_candidates[0] if name_prompt_candidates else None
    response_column = name_response_candidates[0] if name_response_candidates else None

    if prompt_column is None or response_column is None:
        object_columns = [column for column in train.columns if train[column].dtype == "object"]
        if len(object_columns) == 1:
            prompt_column = object_columns[0]
            response_column = object_columns[0]
        else:
            length_order = sorted(
                object_columns,
                key=lambda column: train[column].fillna("").astype(str).str.len().mean(),
            )
            if prompt_column is None and length_order:
                prompt_column = length_order[0]
            if response_column is None and len(length_order) > 1:
                response_column = length_order[-1]

    if prompt_column is None:
        raise ValueError("Unable to infer prompt column from train.csv")
    if response_column is None:
        response_column = prompt_column

    return prompt_column, response_column, train_object_columns, test_object_columns


def duplicate_groups(series: pd.Series) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in series.items():
        normalized = normalize_text(value)
        if normalized:
            groups[normalized].append(int(index))
    return groups


def row_statistics(series: pd.Series) -> dict[str, Any]:
    values = series.fillna("").astype(str)
    char_counts = values.str.len()
    word_counts = values.str.split().map(len)
    return {
        "count": int(len(values)),
        "missing": int(series.isna().sum()),
        "unique": int(values.nunique(dropna=False)),
        "chars": {
            "min": int(char_counts.min()),
            "mean": float(char_counts.mean()),
            "median": float(char_counts.median()),
            "p95": float(char_counts.quantile(0.95)),
            "max": int(char_counts.max()),
        },
        "words": {
            "min": int(word_counts.min()),
            "mean": float(word_counts.mean()),
            "median": float(word_counts.median()),
            "p95": float(word_counts.quantile(0.95)),
            "max": int(word_counts.max()),
        },
    }


def risk_flags(series: pd.Series) -> dict[str, int]:
    hits: Counter[str] = Counter()
    for text in series.fillna("").astype(str).drop_duplicates():
        lower = text.lower()
        for keyword in RISK_KEYWORDS:
            if keyword.lower() in lower:
                hits[keyword] += 1
    return dict(hits.most_common())


def conflict_summary(train: pd.DataFrame, prompt_column: str, response_column: str) -> dict[str, Any]:
    prompt_values = train[prompt_column].fillna("").astype(str)
    response_values = train[response_column].fillna("").astype(str)
    frame = pd.DataFrame({"prompt": prompt_values, "response": response_values})

    grouped = frame.groupby("prompt", sort=False)
    prompt_map = {
        prompt: [int(index) for index in row_indexes]
        for prompt, row_indexes in grouped.indices.items()
        if prompt
    }

    response_nunique = grouped["response"].nunique(dropna=False)
    exact_conflicts: dict[str, dict[str, Any]] = {}
    for prompt, unique_response_count in response_nunique.items():
        if prompt and unique_response_count > 1:
            row_indexes = prompt_map.get(prompt, [])
            exact_conflicts[prompt] = {
                "row_count": int(len(row_indexes)),
                "unique_responses": int(unique_response_count),
                "example_rows": row_indexes[:10],
            }

    return {
        "duplicate_prompt_groups": {prompt: row_indexes for prompt, row_indexes in prompt_map.items() if len(row_indexes) > 1},
        "conflicting_prompt_groups": exact_conflicts,
    }


def embedding_near_duplicates(
    train_prompts: pd.Series,
    test_prompts: pd.Series,
    threshold: float = 0.90,
) -> dict[str, Any]:
    train_count = len(train_prompts)
    train_lengths = train_prompts.fillna("").astype(str).str.len().to_numpy()
    test_lengths = test_prompts.fillna("").astype(str).str.len().to_numpy()

    if train_count == 0 or len(test_prompts) == 0:
        return {
            "method": "hashing_char_wb",
            "threshold": threshold,
            "pairs": [],
            "status": "insufficient_rows",
        }

    all_lengths = np.concatenate([train_lengths, test_lengths])
    min_length = int(all_lengths.min())
    max_length = int(all_lengths.max())
    if min_length == max_length:
        bins = np.array([min_length - 1, max_length + 1])
    else:
        bins = np.linspace(min_length, max_length + 1, num=6)

    train_bins = pd.cut(train_lengths, bins=bins, labels=False, include_lowest=True, duplicates="drop")
    test_bins = pd.cut(test_lengths, bins=bins, labels=False, include_lowest=True, duplicates="drop")
    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
    )

    max_train_candidates_per_bucket = 1000
    pairs = []

    for bucket in sorted(pd.Series(test_bins).dropna().unique()):
        bucket = int(bucket)
        train_mask = train_bins == bucket
        test_mask = test_bins == bucket
        if not train_mask.any() or not test_mask.any():
            continue

        train_positions = np.flatnonzero(train_mask)
        if len(train_positions) > max_train_candidates_per_bucket:
            bucket_center = (bins[bucket] + bins[bucket + 1]) / 2 if bucket + 1 < len(bins) else bins[bucket]
            bucket_train_lengths = train_lengths[train_positions]
            keep_order = np.argsort(np.abs(bucket_train_lengths - bucket_center))[:max_train_candidates_per_bucket]
            train_positions = train_positions[keep_order]

        bucket_test_positions = np.flatnonzero(test_mask)
        bucket_train_texts = normalize_series(train_prompts.iloc[train_positions])
        bucket_test_texts = normalize_series(test_prompts.iloc[bucket_test_positions])
        bucket_texts = pd.concat([bucket_train_texts, bucket_test_texts], ignore_index=True)
        bucket_matrix = vectorizer.transform(bucket_texts.tolist())
        bucket_train_embeddings = bucket_matrix[: len(bucket_train_texts)]
        bucket_test_embeddings = bucket_matrix[len(bucket_train_texts) :]
        nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
        nn.fit(bucket_train_embeddings)
        distances, indices = nn.kneighbors(bucket_test_embeddings)

        for local_test_index, (distance, train_index) in enumerate(zip(distances[:, 0], indices[:, 0])):
            score = float(1.0 - distance)
            if score >= threshold:
                pairs.append(
                    {
                        "test_row": int(bucket_test_positions[local_test_index]),
                        "train_row": int(train_positions[train_index]),
                        "score": score,
                    }
                )

    return {
        "method": "hashing_char_wb_embeddings",
        "threshold": threshold,
        "pairs": pairs,
        "status": "ok",
    }


def write_config_updates(prompt_column: str, response_column: str) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    updated = {
        "competition_input_directory": "data/raw",
        "train_filename": "train.csv",
        "test_filename": "test.csv",
        "sample_submission_filename": config.get("sample_submission_filename", "TBD"),
        "id_column": config.get("id_column", "TBD"),
        "prompt_column": prompt_column,
        "response_column": response_column,
        "submission_prediction_column": response_column,
    }

    if config != updated:
        CONFIG_PATH.write_text(yaml.safe_dump(updated, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return updated


def mark_status(message: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(message + "\n", encoding="utf-8")


def build_report(train: pd.DataFrame, test: pd.DataFrame, prompt_column: str, response_column: str, train_object_columns: list[str], test_object_columns: list[str]) -> dict[str, Any]:
    train_prompt_text = train[prompt_column].fillna("").astype(str)
    test_prompt_text = test[prompt_column].fillna("").astype(str)
    train_response_text = train[response_column].fillna("").astype(str)
    mark_status("build_report:text_columns_loaded")

    train_prompt_norm = train_prompt_text
    test_prompt_norm = test_prompt_text
    train_response_norm = train_response_text
    train_prompt_lookup = set(train_prompt_norm[train_prompt_norm != ""].drop_duplicates())
    exact_cross_mask = test_prompt_norm.isin(train_prompt_lookup)
    exact_cross_prompt_overlap = int(exact_cross_mask.sum())
    exact_cross_pairs = []
    prompt_to_train_rows = train_prompt_norm[train_prompt_norm != ""].groupby(train_prompt_norm[train_prompt_norm != ""]).indices
    for test_index, value in test_prompt_norm[exact_cross_mask].items():
        if value in prompt_to_train_rows:
            exact_cross_pairs.append(
                {
                    "test_row": int(test_index),
                    "prompt": value,
                    "train_rows": prompt_to_train_rows[value][:10],
                }
            )
    mark_status("build_report:exact_duplicates_done")

    train_full_row_duplicates = int(train.duplicated().sum())
    test_full_row_duplicates = int(test.duplicated().sum())
    train_prompt_duplicates = int(train_prompt_norm.duplicated().sum())
    test_prompt_duplicates = int(test_prompt_norm.duplicated().sum())

    prompt_conflicts = conflict_summary(train, prompt_column, response_column)

    train_unique_prompt_rows = (
        train.assign(_normalized_prompt=train_prompt_norm)
        .loc[lambda frame: frame["_normalized_prompt"] != ""]
        .drop_duplicates(subset=["_normalized_prompt"], keep="first")
    )
    test_unique_prompt_rows = (
        test.assign(_normalized_prompt=test_prompt_norm)
        .loc[lambda frame: frame["_normalized_prompt"] != ""]
        .drop_duplicates(subset=["_normalized_prompt"], keep="first")
    )
    mark_status(f"build_report:unique_prompts train={len(train_unique_prompt_rows)} test={len(test_unique_prompt_rows)}")

    near_duplicate_summary = embedding_near_duplicates(
        train_unique_prompt_rows[prompt_column],
        test_unique_prompt_rows[prompt_column],
        threshold=0.90,
    )
    near_duplicate_pairs = near_duplicate_summary["pairs"]
    mark_status(f"build_report:near_duplicates_done pairs={len(near_duplicate_pairs)}")

    train_unique_prompt_series = train_unique_prompt_rows[prompt_column].fillna("").astype(str)
    train_unique_response_series = (
        train.assign(_normalized_response=train_response_norm)
        .loc[lambda frame: frame["_normalized_response"] != ""]
        .drop_duplicates(subset=["_normalized_response"], keep="first")[response_column]
        .fillna("")
        .astype(str)
    )
    test_unique_prompt_series = test_unique_prompt_rows[prompt_column].fillna("").astype(str)
    mark_status("build_report:unique_text_series_ready")

    repeated_responses = train[response_column].fillna("").astype(str).value_counts()
    unique_response_ratio = float(train[response_column].fillna("").astype(str).nunique() / len(train)) if len(train) else 0.0

    category_candidates = [
        column
        for column in train.columns
        if any(token in column.lower() for token in ["category", "specialty", "department", "field"])
    ]

    report = {
        "source_files": {
            "train_filename": TRAIN_PATH.name,
            "test_filename": TEST_PATH.name,
            "sample_submission_filename": None,
        },
        "inferred_columns": {
            "prompt_column": prompt_column,
            "response_column": response_column,
        },
        "schema": {
            "train": {
                "columns": list(train.columns),
                "dtypes": {column: str(dtype) for column, dtype in train.dtypes.items()},
                "row_count": int(len(train)),
            },
            "test": {
                "columns": list(test.columns),
                "dtypes": {column: str(dtype) for column, dtype in test.dtypes.items()},
                "row_count": int(len(test)),
            },
        },
        "missing_values": {
            "train": {column: int(value) for column, value in train.isna().sum().items()},
            "test": {column: int(value) for column, value in test.isna().sum().items()},
        },
        "duplicates": {
            "train_full_row_duplicates": train_full_row_duplicates,
            "test_full_row_duplicates": test_full_row_duplicates,
            "train_prompt_duplicates": train_prompt_duplicates,
            "test_prompt_duplicates": test_prompt_duplicates,
            "exact_cross_split_prompt_overlap_count": exact_cross_prompt_overlap,
            "exact_cross_split_pairs": exact_cross_pairs[:200],
            "near_duplicate_method": near_duplicate_summary["method"],
            "near_duplicate_threshold": near_duplicate_summary["threshold"],
            "near_cross_split_pairs": near_duplicate_pairs[:200],
            "train_conflicting_prompt_groups": prompt_conflicts["conflicting_prompt_groups"],
            "train_duplicate_prompt_groups": prompt_conflicts["duplicate_prompt_groups"],
        },
        "text_statistics": {
            "train_prompt": row_statistics(train[prompt_column]),
            "train_response": row_statistics(train[response_column]),
            "test_prompt": row_statistics(test[prompt_column]),
        },
        "vocabulary": {
            "prompt_unique_tokens": int(len({token for token, _ in top_tokens(train_unique_prompt_series, limit=10_000)})),
            "prompt_top_tokens": top_tokens(train_unique_prompt_series, limit=50),
            "response_unique_tokens": int(len({token for token, _ in top_tokens(train_unique_response_series, limit=10_000)})),
            "response_top_tokens": top_tokens(train_unique_response_series, limit=50),
        },
        "language_quality": {
            "train_prompt": text_quality(train[prompt_column]),
            "train_response": text_quality(train[response_column]),
            "test_prompt": text_quality(test[prompt_column]),
        },
        
        "response_diversity": {
            "unique_response_ratio": unique_response_ratio,
            "responses_seen_2plus": int((repeated_responses >= 2).sum()),
            "responses_seen_5plus": int((repeated_responses >= 5).sum()),
            "top_repeated_responses": repeated_responses.head(25).to_dict(),
        },
        "medical_category": {
            "available_columns": category_candidates,
            "note": "No obvious category/specialty field detected" if not category_candidates else "Category-like fields detected",
        },
        "split_recommendation": {
            "strategy": "Fixed seed, content-aware split by prompt text with exact and near-duplicate groups kept together, length-bucket stratification, and category stratification if a category field exists.",
            "seed": 42,
            "grouping_key": prompt_column,
            "length_buckets": "Quantile buckets from prompt length, with response length as a secondary diagnostic.",
            "leakage_guard": "Re-check exact and embedding-near duplicates across the final train/validation boundary before freezing the split.",
        },
        "retrieval_suitability": {
            "assessment": "High if responses are repetitive or templated; exact-match and BM25 should be strong baselines, with embedding retrieval useful for paraphrases and lexical drift.",
            "signals": {
                "duplicate_prompt_groups": len(prompt_conflicts["duplicate_prompt_groups"]),
                "conflicting_prompt_groups": len(prompt_conflicts["conflicting_prompt_groups"]),
                "near_duplicate_cross_split_pairs": len(near_duplicate_pairs),
            },
        },
        "safety_risk_surface": {
            "train_prompt_keyword_hits": risk_flags(train_unique_prompt_series),
            "train_response_keyword_hits": risk_flags(train_unique_response_series),
            "test_prompt_keyword_hits": risk_flags(test_unique_prompt_series),
            "purpose": "Triage aid for manual review, not a diagnostic classifier.",
        },
        "notes": {
            "train_object_columns": train_object_columns,
            "test_object_columns": test_object_columns,
            "sample_submission_present": (ROOT / "data" / "raw" / "sample_submission.csv").exists(),
        },
    }

    return report


def render_markdown(report: dict[str, Any]) -> str:
    schema_train = report["schema"]["train"]
    schema_test = report["schema"]["test"]
    duplicates = report["duplicates"]
    split = report["split_recommendation"]
    retrieval = report["retrieval_suitability"]
    safety = report["safety_risk_surface"]

    lines = [
        "# Data Audit",
        "",
        "## Source Files",
        f"- train: `{report['source_files']['train_filename']}`",
        f"- test: `{report['source_files']['test_filename']}`",
        f"- sample submission present: `{report['notes']['sample_submission_present']}`",
        "",
        "## Inferred Columns",
        f"- prompt column: `{report['inferred_columns']['prompt_column']}`",
        f"- response column: `{report['inferred_columns']['response_column']}`",
        "",
        "## Schema",
        f"- train rows: `{schema_train['row_count']}`",
        f"- test rows: `{schema_test['row_count']}`",
        f"- train columns: `{', '.join(schema_train['columns'])}`",
        f"- test columns: `{', '.join(schema_test['columns'])}`",
        "",
        "## Duplicate Findings",
        f"- train full-row duplicates: `{duplicates['train_full_row_duplicates']}`",
        f"- test full-row duplicates: `{duplicates['test_full_row_duplicates']}`",
        f"- train prompt duplicates: `{duplicates['train_prompt_duplicates']}`",
        f"- test prompt duplicates: `{duplicates['test_prompt_duplicates']}`",
        f"- exact cross-split prompt overlap: `{duplicates['exact_cross_split_prompt_overlap_count']}`",
        f"- near-duplicate method: `{duplicates['near_duplicate_method']}`",
        f"- near-duplicate threshold: `{duplicates['near_duplicate_threshold']}`",
        f"- near cross-split pairs found: `{len(duplicates['near_cross_split_pairs'])}`",
        f"- conflicting prompt groups in train: `{len(duplicates['train_conflicting_prompt_groups'])}`",
        "",
        "## Split Recommendation",
        f"- strategy: {split['strategy']}",
        f"- seed: `{split['seed']}`",
        f"- grouping key: `{split['grouping_key']}`",
        f"- leakage guard: {split['leakage_guard']}",
        "",
        "## Retrieval Suitability",
        f"- assessment: {retrieval['assessment']}",
        f"- duplicate prompt groups: `{retrieval['signals']['duplicate_prompt_groups']}`",
        f"- conflicting prompt groups: `{retrieval['signals']['conflicting_prompt_groups']}`",
        f"- near-duplicate cross-split pairs: `{retrieval['signals']['near_duplicate_cross_split_pairs']}`",
        "",
        "## Safety Surface",
        f"- prompt keyword hits: `{safety['train_prompt_keyword_hits']}`",
        f"- response keyword hits: `{safety['train_response_keyword_hits']}`",
        f"- test prompt keyword hits: `{safety['test_prompt_keyword_hits']}`",
        "",
        "## Notes",
        f"- category-like columns: `{report['medical_category']['available_columns']}`",
        f"- category note: {report['medical_category']['note']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mark_status("starting")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    mark_status(f"loaded_csv train={len(train)} test={len(test)}")

    prompt_column, response_column, train_object_columns, test_object_columns = infer_text_columns(train, test)
    config_update = write_config_updates(prompt_column, response_column)
    mark_status(f"config_updated prompt={prompt_column} response={response_column}")
    report = build_report(train, test, prompt_column, response_column, train_object_columns, test_object_columns)
    mark_status("report_built")
    report["config_update"] = config_update

    REPORT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    mark_status("finished")

    print(json.dumps(
        {
            "prompt_column": prompt_column,
            "response_column": response_column,
            "train_rows": len(train),
            "test_rows": len(test),
            "config_updated": config_update,
            "markdown_report": str(REPORT_MD_PATH),
            "json_report": str(REPORT_JSON_PATH),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
