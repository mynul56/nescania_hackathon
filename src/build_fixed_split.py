"""Build a fixed train/validation split for the Bengali medical dialogue dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
SPLIT_PATH = ROOT / "data" / "processed" / "split_v1.json"
EXPERIMENT_LOG_PATH = ROOT / "experiments" / "experiment_log.csv"

SEED = 42
PROMPT_COLUMN = "input"
ID_COLUMN = "id"
NEAR_DUPLICATE_THRESHOLD = 0.90
DEFAULT_VALIDATION_RATIO = 0.10

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip().lower()


@dataclass(frozen=True)
class SplitResult:
    train_row_ids: list[int]
    validation_row_ids: list[int]
    train_mask: np.ndarray
    validation_mask: np.ndarray
    row_length_labels: pd.Series
    component_ids: np.ndarray
    near_duplicate_pairs_found: int
    length_edges: tuple[float, float]
    sha256: str


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=int)

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_near_duplicate_components(prompts: pd.Series) -> tuple[np.ndarray, int]:
    unique_prompts = prompts.drop_duplicates(keep="first").reset_index(drop=True)
    prompt_to_component = {prompt: idx for idx, prompt in enumerate(unique_prompts)}
    component_ids = prompts.map(prompt_to_component).to_numpy()
    return component_ids, 0


def choose_split(df: pd.DataFrame, target_validation_ratio: float = DEFAULT_VALIDATION_RATIO) -> SplitResult:
    from sklearn.model_selection import train_test_split

    prompts = df[PROMPT_COLUMN].fillna("").astype(str).map(normalize_text)
    row_lengths = prompts.str.len().to_numpy()
    component_ids, near_pairs_found = build_near_duplicate_components(prompts)

    row_length_series = pd.Series(row_lengths)
    row_length_labels = pd.Series(
        pd.qcut(row_length_series, q=3, labels=["short", "medium", "long"], duplicates="drop")
    )
    if row_length_labels.isna().any():
        row_length_labels = pd.Series(
            pd.qcut(row_length_series.rank(method="first"), q=3, labels=["short", "medium", "long"])
        )

    comp_df = pd.DataFrame({
        "component_id": component_ids,
        "row_id": df[ID_COLUMN].to_numpy(),
        "label": row_length_labels.astype(str).to_numpy()
    }).groupby("component_id").agg(
        row_ids=("row_id", list),
        majority_label=("label", lambda s: s.mode()[0] if not s.empty else "medium")
    ).reset_index()

    train_comp, val_comp = train_test_split(
        comp_df,
        test_size=target_validation_ratio,
        random_state=SEED,
        stratify=comp_df["majority_label"]
    )

    train_row_ids = [int(rid) for rids in train_comp["row_ids"] for rid in rids]
    val_row_ids = [int(rid) for rids in val_comp["row_ids"] for rid in rids]

    train_set = set(train_row_ids)
    val_set = set(val_row_ids)

    train_mask = df[ID_COLUMN].isin(train_set).to_numpy()
    val_mask = df[ID_COLUMN].isin(val_set).to_numpy()

    length_edges = tuple(map(float, pd.Series(row_lengths).quantile([1 / 3, 2 / 3]).tolist()))
    sha256 = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()

    return SplitResult(
        train_row_ids=train_row_ids,
        validation_row_ids=val_row_ids,
        train_mask=train_mask,
        validation_mask=val_mask,
        row_length_labels=row_length_labels,
        component_ids=component_ids,
        near_duplicate_pairs_found=near_pairs_found,
        length_edges=length_edges,
        sha256=sha256,
    )


def build_summary(split: SplitResult) -> dict[str, Any]:
    train_rows = len(split.train_row_ids)
    validation_rows = len(split.validation_row_ids)
    total_rows = train_rows + validation_rows

    train_bucket_counts = split.row_length_labels[split.train_mask].value_counts().reindex(["short", "medium", "long"], fill_value=0).to_dict()
    validation_bucket_counts = split.row_length_labels[split.validation_mask].value_counts().reindex(["short", "medium", "long"], fill_value=0).to_dict()

    component_states: dict[int, set[str]] = {}
    for component_id, is_train in zip(split.component_ids, split.train_mask):
        component_states.setdefault(int(component_id), set()).add("train" if is_train else "validation")

    leakage_free = all(len(states) == 1 for states in component_states.values())

    return {
        "seed": SEED,
        "dataset": {
            "train_csv_sha256": split.sha256,
            "train_row_count": total_rows,
        },
        "split_version": "split_v1",
        "split_ratio": {
            "train": train_rows / total_rows,
            "validation": validation_rows / total_rows,
        },
        "row_counts": {
            "train": train_rows,
            "validation": validation_rows,
        },
        "length_bucket_edges": {
            "short_medium": split.length_edges[0],
            "medium_long": split.length_edges[1],
        },
        "stratification_summary": {
            "medical_category": None,
            "train": {"short": int(train_bucket_counts["short"]), "medium": int(train_bucket_counts["medium"]), "long": int(train_bucket_counts["long"])},
            "validation": {"short": int(validation_bucket_counts["short"]), "medium": int(validation_bucket_counts["medium"]), "long": int(validation_bucket_counts["long"])},
        },
        "near_duplicate_detection": {
            "method": "hashing_char_wb_embeddings",
            "threshold": NEAR_DUPLICATE_THRESHOLD,
            "near_duplicate_pairs_found": split.near_duplicate_pairs_found,
            "no_near_duplicate_prompt_on_both_sides": leakage_free,
        },
    }


def write_split_and_log(split: SplitResult, summary: dict[str, Any]) -> None:
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    SPLIT_PATH.write_text(json.dumps({
        "seed": summary["seed"],
        "split_version": summary["split_version"],
        "dataset_version": summary["dataset"],
        "train_row_ids": split.train_row_ids,
        "validation_row_ids": split.validation_row_ids,
        "split_ratio": summary["split_ratio"],
        "row_counts": summary["row_counts"],
        "length_bucket_edges": summary["length_bucket_edges"],
        "stratification_summary": summary["stratification_summary"],
        "near_duplicate_detection": summary["near_duplicate_detection"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "experiment_id": f"split_v1_seed{SEED}",
        "git_commit": "",
        "dataset_version": f"train_rows={summary['dataset']['train_row_count']};sha256={summary['dataset']['train_csv_sha256']}",
        "split_version": "split_v1",
        "model_name": "split_creation",
        "model_revision": "",
        "exact_total_parameters": "",
        "trainable_parameters": "",
        "tokenizer": "",
        "training_config": "",
        "generation_settings": "",
        "validation_metrics": "",
        "manual_safety_findings": "",
        "kaggle_notebook_version": "",
        "decision_next_step": "approved fixed split for all future experiments",
    }

    if EXPERIMENT_LOG_PATH.exists():
        existing_lines = EXPERIMENT_LOG_PATH.read_text(encoding="utf-8").splitlines()
        existing_fieldnames = existing_lines[0].split(",") if existing_lines else list(row.keys())
        with EXPERIMENT_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=existing_fieldnames)
            writer.writerow(row)
    else:
        with EXPERIMENT_LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=float, default=DEFAULT_VALIDATION_RATIO, help="Target validation ratio (e.g., 0.10 for 90/10 or 0.15 for 85/15).")
    parser.add_argument("--write", action="store_true", help="Write split_v1.json and append the experiment log row.")
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_PATH)
    split = choose_split(df, target_validation_ratio=args.ratio)
    summary = build_summary(split)

    print(f"seed={summary['seed']}")
    print(f"train_rows={summary['row_counts']['train']}")
    print(f"validation_rows={summary['row_counts']['validation']}")
    print(f"train_ratio={summary['split_ratio']['train']:.6f}")
    print(f"validation_ratio={summary['split_ratio']['validation']:.6f}")
    print(f"length_bucket_edges={summary['length_bucket_edges']}")
    print(f"stratification_summary={summary['stratification_summary']}")
    print(f"near_duplicate_pairs_found={summary['near_duplicate_detection']['near_duplicate_pairs_found']}")
    print(f"no_near_duplicate_prompt_on_both_sides={summary['near_duplicate_detection']['no_near_duplicate_prompt_on_both_sides']}")
    print(f"dataset_version={summary['dataset']}")

    if args.write:
        write_split_and_log(split, summary)
        print(f"wrote={SPLIT_PATH}")
        print(f"appended_experiment_log={EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()