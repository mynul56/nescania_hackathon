from pathlib import Path

import pandas as pd

from src.build_fixed_split import SEED, normalize_text

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"


def test_normalize_text():
    assert normalize_text("  হেলো   ডাক্তার  ") == "হেলো ডাক্তার"
    assert normalize_text(None) == ""


def test_raw_train_exists_and_read_only():
    assert TRAIN_PATH.exists()
    df = pd.read_csv(TRAIN_PATH)
    assert len(df) == 108954
    assert set(df.columns) == {"id", "input", "output"}


def test_seed_constant():
    assert SEED == 42

