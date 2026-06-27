"""Load and clean the ASRS aviation-safety reports.

The raw dataset has hundreds of coded columns; we only need the free-text
narrative. The text column is taken from config, with an automatic fallback
that picks the column holding the longest strings (the narrative) in case the
schema changes.
"""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset

import config


def _detect_text_column(df: pd.DataFrame) -> str:
    """Return the object column whose values are, on average, the longest.

    The narrative is by far the longest free-text field, so this is a robust
    way to find it even if the exact column name changes.
    """
    obj = df.select_dtypes(include="object")
    avg_len = obj.apply(lambda c: c.dropna().astype(str).str.len().mean())
    return avg_len.sort_values(ascending=False).index[0]


def load_reports(sample_size: int | None = config.SAMPLE_SIZE,
                 random_state: int = 42) -> pd.DataFrame:
    """Load the ASRS reports as a tidy DataFrame with `doc_id` and `text` columns.

    Parameters
    ----------
    sample_size : int or None
        Number of reports to keep (None keeps all). Sampling keeps iteration fast.
    random_state : int
        Seed for reproducible sampling.
    """
    df = load_dataset(config.DATASET, split="train").to_pandas()

    text_col = config.TEXT_COLUMN
    if text_col not in df.columns:
        text_col = _detect_text_column(df)
        print(f"[data] '{config.TEXT_COLUMN}' not found; using detected column '{text_col}'")

    df = df[[text_col]].rename(columns={text_col: "text"})
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    df["doc_id"] = [f"RPT-{i:05d}" for i in range(len(df))]
    print(f"[data] loaded {len(df)} reports")
    return df[["doc_id", "text"]]
