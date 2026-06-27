"""Build the vector index from the ASRS reports.

Run this once before asking questions:

    python scripts/01_build_index.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make `config` / `src` importable

import torch

from src.data import load_reports
from src.ingest import build_index


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build] device: {device}")
    df = load_reports()
    n = build_index(df, device=device)
    print(f"[build] done — {n} chunks indexed.")


if __name__ == "__main__":
    main()
