"""Standalone leakage verification — run this to prove no 2026 data is used.

Usage: py src/check_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.load_data import TRAIN_CUTOFF, load_lice, load_treatment, assert_no_leakage  # noqa: E402


def main() -> int:
    print(f"Leakage check — TRAIN_CUTOFF = {TRAIN_CUTOFF.date()}")

    lice = load_lice(apply_cutoff=True)
    treat = load_treatment(apply_cutoff=True)

    assert_no_leakage(lice)
    assert_no_leakage(treat)

    print(f"  lice:      max WEEK_START = {lice['WEEK_START'].max().date()}  ✓")
    print(f"  treatment: max WEEK_START = {treat['WEEK_START'].max().date()}  ✓")

    full = load_lice(apply_cutoff=False)
    n_excluded = (full["WEEK_START"] >= TRAIN_CUTOFF).sum()
    print(f"  Rows excluded by cutoff: {n_excluded}")

    print("\nPASS: No 2026 data is present in training datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
