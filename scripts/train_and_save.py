"""Train and persist LightGBM v1 boosters for horizons 1, 2, 12.

Reproduces the same fits as notebook 03 (v1 only) and writes the resulting
boosters to `models/lgbm_v1_h{h}.txt` so the step-5 LLM agent can load them
without retraining.

The training split matches the notebook: WEEK_START.year <= 2024, with 2024
held out internally for early stopping. 2025 is untouched (final holdout).

Run:
    python -m scripts.train_and_save
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as `python scripts/train_and_save.py` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (
    FEATURE_COLUMNS_V1, CATEGORICAL_FEATURES, build_feature_frame,
)
from src.load_data import load_training_data
from src.models import LightGBMBreach
from src.utils import train_test_split_by_year


HORIZONS = (1, 2, 12)
MODELS_DIR = ROOT / "models"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("Loading training data...")
    lice, treat = load_training_data()
    print(f"  lice={len(lice):,} rows, treatment={len(treat):,} rows")

    for h in HORIZONS:
        t0 = time.time()
        sup = build_feature_frame(lice, treat, horizon=h)
        train, _ = train_test_split_by_year(sup)
        print(f"\nh={h:>2}w: train={len(train):,} rows — fitting v1 with tuning...")
        m = LightGBMBreach(
            name="LightGBM v1", horizon=h, tune=True,
            feature_cols=tuple(FEATURE_COLUMNS_V1),
            cat_cols=tuple(CATEGORICAL_FEATURES),
        ).fit(train)
        out = MODELS_DIR / f"lgbm_v1_h{h}.txt"
        m.save(out)
        print(f"  saved {out.name}  "
              f"(inner-val PR-AUC={m.inner_val_pr_auc_:.4f}, "
              f"best_iter={m.best_iter_}, total {time.time() - t0:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
