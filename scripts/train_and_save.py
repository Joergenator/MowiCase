"""Train and persist LightGBM boosters for horizons 1, 2, 12.

Trains TWO model variants per horizon:
  - v1: original step-4 feature set (52 features)
  - v3: v1 + neighbor-site spatial-diffusion features (step 6 extension)

Writes the resulting boosters to `models/lgbm_{v1,v3}_h{h}.txt` so the
agent can load them without retraining.

The training split matches notebook 03: WEEK_START.year <= 2024, with
2024 held out internally for early stopping. 2025 is untouched (final
holdout).

We build the feature frame ONCE per horizon and fit both v1 and v3 on
the same frame with different `feature_cols` — saves redundant feature
engineering.

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
    FEATURE_COLUMNS_V1, FEATURE_COLUMNS_V3, CATEGORICAL_FEATURES,
    build_feature_frame,
)
from src.load_data import load_training_data
from src.models import LightGBMBreach
from src.utils import train_test_split_by_year


HORIZONS = (1, 2, 12)
MODELS_DIR = ROOT / "models"

# (version_tag, feature_cols, display_name)
VARIANTS = (
    ("v1", FEATURE_COLUMNS_V1, "LightGBM v1"),
    ("v3", FEATURE_COLUMNS_V3, "LightGBM v3 (neighbor features)"),
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("Loading training data...")
    lice, treat = load_training_data()
    print(f"  lice={len(lice):,} rows, treatment={len(treat):,} rows")

    for h in HORIZONS:
        t_h = time.time()
        print(f"\n=== h={h}w ===")
        print(f"  building feature frame (this includes neighbor features)...")
        t_feat = time.time()
        sup = build_feature_frame(lice, treat, horizon=h)
        train, _ = train_test_split_by_year(sup)
        print(f"  feature frame built in {time.time() - t_feat:.1f}s  "
              f"(train={len(train):,} rows, {len(sup.columns)} cols)")

        for tag, feature_cols, display in VARIANTS:
            t0 = time.time()
            print(f"\n  fitting {display} with tuning...")
            m = LightGBMBreach(
                name=display, horizon=h, tune=True,
                feature_cols=tuple(feature_cols),
                cat_cols=tuple(CATEGORICAL_FEATURES),
            ).fit(train)
            out = MODELS_DIR / f"lgbm_{tag}_h{h}.txt"
            m.save(out)
            print(f"  saved {out.name}  "
                  f"(inner-val PR-AUC={m.inner_val_pr_auc_:.4f}, "
                  f"best_iter={m.best_iter_}, {time.time() - t0:.1f}s)")
        print(f"  h={h} total: {time.time() - t_h:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
