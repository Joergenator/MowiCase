"""Train cumulative breach-count regressors for weeks ∈ {2, 12}.

Complements the binary classifiers in `scripts.train_and_save` by directly
modelling the case's primary ask: "Nr of breaches X weeks ahead".

The target is the COUNT of breach weeks the site experiences in
`weeks` future weeks (0..weeks). LightGBM Poisson objective handles the
non-negative integer outcome correctly; predictions are expected counts
(mean of the Poisson process) and so map naturally to MAE / RMSE.

Trains with the v3 feature set (V1 + 8 neighbor features) since v3 won
the binary comparison at h=12 by +13% PR-AUC, and the neighbor signal is
biologically motivated for long horizons (lice drift between farms).

Run:
    python -m scripts.train_cumulative

Outputs:
    models/lgbm_cumulative_w{2,12}.txt  (+ .meta.json sidecars)
    Prints 2025-holdout MAE, RMSE, and per-PO breakdown.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS_V3, build_cumulative_feature_frame,
)
from src.load_data import load_training_data
from src.utils import train_test_split_by_year


WEEKS_TO_TRAIN = (2, 12)
MODELS_DIR = ROOT / "models"

POISSON_PARAMS = {
    "objective": "poisson",
    "metric": "mae",  # tracks MAE on inner validation
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "seed": 42,
    "deterministic": True,
}
MAX_ROUNDS = 2000
EARLY_STOP = 50
INNER_VAL_YEAR = 2024


def _make_dataset(df: pd.DataFrame, feature_cols, cat_cols) -> lgb.Dataset:
    return lgb.Dataset(
        df[list(feature_cols)],
        label=df["target"].to_numpy(dtype=float),
        categorical_feature=list(cat_cols),
        free_raw_data=False,
    )


def _train_one(weeks: int, lice: pd.DataFrame, treat: pd.DataFrame):
    print(f"\n=== weeks={weeks} ===")
    print("  building cumulative feature frame ...")
    t0 = time.time()
    sup = build_cumulative_feature_frame(lice, treat, weeks=weeks)
    train_df, test_df = train_test_split_by_year(sup)
    inner_train = train_df[train_df["WEEK_START"].dt.year < INNER_VAL_YEAR].copy()
    inner_val = train_df[train_df["WEEK_START"].dt.year == INNER_VAL_YEAR].copy()
    print(f"  built in {time.time() - t0:.1f}s  "
          f"(train={len(train_df):,} rows, test={len(test_df):,} rows; "
          f"mean target={train_df['target'].mean():.3f}, "
          f"max={int(train_df['target'].max())})")

    feature_cols = tuple(FEATURE_COLUMNS_V3)
    cat_cols = tuple(CATEGORICAL_FEATURES)

    # Inner-val for early stopping
    t1 = time.time()
    d_train = _make_dataset(inner_train, feature_cols, cat_cols)
    d_val = _make_dataset(inner_val, feature_cols, cat_cols)
    booster_inner = lgb.train(
        POISSON_PARAMS, d_train,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[d_val], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(0)],
    )
    best_iter = int(booster_inner.best_iteration)
    inner_val_mae = float(np.mean(np.abs(
        inner_val["target"].to_numpy(dtype=float) -
        booster_inner.predict(inner_val[list(feature_cols)],
                               num_iteration=best_iter)
    )))
    print(f"  inner-val MAE={inner_val_mae:.4f} at best_iter={best_iter}  "
          f"({time.time() - t1:.1f}s)")

    # Refit on full train (inner_train + inner_val)
    full_dataset = _make_dataset(train_df, feature_cols, cat_cols)
    booster_final = lgb.train(POISSON_PARAMS, full_dataset, num_boost_round=best_iter)

    # Save booster + sidecar
    out = MODELS_DIR / f"lgbm_cumulative_w{weeks}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    booster_final.save_model(str(out))
    (out.with_suffix(out.suffix + ".meta.json")).write_text(json.dumps({
        "name": f"LightGBM Cumulative (Poisson) w={weeks}",
        "weeks": weeks,
        "feature_cols": list(feature_cols),
        "cat_cols": list(cat_cols),
        "best_iter": best_iter,
        "inner_val_mae": inner_val_mae,
        "best_params": POISSON_PARAMS,
    }, indent=2))

    # 2025 holdout metrics
    y_test = test_df["target"].to_numpy(dtype=float)
    y_pred = booster_final.predict(test_df[list(feature_cols)])
    mae = float(np.mean(np.abs(y_test - y_pred)))
    mse = float(np.mean((y_test - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    naive_mae = float(np.mean(np.abs(y_test - train_df["target"].mean())))
    print(f"  2025 holdout (n={len(test_df):,}):  "
          f"MAE={mae:.3f}  RMSE={rmse:.3f}  (naive-mean MAE={naive_mae:.3f})")

    # Per-PO breakdown of MAE
    test_df = test_df.assign(_pred=y_pred, _err=np.abs(y_test - y_pred))
    per_po = (test_df.groupby("PRODUCTIONAREAID")
              .agg(n=("target", "size"),
                   mean_actual=("target", "mean"),
                   mean_pred=("_pred", "mean"),
                   mae=("_err", "mean"))
              .round(3))
    print("\n  Per-PO performance on 2025 holdout:")
    print(per_po.to_string())

    return {
        "weeks": weeks,
        "best_iter": best_iter,
        "inner_val_mae": inner_val_mae,
        "test_mae": mae,
        "test_rmse": rmse,
        "naive_mae": naive_mae,
        "n_test": int(len(test_df)),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("Loading training data ...")
    lice, treat = load_training_data()
    print(f"  lice={len(lice):,} rows, treatment={len(treat):,} rows")

    summaries = [_train_one(w, lice, treat) for w in WEEKS_TO_TRAIN]

    print("\n=== Summary ===")
    print(f"{'weeks':>6}  {'best_iter':>9}  {'inner-MAE':>9}  "
          f"{'test-MAE':>8}  {'test-RMSE':>9}  {'naive-MAE':>9}")
    for s in summaries:
        print(f"{s['weeks']:>6}  {s['best_iter']:>9}  {s['inner_val_mae']:>9.4f}  "
              f"{s['test_mae']:>8.4f}  {s['test_rmse']:>9.4f}  {s['naive_mae']:>9.4f}")


if __name__ == "__main__":
    main()
