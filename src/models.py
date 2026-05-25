"""LightGBM model for breach prediction.

Same fit/predict_proba interface as `src.baselines.Baseline` so the evaluation
harness in `src.evaluation` works unchanged.

Training discipline:
- One model per horizon (h=1, 2, 12). Each is fit on its own supervised frame
  from `src.features.build_feature_frame`.
- The model uses `WEEK_START.year == 2024` as inner validation for early
  stopping. Training uses 2012-2023. The final 2025 evaluation never appears
  during fit — guarded by `train_test_split_by_year` upstream.
- Class imbalance (~4.5% positives) is handled implicitly by tree splits +
  PR-AUC as the early-stopping metric. We deliberately do NOT use
  `scale_pos_weight` or `is_unbalance` because they inflate predicted
  probabilities and wreck calibration (Brier, count-MAE). The case's
  primary metric is PR-AUC (rank-only) but the LLM agent in step 5 needs
  calibrated probabilities — so we trade a tiny bit of PR-AUC for usable
  outputs.

Tuning is intentionally light: a small grid (num_leaves, min_data_in_leaf,
learning_rate) chosen by inner-validation PR-AUC. The case is about
disciplined modelling, not squeezing the last 0.5% out of LightGBM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baselines import Baseline
from src.features import FEATURE_COLUMNS, CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Default LightGBM config — modest depth, conservative regularization
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
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

# A tiny grid for inner-validation selection. Kept short so the full run
# (3 horizons × 4 settings) is bounded to single-digit minutes on a laptop.
TUNING_GRID = [
    {"num_leaves": 31,  "min_data_in_leaf": 100},
    {"num_leaves": 63,  "min_data_in_leaf": 200},
    {"num_leaves": 127, "min_data_in_leaf": 500},
    {"num_leaves": 63,  "min_data_in_leaf": 50, "learning_rate": 0.03},
]

INNER_VAL_YEAR = 2024
EARLY_STOP_ROUNDS = 50
MAX_ROUNDS = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(df: pd.DataFrame, feature_cols: Iterable[str],
                   cat_cols: Iterable[str]) -> lgb.Dataset:
    """Build a LightGBM Dataset with feature names + categoricals registered."""
    return lgb.Dataset(
        df[list(feature_cols)],
        label=df["target"].astype(int),
        categorical_feature=list(cat_cols),
        free_raw_data=False,
    )


def _split_inner_validation(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training frame into pre-2024 training and 2024 inner validation.

    `train_df` is everything with WEEK_START.year <= 2024 (the input to fit()).
    Inner validation gives us early stopping and tuning without touching 2025.
    """
    inner_train = train_df[train_df["WEEK_START"].dt.year < INNER_VAL_YEAR].copy()
    inner_val = train_df[train_df["WEEK_START"].dt.year == INNER_VAL_YEAR].copy()
    return inner_train, inner_val


# ---------------------------------------------------------------------------
# Model — sklearn-like wrapper conforming to the Baseline interface
# ---------------------------------------------------------------------------

@dataclass
class LightGBMBreach(Baseline):
    """LightGBM binary classifier for breach prediction.

    Parameters
    ----------
    horizon
        Forecast horizon (1, 2, or 12). Stored for bookkeeping.
    tune
        If True, run the small TUNING_GRID and pick the best by inner-val PR-AUC.
        If False, train once with DEFAULT_PARAMS.
    feature_cols, cat_cols
        Override the feature set (mostly useful for ablations).
    """
    name: str = "LightGBM"
    horizon: int = 1
    tune: bool = True
    feature_cols: tuple = field(default_factory=lambda: tuple(FEATURE_COLUMNS))
    cat_cols: tuple = field(default_factory=lambda: tuple(CATEGORICAL_FEATURES))

    # Set during fit
    booster_: lgb.Booster | None = None
    best_params_: dict | None = None
    best_iter_: int | None = None
    inner_val_pr_auc_: float | None = None

    def fit(self, train_df: pd.DataFrame) -> "LightGBMBreach":
        inner_train, inner_val = _split_inner_validation(train_df)
        if len(inner_val) == 0:
            raise ValueError(
                "No 2024 rows in training frame — inner validation requires "
                "the default train cutoff (year <= 2024)."
            )

        if self.tune:
            best = None  # (pr_auc, params, booster, best_iter)
            for override in TUNING_GRID:
                params = {**DEFAULT_PARAMS, **override}
                booster, score, best_iter = self._fit_once(
                    params, inner_train, inner_val,
                )
                if best is None or score > best[0]:
                    best = (score, params, booster, best_iter)
            score, params, booster, best_iter = best
            self.best_params_ = params
            self.inner_val_pr_auc_ = float(score)
        else:
            params = {**DEFAULT_PARAMS}
            booster, score, best_iter = self._fit_once(
                params, inner_train, inner_val,
            )
            self.best_params_ = params
            self.inner_val_pr_auc_ = float(score)

        # Refit on the FULL training frame (inner_train + inner_val) for the
        # best_iter rounds we found above. This is the standard "select on
        # validation, refit on everything" pattern.
        full_dataset = _make_dataset(train_df, self.feature_cols, self.cat_cols)
        self.booster_ = lgb.train(
            self.best_params_,
            full_dataset,
            num_boost_round=best_iter,
        )
        self.best_iter_ = int(best_iter)
        return self

    def _fit_once(self, params: dict, inner_train: pd.DataFrame,
                   inner_val: pd.DataFrame) -> tuple[lgb.Booster, float, int]:
        """Train one configuration with early stopping on inner_val."""
        d_train = _make_dataset(inner_train, self.feature_cols, self.cat_cols)
        d_val = _make_dataset(inner_val, self.feature_cols, self.cat_cols)
        booster = lgb.train(
            params,
            d_train,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[d_val],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        # Inner-val PR-AUC at the best iteration
        from sklearn.metrics import average_precision_score
        y_val = inner_val["target"].astype(int).to_numpy()
        p_val = booster.predict(inner_val[list(self.feature_cols)],
                                num_iteration=booster.best_iteration)
        pr_auc = float(average_precision_score(y_val, p_val))
        return booster, pr_auc, int(booster.best_iteration)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("Model not fit — call .fit(train_df) first.")
        return self.booster_.predict(df[list(self.feature_cols)])

    def predict_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-sample feature contributions (LightGBM SHAP-style decomposition).

        Calls `booster.predict(X, pred_contrib=True)` — no external `shap`
        dependency. Returns one row per input sample with one column per
        feature plus a trailing `_bias` column. Per-row invariant:
            row.sum() = raw logit prediction
            sigmoid(row.sum()) ≈ predict_proba(df)[i]

        Used by the step-6 forecast notebook to explain "why is this site
        at risk" and by the agent's `predict_drivers` tool. DataFrame
        return (not ndarray) so callers can do named-column lookup
        ("which feature pushed this site up by the most?").
        """
        if self.booster_ is None:
            raise RuntimeError("Model not fit — call .fit(train_df) first.")
        raw = self.booster_.predict(
            df[list(self.feature_cols)], pred_contrib=True,
        )
        cols = list(self.feature_cols) + ["_bias"]
        return pd.DataFrame(raw, index=df.index, columns=cols)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Return feature importances as a Series indexed by feature name."""
        if self.booster_ is None:
            raise RuntimeError("Model not fit.")
        return (pd.Series(
            self.booster_.feature_importance(importance_type=importance_type),
            index=self.feature_cols,
            name=f"importance_{importance_type}",
        ).sort_values(ascending=False))

    # ------------------------------------------------------------------
    # Persistence — the agent in step 5 needs to call predict without refit.
    # We persist the booster as LightGBM's native text format alongside a
    # tiny JSON sidecar with the feature contract (so a reload that uses a
    # mismatched FEATURE_COLUMNS version fails loudly instead of silently).
    # ------------------------------------------------------------------
    def save(self, path: str | "Path") -> None:
        from pathlib import Path
        import json
        if self.booster_ is None:
            raise RuntimeError("Cannot save — model not fit.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(path))
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        sidecar.write_text(json.dumps({
            "name": self.name,
            "horizon": self.horizon,
            "feature_cols": list(self.feature_cols),
            "cat_cols": list(self.cat_cols),
            "best_iter": self.best_iter_,
            "inner_val_pr_auc": self.inner_val_pr_auc_,
            "best_params": self.best_params_,
        }, indent=2))

    @classmethod
    def load(cls, path: str | "Path") -> "LightGBMBreach":
        from pathlib import Path
        import json
        path = Path(path)
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(sidecar.read_text())
        obj = cls(
            name=meta["name"],
            horizon=meta["horizon"],
            tune=False,
            feature_cols=tuple(meta["feature_cols"]),
            cat_cols=tuple(meta["cat_cols"]),
        )
        obj.booster_ = lgb.Booster(model_file=str(path))
        obj.best_iter_ = meta.get("best_iter")
        obj.inner_val_pr_auc_ = meta.get("inner_val_pr_auc")
        obj.best_params_ = meta.get("best_params")
        return obj
