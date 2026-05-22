"""Baseline forecasters for lice-limit breach prediction.

All baselines share the same interface so they can be swapped freely:

    model = SomeBaseline().fit(train_df)
    p_hat = model.predict_proba(test_df)   # ndarray of P(breach)

The `train_df` passed to `fit` must be a supervised frame from
`src.utils.make_supervised_frame` with columns at minimum:
  - WEEK_START, SITENUMBER, PRODUCTIONAREAID, WEEK, BREACH, target

The `test_df` passed to `predict_proba` must have the same columns
(target is allowed to be present but is not used for prediction).

Why these four:
- B0 GlobalRate: a sanity floor — anything that loses to this is broken
- B1 Persistence: y(t+h) = y(t). Captures "things stay the same" — strong
  for short horizons, degrades for h=12
- B2 SeasonalNaive: y(t+h) = y(t+h-52). Captures the strong annual cycle
- B3 POWeekRate: P(breach | PO, week-of-year) from training data — the
  strongest naive because it exploits both PO and seasonal structure

B3 is the bar the LightGBM model in step 4 will actually need to beat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Base class — defines the interface
# ---------------------------------------------------------------------------

class Baseline:
    """Common interface. Subclasses implement fit() and predict_proba()."""

    name: str = "Baseline"
    horizon: int = 1

    def fit(self, train_df: pd.DataFrame) -> "Baseline":
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# B0 — Global rate
# ---------------------------------------------------------------------------

@dataclass
class GlobalRate(Baseline):
    """Predict the training-set base rate for every row.

    The trivial floor: if a model can't beat this it's worse than guessing.
    """
    name: str = "B0 GlobalRate"
    horizon: int = 1
    rate_: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "GlobalRate":
        self.rate_ = float(train_df["target"].mean())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.rate_, dtype=float)


# ---------------------------------------------------------------------------
# B1 — Persistence
# ---------------------------------------------------------------------------

@dataclass
class Persistence(Baseline):
    """y(t+h) = y(t). Use the site's CURRENT breach status as the prediction.

    Strong for h=1 (lice populations have inertia), degrades as h grows.
    """
    name: str = "B1 Persistence"
    horizon: int = 1

    def fit(self, train_df: pd.DataFrame) -> "Persistence":
        return self  # no fitting required

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # BREACH is the *current-week* breach flag in the supervised frame
        # (the row's WEEK_START). Treat True/False as 1/0 probabilities.
        return df["BREACH"].fillna(False).astype(float).to_numpy()


# ---------------------------------------------------------------------------
# B2 — Seasonal naive (same week last year, same site)
# ---------------------------------------------------------------------------

@dataclass
class SeasonalNaive(Baseline):
    """y(t+h) = y(t+h-52). Look up the breach status at the same calendar
    week one year ago for the same site.

    If the look-up has no observation (new site, sparse history), fall back
    to the training base rate so every test row has a prediction.
    """
    name: str = "B2 SeasonalNaive"
    horizon: int = 1
    fallback_rate_: float = 0.0
    history_: pd.Series = None  # type: ignore[assignment]

    def fit(self, train_df: pd.DataFrame) -> "SeasonalNaive":
        self.fallback_rate_ = float(train_df["target"].mean())
        # Build a (SITENUMBER, target_week_start) → target lookup from the
        # WHOLE supervised history. Look-up will offset target_week_start by
        # 52 weeks at predict-time.
        idx = pd.MultiIndex.from_arrays([
            train_df["SITENUMBER"].to_numpy(),
            train_df["target_week_start"].to_numpy(),
        ], names=["SITENUMBER", "target_week_start"])
        self.history_ = pd.Series(
            train_df["target"].astype(float).to_numpy(),
            index=idx,
            name="prev_year_target",
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Look up the value at target_week_start - 52 weeks
        prior_week = df["target_week_start"] - pd.Timedelta(weeks=52)
        idx = pd.MultiIndex.from_arrays([
            df["SITENUMBER"].to_numpy(),
            prior_week.to_numpy(),
        ], names=["SITENUMBER", "target_week_start"])
        looked_up = self.history_.reindex(idx)
        return looked_up.fillna(self.fallback_rate_).to_numpy()


# ---------------------------------------------------------------------------
# B3 — PO × week-of-year historical rate
# ---------------------------------------------------------------------------

@dataclass
class POWeekRate(Baseline):
    """P(breach | PO, ISO-week-of-year) computed from training data.

    The strongest naive: combines geographic structure (chart 2) with
    seasonality (chart 4). The lookup key uses the *target week's* PO and
    ISO week, since that's what we're trying to predict.
    """
    name: str = "B3 POWeekRate"
    horizon: int = 1
    fallback_rate_: float = 0.0
    table_: pd.Series = None  # type: ignore[assignment]

    def fit(self, train_df: pd.DataFrame) -> "POWeekRate":
        self.fallback_rate_ = float(train_df["target"].mean())
        # Compute rate by (PRODUCTIONAREAID, ISO week-of-target)
        target_week_iso = train_df["target_week_start"].dt.isocalendar().week
        key = pd.MultiIndex.from_arrays([
            train_df["PRODUCTIONAREAID"].to_numpy(),
            target_week_iso.to_numpy(),
        ], names=["PRODUCTIONAREAID", "iso_week"])
        self.table_ = (pd.Series(train_df["target"].astype(float).to_numpy(), index=key)
                       .groupby(level=[0, 1]).mean())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        target_week_iso = df["target_week_start"].dt.isocalendar().week
        idx = pd.MultiIndex.from_arrays([
            df["PRODUCTIONAREAID"].to_numpy(),
            target_week_iso.to_numpy(),
        ], names=["PRODUCTIONAREAID", "iso_week"])
        looked_up = self.table_.reindex(idx)
        return looked_up.fillna(self.fallback_rate_).to_numpy()


# ---------------------------------------------------------------------------
# Convenience: all baselines in one list, ready to fit
# ---------------------------------------------------------------------------

def all_baselines(horizon: int) -> list[Baseline]:
    """Return one fresh instance of each baseline, configured for the horizon."""
    return [
        GlobalRate(horizon=horizon),
        Persistence(horizon=horizon),
        SeasonalNaive(horizon=horizon),
        POWeekRate(horizon=horizon),
    ]
