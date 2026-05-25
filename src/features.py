"""Feature engineering for the LightGBM breach-prediction model.

The contract is the same as step 3: every feature on row T must be derivable
from data at WEEK_START <= T. Nothing from T+1 (the target's neighbourhood)
may leak in. The two traps to watch:

1. `shift(+k)` within a sorted groupby gives the value k ROWS ago. If a site
   has missing weeks, two rows apart can be 4+ calendar weeks apart.
   `make_supervised_frame` already enforces strict-week gaps on the target;
   we *don't* re-enforce on lag features because LightGBM is happy with NaN
   and a sparse site's gappy lags are still legal predictors.
2. `rolling(window)` over a sorted group naturally includes the current row;
   that's fine — FEMALEADULT(T) is observable at predict time T.

The expensive joins (treatment) and rolling reductions happen once on the
full lice frame; then `make_supervised_frame` applies the target shift on
top. Order matters — see `build_feature_frame` at the bottom.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lag features — same site, k weeks ago
# ---------------------------------------------------------------------------

LAG_WEEKS = (1, 2, 4, 8)
LAG_COLS = ("BREACH", "FEMALEADULT", "MOBILELICE", "SEATEMPERATURE")


def add_lag_features(lice: pd.DataFrame) -> pd.DataFrame:
    """Add `{col}_lag{k}` columns for col in LAG_COLS, k in LAG_WEEKS."""
    out = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()
    grouped = out.groupby("SITENUMBER", sort=False)
    for col in LAG_COLS:
        for k in LAG_WEEKS:
            out[f"{col}_lag{k}"] = grouped[col].shift(k)
    # Cast boolean lags to float so LightGBM treats NaN as missing
    for k in LAG_WEEKS:
        out[f"BREACH_lag{k}"] = out[f"BREACH_lag{k}"].astype("float64")
    return out


# ---------------------------------------------------------------------------
# Rolling stats — trailing window ending at row T (inclusive)
# ---------------------------------------------------------------------------

ROLL_WINDOWS = (4, 8, 12)
ROLL_COLS = ("FEMALEADULT", "MOBILELICE")


def add_rolling_features(lice: pd.DataFrame) -> pd.DataFrame:
    """Add `{col}_roll{w}_{mean|max}` over the trailing w weeks (inclusive)."""
    out = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()
    grouped = out.groupby("SITENUMBER", sort=False)
    for col in ROLL_COLS:
        for w in ROLL_WINDOWS:
            roll = grouped[col].rolling(w, min_periods=1)
            out[f"{col}_roll{w}_mean"] = roll.mean().reset_index(level=0, drop=True)
            out[f"{col}_roll{w}_max"] = roll.max().reset_index(level=0, drop=True)
    return out


# ---------------------------------------------------------------------------
# Cumulative degree-weeks above 8°C — biological proxy for lice generation time
# ---------------------------------------------------------------------------

DEGREE_BASE = 8.0
DEGREE_WINDOW = 12


def add_degree_weeks(lice: pd.DataFrame) -> pd.DataFrame:
    """Sum of max(SEATEMP - 8, 0) over the trailing 12 weeks, per site."""
    out = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()
    above = (out["SEATEMPERATURE"] - DEGREE_BASE).clip(lower=0)
    out["_above_base"] = above
    out["degree_weeks_roll12"] = (
        out.groupby("SITENUMBER", sort=False)["_above_base"]
           .rolling(DEGREE_WINDOW, min_periods=1)
           .sum()
           .reset_index(level=0, drop=True)
    )
    return out.drop(columns=["_above_base"])


# ---------------------------------------------------------------------------
# Treatment features — categorised, with rolling counts and days-since-last
# ---------------------------------------------------------------------------

# Map raw ACTION values to three categories matching the EDA chart-10 split.
# 'ikke-medikamentell' is the 2024+ umbrella for mechanical/thermal methods,
# so we fold it into 'mech'.
TREATMENT_CATEGORY = {
    "medikamentell": "chem",
    "rensefisk": "bio",
    "mekanisk fjerning": "mech",
    "ikke-medikamentell": "mech",
}
TREATMENT_CATS = ("chem", "mech", "bio")


def _treatment_pivot(treatment: pd.DataFrame) -> pd.DataFrame:
    """Pivot raw treatment events to one row per (SITENUMBER, WEEK_START, category)."""
    df = treatment[["SITENUMBER", "WEEK_START", "ACTION"]].copy()
    df["category"] = df["ACTION"].map(TREATMENT_CATEGORY)
    df = df.dropna(subset=["category", "WEEK_START"])
    pivot = (df.groupby(["SITENUMBER", "WEEK_START", "category"])
               .size()
               .unstack("category", fill_value=0)
               .reset_index())
    for cat in TREATMENT_CATS:
        if cat not in pivot.columns:
            pivot[cat] = 0
        pivot = pivot.rename(columns={cat: f"treat_{cat}"})
    return pivot


def add_treatment_features(lice: pd.DataFrame, treatment: pd.DataFrame) -> pd.DataFrame:
    """Add per-category treatment counts (current week + rolling 4/8/12) and
    days-since-last per category. Sites with no treatment history get 0 counts
    and NaN days-since (LightGBM handles NaN)."""
    out = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()
    pivot = _treatment_pivot(treatment)
    out = out.merge(pivot, on=["SITENUMBER", "WEEK_START"], how="left")
    for cat in TREATMENT_CATS:
        col = f"treat_{cat}"
        out[col] = out[col].fillna(0).astype(int)

    # Rolling counts (sum of events in the trailing w weeks, per site)
    grouped = out.groupby("SITENUMBER", sort=False)
    for cat in TREATMENT_CATS:
        col = f"treat_{cat}"
        for w in ROLL_WINDOWS:
            out[f"{col}_roll{w}"] = (grouped[col]
                                     .rolling(w, min_periods=1).sum()
                                     .reset_index(level=0, drop=True)
                                     .astype(int))

    # Days since last event in this category, per site. Forward-fill the
    # event-week date along the time axis, then subtract the row's WEEK_START.
    for cat in TREATMENT_CATS:
        col = f"treat_{cat}"
        event_week = out["WEEK_START"].where(out[col] > 0)
        last_event = event_week.groupby(out["SITENUMBER"], sort=False).ffill()
        out[f"days_since_{cat}"] = (out["WEEK_START"] - last_event).dt.days

    return out


# ---------------------------------------------------------------------------
# Cleaner-fish biology (v2) — survival depends on sea temperature
# ---------------------------------------------------------------------------

# Cleaner fish (lumpfish, ballan wrasse) die when SEATEMP drops below ~6°C, so
# a `rensefisk` stocking event from May has no biological effect by late
# autumn. LightGBM can in principle learn this interaction from treat_bio +
# SEATEMP alone, but encoding it explicitly:
#   (a) makes the feature interpretable in the deck;
#   (b) gives the tree splits a cleaner boundary to find.
# Threshold chosen at 6°C — below this both lumpfish and wrasse mortality is
# severe. NaN SEATEMP is treated as "non-warm" (defensive — we don't know).
COLD_THRESHOLD = 6.0


def add_cleaner_fish_features(lice: pd.DataFrame) -> pd.DataFrame:
    """Add `bio_active` and `weeks_since_last_cold`.

    Must run AFTER `add_treatment_features` so the `treat_bio` column exists.

    bio_active(T) = number of `treat_bio` events at warm weeks W <= T such
    that no cold week occurred in (W, T]. On a cold/missing-SEATEMP week,
    bio_active is 0 (the cold week itself kills any stocked fish).

    weeks_since_last_cold(T) = weeks since the most recent cold week, per
    site. NaN if no cold week in the site's history (rare in Norway).
    """
    if "treat_bio" not in lice.columns:
        raise ValueError(
            "add_cleaner_fish_features requires `treat_bio` — call "
            "add_treatment_features first."
        )
    out = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()

    # "Cold or missing" = water too cold for cleaner fish OR temperature unknown
    is_cold = ~(out["SEATEMPERATURE"] > COLD_THRESHOLD).fillna(False)

    # Each cold week starts a new "run" via cumsum. Warm weeks after a cold
    # week share their run-id with the cold week that preceded them. Bio
    # events ON a cold week itself are forced to 0 below (those fish died).
    run_id = is_cold.astype(int).groupby(out["SITENUMBER"], sort=False).cumsum()

    treat_bio_alive = out["treat_bio"].where(~is_cold, 0)
    out["bio_active"] = (treat_bio_alive
                         .groupby([out["SITENUMBER"], run_id]).cumsum()
                         .astype(int))
    # Belt-and-braces: cold-week rows get 0 (a stocking event ON the cold
    # week would have produced a non-zero where() above, so we explicitly
    # zero it here too).
    out.loc[is_cold, "bio_active"] = 0

    # Weeks since the last cold week — proxy for "how long has the warm
    # period lasted" (= how long can cleaner fish have been viable).
    cold_dates = out["WEEK_START"].where(is_cold)
    last_cold = cold_dates.groupby(out["SITENUMBER"], sort=False).ffill()
    out["weeks_since_last_cold"] = (out["WEEK_START"] - last_cold).dt.days // 7

    return out


# ---------------------------------------------------------------------------
# Site cohort context
# ---------------------------------------------------------------------------

def add_site_age(lice: pd.DataFrame) -> pd.DataFrame:
    """Add `site_age_weeks`: weeks since this site's first observation.
    A weak proxy for whether the site is established or new."""
    out = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()
    first = out.groupby("SITENUMBER")["WEEK_START"].transform("min")
    out["site_age_weeks"] = ((out["WEEK_START"] - first).dt.days // 7).astype(int)
    return out


# ---------------------------------------------------------------------------
# Target-week features — derived from target_week_start (a date, not a label)
# ---------------------------------------------------------------------------

def add_target_week_features(sup: pd.DataFrame) -> pd.DataFrame:
    """Add features from the target's week-of-year (the dimension B3 exploits).

    Computed on the supervised frame, after `make_supervised_frame` has run.
    target_week_start is just a date, so this is leakage-safe — we never
    touch the target *label*."""
    out = sup.copy()
    out["target_iso_week"] = out["target_week_start"].dt.isocalendar().week.astype(int)
    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

# Stable list of feature columns the model consumes. Anchor here so the
# notebook, model, and tests can't drift out of sync. Two named feature sets:
# V1 is the original step-4 set; V2 adds the cleaner-fish biology features.
# Both are derived once on every call to build_feature_frame — selecting V1
# vs V2 is a choice the model makes at instantiation time.
FEATURE_COLUMNS_V1 = (
    # Current-row observables (already on the lice frame)
    "FEMALEADULT", "MOBILELICE", "PERSISTENTLICE", "SEATEMPERATURE",
    "LICELIMITWEEK",  # the regime — 0.5 or 0.2
    # Structural
    "PRODUCTIONAREAID", "target_iso_week",
    "site_age_weeks",
    # Lags
    *(f"{c}_lag{k}" for c in LAG_COLS for k in LAG_WEEKS),
    # Rolling
    *(f"{c}_roll{w}_{stat}"
      for c in ROLL_COLS for w in ROLL_WINDOWS for stat in ("mean", "max")),
    # Cumulative thermal exposure
    "degree_weeks_roll12",
    # Treatment counts (current + rolling)
    *(f"treat_{cat}" for cat in TREATMENT_CATS),
    *(f"treat_{cat}_roll{w}" for cat in TREATMENT_CATS for w in ROLL_WINDOWS),
    # Days since last treatment
    *(f"days_since_{cat}" for cat in TREATMENT_CATS),
)

# V2 adds the cleaner-fish-biology features.
FEATURE_COLUMNS_V2 = FEATURE_COLUMNS_V1 + (
    "bio_active",
    "weeks_since_last_cold",
)

# Default alias for backwards compatibility with code that imports the
# unversioned name (tests, the original model wrapper).
FEATURE_COLUMNS = FEATURE_COLUMNS_V2

CATEGORICAL_FEATURES = ("PRODUCTIONAREAID", "target_iso_week")


def build_inference_frame(
    lice: pd.DataFrame,
    treatment: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Build features for live prediction — one row per site at its latest week.

    Used by the LLM agent's `predict_risk` tool. Unlike `build_feature_frame`,
    we do NOT shift a target backwards (there is no future label), so the
    latest weeks of training data — which `make_supervised_frame` would drop
    because they have no target — are kept here. The `target_iso_week`
    feature is computed from `WEEK_START + horizon` to match what the model
    saw at training time.
    """
    aug = add_lag_features(lice)
    aug = add_rolling_features(aug)
    aug = add_degree_weeks(aug)
    aug = add_treatment_features(aug, treatment)
    aug = add_cleaner_fish_features(aug)
    aug = add_site_age(aug)

    latest = (aug.sort_values(["SITENUMBER", "WEEK_START"])
                 .groupby("SITENUMBER", sort=False).tail(1).copy())
    latest["target_week_start"] = latest["WEEK_START"] + pd.to_timedelta(horizon * 7, unit="D")
    latest["target_iso_week"] = latest["target_week_start"].dt.isocalendar().week.astype(int)

    latest = latest.dropna(subset=["PRODUCTIONAREAID"]).copy()
    latest["PRODUCTIONAREAID"] = latest["PRODUCTIONAREAID"].astype(int)
    return latest.reset_index(drop=True)


def build_feature_frame(
    lice: pd.DataFrame,
    treatment: pd.DataFrame,
    horizon: int,
    counted_only: bool = True,
) -> pd.DataFrame:
    """Build a supervised frame for `horizon` with all engineered features.

    Order matters:
      1. Engineer features on the FULL lice frame (so lag/rolling see the
         complete history, including weeks that will be dropped later).
      2. Apply `make_supervised_frame` (negative shift + week-gap filter).
      3. Add target-week features derived from target_week_start.
    """
    from src.utils import make_supervised_frame  # local import to avoid cycle

    aug = add_lag_features(lice)
    aug = add_rolling_features(aug)
    aug = add_degree_weeks(aug)
    aug = add_treatment_features(aug, treatment)
    aug = add_cleaner_fish_features(aug)  # must follow add_treatment_features
    aug = add_site_age(aug)
    sup = make_supervised_frame(aug, horizon=horizon, counted_only=counted_only)
    sup = add_target_week_features(sup)

    # Drop rows missing the PO id (~13 rows in the full frame). LightGBM
    # requires populated categoricals; a sentinel value would create a fake
    # PO category that the model would learn spurious patterns on.
    sup = sup.dropna(subset=["PRODUCTIONAREAID"]).copy()
    sup["PRODUCTIONAREAID"] = sup["PRODUCTIONAREAID"].astype(int)
    return sup.reset_index(drop=True)
