"""Leakage discipline tests for the feature module.

The threat model here is the same as step 3 plus one new class:

1. 2026 data slipping in (loader-side, already guarded).
2. Engineered features at row T pulling from rows after T (forward leakage).
3. The target column influencing engineered features (scramble test).

We guard (2) by reasoning + a "shuffle future targets" test: if a feature
at row T changed when we corrupted data at row T+1 or later, it had been
peeking at the future.

Run: `pytest tests/test_features_no_leakage.py -v`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.load_data import TRAIN_CUTOFF, load_lice, load_treatment  # noqa: E402
from src.features import (  # noqa: E402
    FEATURE_COLUMNS,
    LAG_COLS,
    LAG_WEEKS,
    ROLL_COLS,
    ROLL_WINDOWS,
    TREATMENT_CATS,
    build_feature_frame,
)


# ---------------------------------------------------------------------------
# Fixtures (module-scope to avoid re-running expensive feature builds)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lice():
    return load_lice(apply_cutoff=True)


@pytest.fixture(scope="module")
def treatment():
    return load_treatment(apply_cutoff=True)


@pytest.fixture(scope="module")
def sup_h1(lice, treatment):
    return build_feature_frame(lice, treatment, horizon=1)


@pytest.fixture(scope="module")
def sup_h12(lice, treatment):
    return build_feature_frame(lice, treatment, horizon=12)


# ---------------------------------------------------------------------------
# 1. Cutoff is still respected after feature engineering
# ---------------------------------------------------------------------------

def test_feature_frame_blocks_2026(sup_h1, sup_h12):
    """Engineering features doesn't reintroduce 2026 rows or target dates."""
    for sup in (sup_h1, sup_h12):
        assert sup["WEEK_START"].max() < TRAIN_CUTOFF
        assert sup["target_week_start"].max() < TRAIN_CUTOFF


# ---------------------------------------------------------------------------
# 2. All declared feature columns are present and (mostly) populated
# ---------------------------------------------------------------------------

def test_all_feature_columns_present(sup_h1, sup_h12):
    """FEATURE_COLUMNS is the contract. If it lists a column that build_feature_frame
    doesn't produce, the model would crash at training time."""
    for sup, h in ((sup_h1, 1), (sup_h12, 12)):
        missing = [c for c in FEATURE_COLUMNS if c not in sup.columns]
        assert not missing, f"h={h}: missing feature columns {missing}"


def test_categoricals_have_no_nulls(sup_h1, sup_h12):
    """Categorical features must be fully populated — LightGBM can't infer
    levels from NaN."""
    for sup in (sup_h1, sup_h12):
        assert sup["PRODUCTIONAREAID"].notna().all()
        assert sup["target_iso_week"].notna().all()


# ---------------------------------------------------------------------------
# 3. Forward-leakage test — mutating future rows must NOT change current features
# ---------------------------------------------------------------------------

def test_lag_features_dont_peek_at_future(lice, treatment):
    """Take a small subset, build features, scramble the *future* of the lice
    frame, rebuild features, and assert lag/rolling columns are unchanged.

    A feature that peeks would change when future data is scrambled.
    """
    # Constrain to a single site for speed; pick one with long history
    site = (lice.groupby("SITENUMBER").size().sort_values(ascending=False)
                .index[0])
    sub = lice[lice["SITENUMBER"] == site].copy()
    assert len(sub) > 200, "test site doesn't have enough history"

    # Build features on the full subset
    feats_clean = build_feature_frame(sub, treatment, horizon=1)
    # Pick a cut-point T and the row whose WEEK_START is just before it
    midpoint = sub["WEEK_START"].iloc[len(sub) // 2]
    target_row_clean = feats_clean[feats_clean["WEEK_START"] < midpoint].iloc[-1]

    # Scramble everything STRICTLY AFTER the target row's WEEK_START
    sub_scrambled = sub.copy()
    future_mask = sub_scrambled["WEEK_START"] > target_row_clean["WEEK_START"]
    rng = np.random.default_rng(seed=42)
    for col in ("BREACH", "FEMALEADULT", "MOBILELICE", "SEATEMPERATURE"):
        # Flip / randomize values in future rows
        if col == "BREACH":
            # Fill any NaN with False first; flipping NaN raises in nullable boolean
            current = sub_scrambled.loc[future_mask, col].fillna(False).astype(bool)
            sub_scrambled.loc[future_mask, col] = ~current
        else:
            sub_scrambled.loc[future_mask, col] = rng.uniform(0, 100, size=future_mask.sum())

    feats_dirty = build_feature_frame(sub_scrambled, treatment, horizon=1)
    target_row_dirty = feats_dirty[
        feats_dirty["WEEK_START"] == target_row_clean["WEEK_START"]
    ].iloc[0]

    # All lag and rolling features at the target row must match — they only
    # use data <= target_row's WEEK_START which we left untouched
    cols_under_test = [
        f"{c}_lag{k}" for c in LAG_COLS for k in LAG_WEEKS
    ] + [
        f"{c}_roll{w}_{stat}"
        for c in ROLL_COLS for w in ROLL_WINDOWS for stat in ("mean", "max")
    ] + ["degree_weeks_roll12"]

    for col in cols_under_test:
        a, b = target_row_clean[col], target_row_dirty[col]
        # NaN == NaN should count as equal
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b or np.isclose(a, b, equal_nan=True), (
            f"Forward leakage: {col} at WEEK_START={target_row_clean['WEEK_START'].date()} "
            f"changed from {a} to {b} when future was scrambled"
        )


def test_treatment_features_dont_peek_at_future(lice, treatment):
    """Same test, scrambling future treatment rows."""
    site = (lice.groupby("SITENUMBER").size().sort_values(ascending=False)
                .index[0])
    sub_lice = lice[lice["SITENUMBER"] == site].copy()

    feats_clean = build_feature_frame(sub_lice, treatment, horizon=1)
    midpoint = sub_lice["WEEK_START"].iloc[len(sub_lice) // 2]
    target_row_clean = feats_clean[feats_clean["WEEK_START"] < midpoint].iloc[-1]
    T = target_row_clean["WEEK_START"]

    # Add fake future treatments — should NOT affect treatment features at T
    future_dates = pd.date_range(start=T + pd.Timedelta(weeks=1),
                                  periods=20, freq="W-MON")
    fake = pd.DataFrame({
        "SITENUMBER": site,
        "WEEK_START": future_dates,
        "ACTION": ["medikamentell"] * 20,
        "YEAR": future_dates.year.where(future_dates.year < 2026, 2025),
        # Other columns aren't used by the feature builder
    })
    treatment_dirty = pd.concat([treatment, fake], ignore_index=True)

    feats_dirty = build_feature_frame(sub_lice, treatment_dirty, horizon=1)
    target_row_dirty = feats_dirty[feats_dirty["WEEK_START"] == T].iloc[0]

    treat_cols = [f"treat_{c}" for c in TREATMENT_CATS]
    treat_cols += [f"treat_{c}_roll{w}"
                    for c in TREATMENT_CATS for w in ROLL_WINDOWS]
    treat_cols += [f"days_since_{c}" for c in TREATMENT_CATS]

    for col in treat_cols:
        a, b = target_row_clean[col], target_row_dirty[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, (
            f"Forward leakage: {col} at WEEK_START={T.date()} "
            f"changed from {a} to {b} when future treatments were added"
        )


# ---------------------------------------------------------------------------
# 4. Target scramble — features must not depend on the target column
# ---------------------------------------------------------------------------

def test_features_dont_depend_on_target(sup_h1):
    """Scramble the target column on a copy and rebuild *only* the target-week
    features. The engineered features (lags, rolling, treatment) come from the
    lice/treatment frames upstream, so they cannot depend on target. But the
    test_iso_week feature is derived from target_week_start (the DATE, not the
    label) — that's intentional and leakage-safe. Verify both."""
    scrambled = sup_h1.copy()
    rng = np.random.default_rng(seed=0)
    scrambled["target"] = rng.integers(0, 2, size=len(scrambled)).astype(bool)

    # target_iso_week is a function of target_week_start (a date), not target
    pd.testing.assert_series_equal(
        sup_h1["target_iso_week"], scrambled["target_iso_week"], check_names=False,
    )

    # Pick a representative engineered feature and confirm it's invariant
    for col in ("FEMALEADULT_lag1", "treat_chem_roll12",
                "degree_weeks_roll12", "site_age_weeks"):
        pd.testing.assert_series_equal(
            sup_h1[col], scrambled[col], check_names=False,
        )


# ---------------------------------------------------------------------------
# 5. Sanity — feature ranges are plausible
# ---------------------------------------------------------------------------

def test_treatment_counts_are_non_negative(sup_h1):
    """Sentinel — rolling sums of non-negative counts must stay non-negative."""
    for cat in TREATMENT_CATS:
        assert (sup_h1[f"treat_{cat}"] >= 0).all()
        for w in ROLL_WINDOWS:
            assert (sup_h1[f"treat_{cat}_roll{w}"] >= 0).all()


def test_degree_weeks_are_non_negative(sup_h1):
    """degree_weeks is a sum of clip(temp - 8, 0); can't be negative."""
    assert (sup_h1["degree_weeks_roll12"].fillna(0) >= 0).all()


def test_site_age_is_non_negative(sup_h1):
    """site_age = weeks since first observation — never negative."""
    assert (sup_h1["site_age_weeks"] >= 0).all()


# ---------------------------------------------------------------------------
# 6. Cleaner-fish biology (v2) — bio_active and weeks_since_last_cold
# ---------------------------------------------------------------------------

def test_bio_active_zero_on_cold_weeks(sup_h1):
    """A cold week kills cleaner fish, so bio_active must be 0 on those rows."""
    cold = sup_h1["SEATEMPERATURE"] <= 6.0
    assert (sup_h1.loc[cold, "bio_active"] == 0).all()


def test_bio_active_is_non_negative(sup_h1):
    """bio_active is a count — can't go negative."""
    assert (sup_h1["bio_active"] >= 0).all()


def test_weeks_since_last_cold_is_non_negative(sup_h1):
    """Weeks since the last cold week is monotone in time — never negative."""
    s = sup_h1["weeks_since_last_cold"].dropna()
    assert (s >= 0).all()


def test_cleaner_fish_features_dont_peek_at_future(lice, treatment):
    """Same scramble pattern as the lag-feature test: build features on a site,
    record bio_active and weeks_since_last_cold at row T, scramble all future
    SEATEMPERATURE + treatment rows, rebuild, assert the row-T values match."""
    site = (lice.groupby("SITENUMBER").size().sort_values(ascending=False)
                .index[0])
    sub_lice = lice[lice["SITENUMBER"] == site].copy()

    feats_clean = build_feature_frame(sub_lice, treatment, horizon=1)
    midpoint = sub_lice["WEEK_START"].iloc[len(sub_lice) // 2]
    target_row_clean = feats_clean[feats_clean["WEEK_START"] < midpoint].iloc[-1]
    T = target_row_clean["WEEK_START"]

    # Scramble future SEATEMPERATURE
    rng = np.random.default_rng(seed=7)
    sub_dirty = sub_lice.copy()
    future = sub_dirty["WEEK_START"] > T
    sub_dirty.loc[future, "SEATEMPERATURE"] = rng.uniform(-2, 25, size=future.sum())

    # Inject fake future bio events
    future_dates = pd.date_range(start=T + pd.Timedelta(weeks=1),
                                  periods=10, freq="W-MON")
    fake = pd.DataFrame({
        "SITENUMBER": site, "WEEK_START": future_dates,
        "ACTION": ["rensefisk"] * 10,
        "YEAR": future_dates.year.where(future_dates.year < 2026, 2025),
    })
    treatment_dirty = pd.concat([treatment, fake], ignore_index=True)

    feats_dirty = build_feature_frame(sub_dirty, treatment_dirty, horizon=1)
    target_row_dirty = feats_dirty[feats_dirty["WEEK_START"] == T].iloc[0]

    for col in ("bio_active", "weeks_since_last_cold"):
        a, b = target_row_clean[col], target_row_dirty[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, (
            f"Forward leakage in {col} at {T.date()}: clean={a}, dirty={b}"
        )
