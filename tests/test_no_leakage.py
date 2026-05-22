"""Leakage discipline tests.

What we're guarding against
---------------------------
1. A supervised frame containing rows whose source WEEK_START is in 2026 or
   later (the case's hard rule).
2. A baseline using the *target's* observation week (T+h) as part of its
   prediction inputs at row T. This is the most dangerous class of leakage
   for time-series models — the row "knows" something it shouldn't.
3. SeasonalNaive looking up future data: its 52-week look-back must reach
   into the training set, never into the holdout.

Run: `pytest tests/test_no_leakage.py -v`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.load_data import TRAIN_CUTOFF, load_lice, assert_no_leakage  # noqa: E402
from src.utils import make_supervised_frame, train_test_split_by_year  # noqa: E402
from src.baselines import (  # noqa: E402
    GlobalRate, Persistence, SeasonalNaive, POWeekRate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lice():
    """Cleaned lice frame, loaded once for the whole module."""
    return load_lice(apply_cutoff=True)


@pytest.fixture(scope="module")
def df_h1(lice):
    return make_supervised_frame(lice, horizon=1, counted_only=True)


@pytest.fixture(scope="module")
def df_h12(lice):
    return make_supervised_frame(lice, horizon=12, counted_only=True)


# ---------------------------------------------------------------------------
# 1. The supervised frame must not contain any 2026 data
# ---------------------------------------------------------------------------

def test_loader_blocks_2026(lice):
    """The base lice frame should already be cut off."""
    assert_no_leakage(lice)
    assert lice["WEEK_START"].max() < TRAIN_CUTOFF
    assert lice["YEAR"].max() < 2026


def test_supervised_frame_blocks_2026(df_h1, df_h12):
    """Even after the negative shift, no row should have WEEK_START in 2026.

    A row's WEEK_START is the prediction-time week. Its target_week_start
    may legally still be < 2026 (only weeks where the target_week_start exists
    in the source data survive the join).
    """
    for df in (df_h1, df_h12):
        assert df["WEEK_START"].max() < TRAIN_CUTOFF
        assert df["target_week_start"].max() < TRAIN_CUTOFF


# ---------------------------------------------------------------------------
# 2. The target column at row T must not be derivable from any feature at T
# ---------------------------------------------------------------------------

def test_target_horizon_is_exact(df_h1, df_h12):
    """Gap between WEEK_START and target_week_start must equal `horizon`
    weeks exactly — no row-vs-calendar drift."""
    gap_h1 = (df_h1["target_week_start"] - df_h1["WEEK_START"]).dt.days / 7
    assert (gap_h1 == 1).all(), "h=1 frame has rows with gap != 1 week"

    gap_h12 = (df_h12["target_week_start"] - df_h12["WEEK_START"]).dt.days / 7
    assert (gap_h12 == 12).all(), "h=12 frame has rows with gap != 12 weeks"


def test_target_matches_source_breach(lice, df_h1):
    """Hand-verify: for a random row, the supervised frame's `target` must
    equal the source `BREACH` at SITENUMBER, WEEK_START + horizon."""
    rng = np.random.default_rng(seed=0)
    sample = df_h1.sample(n=200, random_state=0)
    src = lice.set_index(["SITENUMBER", "WEEK_START"])["BREACH"]
    for _, row in sample.iterrows():
        key = (row["SITENUMBER"], row["target_week_start"])
        expected = src.get(key, None)
        if pd.isna(expected):
            continue  # rows where target_week is uncounted are filtered upstream
        assert bool(expected) == bool(row["target"]), (
            f"Mismatch at {key}: expected={expected}, got={row['target']}"
        )


# ---------------------------------------------------------------------------
# 3. Baselines must only see training data when fitted
# ---------------------------------------------------------------------------

def test_baselines_dont_use_test_targets(df_h1):
    """Fit on train, predict on test — verify predictions don't peek at
    test targets. We do this by mutating test targets after fitting and
    confirming predictions are unchanged."""
    train, test = train_test_split_by_year(df_h1)
    for B in (GlobalRate, Persistence, SeasonalNaive, POWeekRate):
        model = B(horizon=1).fit(train)
        p_before = model.predict_proba(test).copy()

        # Mutate test targets to garbage and re-predict
        scrambled = test.copy()
        scrambled["target"] = ~scrambled["target"]  # flip all
        p_after = model.predict_proba(scrambled)

        assert np.allclose(p_before, p_after, equal_nan=True), (
            f"{B.__name__} predictions changed when test targets were scrambled — "
            "baseline is reading the target column instead of features"
        )


def test_seasonal_naive_lookups_stay_in_training(df_h1):
    """For test predictions, SeasonalNaive looks up (site, target_week - 52w).
    Those lookup dates must land in training-era data only (≤ 2024-12-31).

    This is the genuine leakage concern. Training rows whose target falls in
    early 2025 are fine (the model learned the relationship from a 2024
    prediction-time view) — they just sit in the lookup table unused.
    """
    train, test = train_test_split_by_year(df_h1)
    model = SeasonalNaive(horizon=1).fit(train)

    # For each test row, the date the model will query
    lookup_dates = test["target_week_start"] - pd.Timedelta(weeks=52)
    assert lookup_dates.max().year <= 2024, (
        f"SeasonalNaive will query a date in {lookup_dates.max().year} when "
        "predicting on the test set — that's not training-era data"
    )


def test_seasonal_naive_no_test_targets_in_history(df_h1):
    """A stronger check: when fit on training rows only, the lookup table
    must not contain any entry whose value is a test-set target.

    This catches the worst-case bug where train and test are accidentally
    merged before fit.
    """
    train, test = train_test_split_by_year(df_h1)
    model = SeasonalNaive(horizon=1).fit(train)

    # Reconstruct keys for test-row targets
    test_keys = pd.MultiIndex.from_arrays([
        test["SITENUMBER"].to_numpy(),
        test["target_week_start"].to_numpy(),
    ], names=["SITENUMBER", "target_week_start"])
    overlap = model.history_.index.intersection(test_keys)
    assert len(overlap) == 0, (
        f"SeasonalNaive history contains {len(overlap)} entries whose key "
        "matches a test-set target — train/test must have leaked"
    )


# ---------------------------------------------------------------------------
# 4. Train/test split itself is clean
# ---------------------------------------------------------------------------

def test_train_test_split_is_disjoint_in_time(df_h1):
    """Test rows must all be in 2025; train rows must all be ≤ 2024."""
    train, test = train_test_split_by_year(df_h1)
    assert (train["WEEK_START"].dt.year <= 2024).all()
    assert (test["WEEK_START"].dt.year == 2025).all()


# ---------------------------------------------------------------------------
# Sanity: there are enough test rows to compute meaningful metrics
# ---------------------------------------------------------------------------

def test_test_set_has_enough_rows(df_h1):
    _, test = train_test_split_by_year(df_h1)
    assert len(test) > 1000, f"only {len(test)} test rows — split likely broken"
    n_pos = int(test["target"].sum())
    assert n_pos > 100, f"only {n_pos} positives in test — evaluation will be noisy"


# ---------------------------------------------------------------------------
# 5. Structural integrity: lice frame is one row per site-week
# ---------------------------------------------------------------------------
# The loader in src/load_data.py dedupes the raw lice CSV (1,067 byte-identical
# duplicate site-weeks were an extraction artefact). The supervised frame and
# every baseline rely on this contract. Treatment data is deliberately NOT
# deduped — compound treatments (e.g. Cypermethrin + Deltamethrin in one bath)
# legitimately produce multiple rows per site-week.

def test_lice_frame_is_unique_per_site_week(lice):
    """Raw lice frame must be one row per (SITENUMBER, WEEK_START)."""
    dups = lice.duplicated(["SITENUMBER", "WEEK_START"]).sum()
    assert dups == 0, (
        f"lice frame has {dups} duplicate (SITENUMBER, WEEK_START) rows — "
        "loader dedup appears broken"
    )


def test_supervised_frame_is_unique_per_site_week(df_h1, df_h12):
    """Supervised frames inherit one-row-per-site-week from the lice frame.
    If a duplicate slipped through, the negative shift in make_supervised_frame
    could pull a label from a sibling row instead of the next calendar week."""
    for df, h in ((df_h1, 1), (df_h12, 12)):
        dups = df.duplicated(["SITENUMBER", "WEEK_START"]).sum()
        assert dups == 0, (
            f"h={h} supervised frame has {dups} duplicate site-weeks"
        )
