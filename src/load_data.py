"""Load and clean BarentsWatch lice + treatment data.

The single source of truth for data loading. All EDA, feature engineering,
and modelling MUST go through this module so the 2026 leakage cutoff
cannot be bypassed by accident.

Public API
----------
- TRAIN_CUTOFF: pd.Timestamp = the first date that may NOT appear in training
- load_lice(...): full lice dataset (cleaned)
- load_treatment(...): full treatment dataset (cleaned)
- load_training_data(): convenience — both datasets filtered to < 2026-01-01
- assert_no_leakage(df): raises if df contains any rows with date >= TRAIN_CUTOFF
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# Hard cutoff: no row with WEEK_START >= this date may be used for training
# or feature engineering. This is enforced by every public loader.
TRAIN_CUTOFF = pd.Timestamp("2026-01-01")

# Plausible bounds for data cleaning (defensive — peek found a 196 °C outlier)
SEA_TEMP_MIN = -2.0       # below seawater freezing → sensor error
SEA_TEMP_MAX = 30.0       # above this in Norwegian waters → sensor error
LICE_MAX = 200.0          # mobile/female adult lice — anything > 200 is implausible
PERSISTENT_LICE_MAX = 50.0  # persistent (sessile) lice — q99 ≈ 2.9, max 99
                             # row found at value 99 was clearly a sentinel


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def iso_week_to_date(year: pd.Series, week: pd.Series) -> pd.Series:
    """Convert ISO year + week (1-53) to the Monday of that ISO week."""
    s = year.astype(int).astype(str) + "-W" + week.astype(int).astype(str).str.zfill(2) + "-1"
    return pd.to_datetime(s, format="%G-W%V-%u", errors="coerce")


def _yes_no_to_bool(s: pd.Series) -> pd.Series:
    """Convert Norwegian 'Ja'/'Nei' string column to nullable boolean."""
    return s.map({"Ja": True, "Nei": False}).astype("boolean")


def _normalize_pa(name: pd.Series) -> pd.Series:
    """Normalize known source-data inconsistencies in production area names."""
    return (name
            .str.replace("Nordhordland til Stadt", "Nordhordaland til Stadt", regex=False)
            # 'Ryfylke' is the older spelling; 'Ryfylket' (definite article) is current.
            .replace({"Ryfylke": "Ryfylket"}))


# ----------------------------------------------------------------------------
# Public loaders
# ----------------------------------------------------------------------------

def load_lice(
    apply_cutoff: bool = True,
    drop_fallow: bool = False,
) -> pd.DataFrame:
    """Load and clean vlice.csv.

    Parameters
    ----------
    apply_cutoff
        If True (default), filter to weeks before TRAIN_CUTOFF. Set to False
        ONLY for downstream scoring / serving — never for training or features.
    drop_fallow
        If True, drop rows where LIKELYNOFISH == True (no fish on site).
    """
    path = RAW / "vlice.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", engine="python", sep=None)

    # Build the week-start date column up-front; everything downstream uses it
    df["WEEK_START"] = iso_week_to_date(df["YEAR"], df["WEEK"])

    # Cast booleans.
    #
    # BREACH is built directly from the source `OVERTHELICELIMITWEEK` column,
    # NOT re-derived from FEMALEADULT vs LICELIMITWEEK. The reason: the source
    # flag uses the regulatory rule which is FEMALEADULT >= LICELIMITWEEK
    # ("reaching the limit counts as a breach"), not strict `>`. The audit
    # confirmed 851 rows where FEMALEADULT == LICELIMITWEEK exactly all have
    # OVERTHELICELIMITWEEK = "Ja". If you ever need to derive BREACH yourself,
    # use `>=`, not `>`.
    #
    # OVERTHELICELIMITWEEK also has a third value "Ukjent" (Unknown), used when
    # no lice count happened that week. `_yes_no_to_bool` maps that to <NA>,
    # which is the correct semantics — we don't know whether the site breached.
    df["LIKELYNOFISH"] = _yes_no_to_bool(df["LIKELYNOFISH"])
    df["HAVECOUNTEDLICE"] = _yes_no_to_bool(df["HAVECOUNTEDLICE"])
    df["BREACH"] = _yes_no_to_bool(df["OVERTHELICELIMITWEEK"])

    # The weekly limit comes as a string — cast to float
    df["LICELIMITWEEK"] = pd.to_numeric(df["LICELIMITWEEK"], errors="coerce")

    # Clean implausible sensor / count values (peek found 196 °C and outliers)
    mask_bad_temp = (df["SEATEMPERATURE"] < SEA_TEMP_MIN) | (df["SEATEMPERATURE"] > SEA_TEMP_MAX)
    df.loc[mask_bad_temp, "SEATEMPERATURE"] = np.nan

    # Per-column upper bounds (PERSISTENTLICE has a tighter cap — see constants)
    col_max = {"FEMALEADULT": LICE_MAX,
               "MOBILELICE": LICE_MAX,
               "PERSISTENTLICE": PERSISTENT_LICE_MAX}
    for col, maximum in col_max.items():
        df.loc[df[col] > maximum, col] = np.nan
        df.loc[df[col] < 0, col] = np.nan

    # Drop exact-duplicate rows on the natural key (SITENUMBER, YEAR, WEEK).
    # The audit found 1,067 site-week pairs that appear twice with byte-identical
    # values — a data-extraction artefact. Treatment data is NOT deduped because
    # its multi-row groups are legitimate compound treatments (e.g. Cypermethrin
    # + Deltamethrin in the same bath).
    df = df.drop_duplicates(subset=["SITENUMBER", "YEAR", "WEEK"], keep="first")

    # Normalize PA name typo
    df["PRODUCTIONAREA"] = _normalize_pa(df["PRODUCTIONAREA"])

    # Drop EDW housekeeping columns — irrelevant to analysis
    df = df.drop(columns=["EDWDATELOAD", "EDWDATECHANGE", "EDWPROCESSID",
                          "LICE_SK", "LICE_HK"])

    if drop_fallow:
        df = df[df["LIKELYNOFISH"] != True].copy()  # noqa: E712

    if apply_cutoff:
        # Two-pronged filter: WEEK_START strictly before 2026-01-01 AND source
        # YEAR column strictly < 2026. The second clause guards against the ISO
        # calendar edge case where week 1 of 2026 starts on Mon 2025-12-29 —
        # that row's WEEK_START is in December but its YEAR is 2026, and we
        # treat anything labeled 2026 in the source as 2026 data.
        df = df[(df["WEEK_START"] < TRAIN_CUTOFF) & (df["YEAR"] < 2026)].copy()
        assert_no_leakage(df)

    return df.sort_values(["SITENUMBER", "WEEK_START"]).reset_index(drop=True)


def load_treatment(apply_cutoff: bool = True) -> pd.DataFrame:
    """Load and clean vtreatment.csv."""
    path = RAW / "vtreatment.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", engine="python", sep=None)

    df["WEEK_START"] = iso_week_to_date(df["YEAR"], df["WEEK"])
    df["PRODUCTIONAREA"] = _normalize_pa(df["PRODUCTIONAREA"])

    df = df.drop(columns=["EDWDATELOAD", "EDWDATECHANGE", "EDWPROCESSID",
                          "TREATMENT_SK", "TREATMENT_HK"])

    if apply_cutoff:
        df = df[(df["WEEK_START"] < TRAIN_CUTOFF) & (df["YEAR"] < 2026)].copy()
        assert_no_leakage(df)

    return df.sort_values(["SITENUMBER", "WEEK_START"]).reset_index(drop=True)


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: return (lice, treatment) both filtered to < 2026."""
    return load_lice(apply_cutoff=True), load_treatment(apply_cutoff=True)


# ----------------------------------------------------------------------------
# Leakage guard
# ----------------------------------------------------------------------------

class LeakageError(AssertionError):
    """Raised when a DataFrame contains data on or after TRAIN_CUTOFF."""


def assert_no_leakage(df: pd.DataFrame, date_col: str = "WEEK_START") -> None:
    """Raise LeakageError if any row contains 2026 data.

    Two independent checks:
      1. WEEK_START < TRAIN_CUTOFF (2026-01-01)
      2. YEAR < 2026 (catches the ISO-week-1-of-2026 edge case where a row's
         WEEK_START Monday falls in late December but its source YEAR is 2026)

    Call this on every training-time DataFrame. It's cheap and acts as a
    runtime contract that the cutoff has not been violated by joins, feature
    engineering, or rolling windows.
    """
    if date_col not in df.columns:
        raise LeakageError(f"Cannot verify leakage: column {date_col!r} missing")
    if len(df) == 0:
        return

    max_date = df[date_col].max()
    if pd.notna(max_date) and max_date >= TRAIN_CUTOFF:
        n_bad = (df[date_col] >= TRAIN_CUTOFF).sum()
        raise LeakageError(
            f"Found {n_bad} rows with {date_col} >= {TRAIN_CUTOFF.date()} "
            f"(max={max_date.date()}). This data must not be used for training."
        )

    if "YEAR" in df.columns:
        max_year = df["YEAR"].max()
        if pd.notna(max_year) and max_year >= 2026:
            n_bad = (df["YEAR"] >= 2026).sum()
            raise LeakageError(
                f"Found {n_bad} rows with YEAR >= 2026 (max={int(max_year)}). "
                "Source labels these as 2026 data — must not be used for training."
            )


# ----------------------------------------------------------------------------
# CLI: run as `python -m src.load_data` for a sanity check
# ----------------------------------------------------------------------------

def _summary(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name}: shape={df.shape}, "
          f"date range {df['WEEK_START'].min().date()} → {df['WEEK_START'].max().date()}, "
          f"sites={df['SITENUMBER'].nunique()}, POs={df['PRODUCTIONAREA'].nunique()}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"TRAIN_CUTOFF = {TRAIN_CUTOFF.date()}")
    lice, treat = load_training_data()
    _summary(lice, "lice (training)")
    _summary(treat, "treatment (training)")

    if "BREACH" in lice.columns:
        n_obs = lice["BREACH"].notna().sum()
        n_breach = (lice["BREACH"] == True).sum()  # noqa: E712
        print(f"breach rate in training data: {n_breach}/{n_obs} = "
              f"{100 * n_breach / n_obs:.2f}%")

    # Show what 2026 data was excluded
    full = load_lice(apply_cutoff=False)
    excluded = full[full["WEEK_START"] >= TRAIN_CUTOFF]
    print(f"\n2026 rows EXCLUDED from training: {len(excluded)} "
          f"({excluded['WEEK_START'].min().date()} → {excluded['WEEK_START'].max().date()})"
          if len(excluded) else "\nNo 2026 data present in source.")
