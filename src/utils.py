"""Shared utilities reused across notebooks and modeling code."""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# PO labelling (originally in notebooks/01_eda.py)
# ---------------------------------------------------------------------------

def short_po_name(name: str) -> str:
    """Shorten long Norwegian PA names for chart labels."""
    if pd.isna(name):
        return "Unknown"
    return (str(name)
            .replace("Nordhordaland", "Nordhord.")
            .replace("Trøndelag", "Trønd.")
            .replace("Sør-", "S-")
            .replace("Nord-", "N-"))


def po_label(po_id, po_name) -> str:
    """Format as 'PO5: Stadt til Hustadvika'."""
    if pd.isna(po_id) or pd.isna(po_name):
        return "Unknown"
    return f"PO{int(po_id)}: {short_po_name(po_name)}"


# ---------------------------------------------------------------------------
# Supervised-frame construction (target = BREACH at t + horizon)
# ---------------------------------------------------------------------------

def make_supervised_frame(
    lice: pd.DataFrame,
    horizon: int,
    counted_only: bool = True,
) -> pd.DataFrame:
    """Build a (features_at_T, label_at_T+horizon) frame.

    Within each site, shift BREACH backwards by `horizon` weeks so that
    row at WEEK_START = T carries the label observed at T + horizon. The
    shift is leakage-safe: the prediction-time data on row T never sees
    information from T + horizon — only its label does.

    Parameters
    ----------
    lice
        Frame from `src.load_data.load_lice(apply_cutoff=True)`.
    horizon
        Forecast horizon in weeks (1, 2, or 12).
    counted_only
        If True, drop rows where the *target week* did not have a lice
        count. This restricts evaluation to weeks where reality was also
        observed — the operational view.

    Returns
    -------
    DataFrame with the original columns plus:
      - `target`: BREACH at WEEK_START + horizon (boolean)
      - `target_counted`: HAVECOUNTEDLICE at WEEK_START + horizon (boolean)
      - `target_week_start`: the WEEK_START of the target observation
    Rows with NaN target are dropped.
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer (weeks)")

    df = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()

    # negative shift: row at T gets the value from T + horizon (within site)
    target = df.groupby("SITENUMBER", sort=False)["BREACH"].shift(-horizon)
    counted = df.groupby("SITENUMBER", sort=False)["HAVECOUNTEDLICE"].shift(-horizon)
    target_week = df.groupby("SITENUMBER", sort=False)["WEEK_START"].shift(-horizon)

    df["target"] = target
    df["target_counted"] = counted
    df["target_week_start"] = target_week

    # Drop rows with no label
    df = df.dropna(subset=["target"]).copy()

    # Enforce strict calendar-week gap: shift(-h) shifts by ROWS, not weeks.
    # If a site has missing weeks in its history, two rows apart can be
    # 4+ weeks apart in calendar time — that's a different forecast horizon.
    # Keep only rows where target is exactly `horizon` calendar weeks ahead.
    gap_weeks = (df["target_week_start"] - df["WEEK_START"]).dt.days / 7
    df = df[gap_weeks == horizon].copy()

    if counted_only:
        df = df[df["target_counted"] == True].copy()  # noqa: E712

    df["target"] = df["target"].astype(bool)
    return df.reset_index(drop=True)


def train_test_split_by_year(
    df: pd.DataFrame,
    train_max_year: int = 2024,
    test_year: int = 2025,
    date_col: str = "WEEK_START",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a supervised frame by the WEEK_START year of the prediction-time row.

    Default: training uses WEEK_START in [2012, 2024]; test uses 2025.
    The 2025 split is the final holdout per the case spec ("hold out a
    portion of historical data fully unseen for validation").
    """
    train = df[df[date_col].dt.year <= train_max_year].copy()
    test = df[df[date_col].dt.year == test_year].copy()
    return train, test
