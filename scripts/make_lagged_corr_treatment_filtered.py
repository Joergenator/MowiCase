"""Lagged mobile -> adult-female correlation per PO, with and without
treatment-window pairs.

Companion to chart 5 (`05_mobile_adultfemale_corr.png`). Chart 5 shows two
correlations per PO: contemporaneous (same week) and 2-week lagged. This
script breaks the lagged correlation into TWO versions:

  1. ALL pairs    — same as chart 5's orange stolpe.
  2. Treatment-free pairs — only (site, week) pairs where NO treatment events
     landed at week t, week t+1, or week t+2. Isolates the BIOLOGICAL cascade
     from operator interventions.

The gap between the two stolpes per PO = "how much signal does management
eat into the natural mobile -> adult-female progression?"

Run:
    python -m scripts.make_lagged_corr_treatment_filtered
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.load_data import load_lice, load_treatment
from src.utils import po_label


FIG_DIR = ROOT / "reports" / "figures" / "Lice_correlation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_PAIRS_PER_SITE = 10  # match chart 5's per-site Pearson threshold


def build_pairs(lice: pd.DataFrame, treat: pd.DataFrame) -> pd.DataFrame:
    """Build (site, week_t, mobile_t, FA_t+2, PO) triples, with treatment flag.

    `treated_window` is True if any treatment event lands at week t, t+1, or t+2
    for the same site. The caller can filter on it to isolate biology.
    """
    df = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()
    grouped = df.groupby("SITENUMBER", sort=False)
    df["FA_t_plus_2"] = grouped["FEMALEADULT"].shift(-2)
    df["t_plus_2_date"] = grouped["WEEK_START"].shift(-2)
    # Drop rows where the 2-week-later row is missing or there was a fallow gap
    gap_ok = (df["t_plus_2_date"] - df["WEEK_START"]).dt.days == 14
    df = df[gap_ok].copy()
    df = df.dropna(subset=["MOBILELICE", "FA_t_plus_2", "PRODUCTIONAREAID"])

    # Mark pairs with any treatment in the [t, t+2] window.
    # Treatment table is small enough to lookup via a set of (site, week) keys.
    treat_keys = set(zip(treat["SITENUMBER"], treat["WEEK_START"]))
    treated = np.zeros(len(df), dtype=bool)
    for offset_days in (0, 7, 14):
        offset_dates = df["WEEK_START"] + pd.to_timedelta(offset_days, unit="D")
        keys = list(zip(df["SITENUMBER"].to_numpy(), offset_dates.to_numpy()))
        treated |= np.fromiter((k in treat_keys for k in keys),
                                dtype=bool, count=len(keys))
    df["treated_window"] = treated
    return df


def per_po_lagged_corr(pairs: pd.DataFrame) -> pd.Series:
    """Per-site lagged Pearson, averaged within PO. Returns Series indexed by PO."""
    def _site_corr(g: pd.DataFrame) -> float:
        if len(g) < MIN_PAIRS_PER_SITE:
            return np.nan
        return float(np.corrcoef(g["MOBILELICE"], g["FA_t_plus_2"])[0, 1])

    per_site = (pairs.groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA", "SITENUMBER"],
                              group_keys=False)
                     .apply(_site_corr, include_groups=False))
    # Average across sites within each PO
    return (per_site.groupby(level=[0, 1]).mean())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    print("Loading data...")
    lice = load_lice()
    treat = load_treatment()

    print("Building (mobile_t, FA_t+2) pairs...")
    pairs = build_pairs(lice, treat)
    n_total = len(pairs)
    n_clean = (~pairs["treated_window"]).sum()
    print(f"  Total pairs: {n_total:,}")
    print(f"  Treatment-free pairs: {n_clean:,} "
          f"({100 * n_clean / n_total:.1f} %)")

    print("Computing per-PO lagged correlations...")
    corr_all = per_po_lagged_corr(pairs)
    corr_clean = per_po_lagged_corr(pairs[~pairs["treated_window"]])

    corr_df = pd.DataFrame({
        "lagged_all_pairs": corr_all,
        "lagged_treatment_free": corr_clean,
    }).dropna()
    corr_df["gap"] = corr_df["lagged_treatment_free"] - corr_df["lagged_all_pairs"]
    corr_df = corr_df.reset_index()
    corr_df["label"] = corr_df.apply(
        lambda r: po_label(r["PRODUCTIONAREAID"], r["PRODUCTIONAREA"]), axis=1)
    # Sort by PO id so the chart reads south -> north
    corr_df = corr_df.sort_values("PRODUCTIONAREAID", ascending=False)

    print("\nPer-PO lagged correlation (mobile_t -> FA_t+2):")
    print(corr_df[["label", "lagged_all_pairs", "lagged_treatment_free", "gap"]]
          .round(3).to_string(index=False))

    # --- Plot: two stolpes per PO ---
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(corr_df))
    h = 0.38
    bars1 = ax.barh(y - h / 2, corr_df["lagged_all_pairs"], height=h,
                    color="#ff7f0e", label="Alle 2-ukers par (matcher chart 5 oransje)")
    bars2 = ax.barh(y + h / 2, corr_df["lagged_treatment_free"], height=h,
                    color="#2ca02c", label="Kun behandlingsfrie par (ren biologi)")
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(corr_df["label"], fontsize=9)
    ax.set_xlabel("Snitt per-site Pearson korrelasjon  "
                  "(mobile_t  →  adult-female_t+2)")
    ax.set_xlim(0, max(0.8, corr_df["lagged_treatment_free"].max() * 1.15))
    ax.set_title("Lagged korrelasjon per PO — med og uten behandlingsvindu\n"
                 "Gapet = biologisk signal som maskeres av management",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "05e_lagged_pearson_correlation_treatment_filtered.png"
    fig.savefig(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
