"""Chart 5 in the treatment-free regime: per-PO Pearson correlation between
mobile and adult-female lice, computed only on (site, week_t) pairs where NO
treatment landed at t, t+1, or t+2.

Same shape as the original chart 5 (`05_mobile_adultfemale_pearson_correlation.png`):
two stolpes per PO — same-week and 2-week lagged — but with the management
noise filtered out. This is the "pure biology" view.

Why both correlations on the same subset:
- Picking site-weeks where no treatment lands in [t, t+2] gives us a clean
  forward-2-week window for the lagged Pearson.
- That same site-week (with no t-week treatment) is also a clean point for
  the same-week Pearson. Using the same subset makes the two stolpes
  apples-to-apples.

Run:
    python -m scripts.make_pearson_corr_treatment_free
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
    """Return (site, week_t) rows with mobile_t, FA_t, FA_t+2 plus PO labels.

    Drops fallow / gap rows (the t+2 row must be exactly 14 days later in
    the same site). Marks each pair with `treated_window` = True iff any
    treatment landed at week t, t+1, or t+2 for the same site.
    """
    df = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()
    grouped = df.groupby("SITENUMBER", sort=False)
    df["FA_t_plus_2"] = grouped["FEMALEADULT"].shift(-2)
    df["t_plus_2_date"] = grouped["WEEK_START"].shift(-2)
    gap_ok = (df["t_plus_2_date"] - df["WEEK_START"]).dt.days == 14
    df = df[gap_ok].copy()
    df = df.dropna(subset=["MOBILELICE", "FEMALEADULT", "FA_t_plus_2",
                            "PRODUCTIONAREAID"])

    treat_keys = set(zip(treat["SITENUMBER"], treat["WEEK_START"]))
    treated = np.zeros(len(df), dtype=bool)
    for offset_days in (0, 7, 14):
        offset_dates = df["WEEK_START"] + pd.to_timedelta(offset_days, unit="D")
        keys = list(zip(df["SITENUMBER"].to_numpy(), offset_dates.to_numpy()))
        treated |= np.fromiter((k in treat_keys for k in keys),
                                dtype=bool, count=len(keys))
    df["treated_window"] = treated
    return df


def per_po_correlation(pairs: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """Per-site Pearson(x_col, y_col), averaged within PO. Indexed by PO."""
    def _site_corr(g: pd.DataFrame) -> float:
        if len(g) < MIN_PAIRS_PER_SITE:
            return np.nan
        return float(np.corrcoef(g[x_col], g[y_col])[0, 1])

    per_site = (pairs.groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA", "SITENUMBER"],
                              group_keys=False)
                     .apply(_site_corr, include_groups=False))
    return per_site.groupby(level=[0, 1]).mean()


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

    print("Building (mobile_t, FA_t, FA_t+2) triples...")
    pairs = build_pairs(lice, treat)
    n_total = len(pairs)
    clean = pairs[~pairs["treated_window"]]
    print(f"  Total pairs: {n_total:,}")
    print(f"  Treatment-free pairs: {len(clean):,} "
          f"({100 * len(clean) / n_total:.1f} %)")

    print("Computing per-PO correlations on treatment-free subset...")
    contemp = per_po_correlation(clean, "MOBILELICE", "FEMALEADULT")
    lagged = per_po_correlation(clean, "MOBILELICE", "FA_t_plus_2")

    corr_df = pd.DataFrame({
        "contemporaneous_t": contemp,
        "lagged_t_to_t_plus_2": lagged,
    }).dropna().reset_index()
    corr_df["label"] = corr_df.apply(
        lambda r: po_label(r["PRODUCTIONAREAID"], r["PRODUCTIONAREA"]), axis=1)
    # Sort by PO id so the chart reads south to north top-down (PO1 on top)
    corr_df = corr_df.sort_values("PRODUCTIONAREAID", ascending=False)

    print("\nPer-PO Pearson correlation (treatment-free pairs only):")
    print(corr_df[["label", "contemporaneous_t", "lagged_t_to_t_plus_2"]]
          .round(3).to_string(index=False))

    # --- Plot: two stolpes per PO (matches chart 5's layout) ---
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(corr_df))
    h = 0.38
    bars_c = ax.barh(y - h / 2, corr_df["contemporaneous_t"], height=h,
                     color="#1f77b4",
                     label="Samtidig (mobile_t  vs  adult-female_t)")
    bars_l = ax.barh(y + h / 2, corr_df["lagged_t_to_t_plus_2"], height=h,
                     color="#ff7f0e",
                     label="2-ukers lagget (mobile_t  vs  adult-female_t+2)")
    ax.bar_label(bars_c, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars_l, fmt="%.2f", padding=3, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(corr_df["label"], fontsize=9)
    ax.set_xlabel("Snitt per-site Pearson korrelasjon")
    ax.set_xlim(0, max(0.9, max(corr_df["contemporaneous_t"].max(),
                                 corr_df["lagged_t_to_t_plus_2"].max()) * 1.15))
    ax.set_title("Pearson korrelasjon per PO — kun behandlingsfrie 2-ukers vinduer\n"
                 "(samme format som chart 5, men ren biologi)",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "05f_mobile_adultfemale_pearson_correlation_treatment_free.png"
    fig.savefig(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
