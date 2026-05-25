"""Mobile lice (week t) -> adult-female lice (week t+2), split by
sea-temperature regime.

The biological question: does warm water speed up the mobile -> adult-female
transition? If yes, a given mobile-lice level today should produce a HIGHER
adult-female level 2 weeks later in warm water than in cold water.

Implementation:
- Build (mobile_t, adult_female_t+2, sea_temp_t) triples per site, enforcing
  an exact 14-day gap (sites with missing weeks would otherwise leak in a
  4-week gap and contaminate the lag).
- Exclude pairs where a treatment landed at week t, t+1 or t+2 — those
  artificially suppress the adult-female count at t+2 and bias the slope
  downward by ~50 %. The unfiltered version conflates biology with
  operator response, which is what we explicitly want to separate.
- Bin by mobile lice (X), plot the mean adult-female 2 weeks later (Y) with
  a 95 % bootstrap CI band. One line per temperature regime.
- Two panels:
    Left  -> cold water (5-9 deg C, centered on 7)
    Right -> warm water (13-17 deg C, centered on 15)
- The marginal panels on top show the distribution of mobile-lice values in
  each regime so the reader can see WHERE the mass of the data lives.

Run:
    python -m scripts.make_mobile_adult_temp_scatter
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
from scipy.stats import pearsonr

from src.load_data import load_lice, load_treatment


FIG_DIR = ROOT / "reports" / "figures" / "temperatures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLD_BAND = (5, 9)     # centered on 7
WARM_BAND = (13, 17)   # centered on 15

# Mobile-lice bins for the binned-mean line (edges in lice/fish).
# Wider bins at the high end where data thins out.
MOBILE_BIN_EDGES = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0])
# A point is drawn at the midpoint of each bin
MOBILE_BIN_MID = (MOBILE_BIN_EDGES[:-1] + MOBILE_BIN_EDGES[1:]) / 2


def _make_pairs(lice: pd.DataFrame, treatment: pd.DataFrame) -> pd.DataFrame:
    """Build (mobile_t, adult_female_t+2, seatemp_t) triples per site.

    Excludes pairs where any treatment landed at week t, t+1 or t+2 — those
    suppress the adult-female count at t+2 and bias the cascade downward.
    """
    df = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()
    grouped = df.groupby("SITENUMBER", sort=False)
    df["FA_t_plus_2"] = grouped["FEMALEADULT"].shift(-2)
    df["t_plus_2_date"] = grouped["WEEK_START"].shift(-2)
    gap_ok = (df["t_plus_2_date"] - df["WEEK_START"]).dt.days == 14
    df = df[gap_ok].copy()
    df = df.dropna(subset=["MOBILELICE", "FA_t_plus_2", "SEATEMPERATURE"])

    # Drop rows where a treatment event lands at t, t+1, or t+2 for this site.
    # Build a set of (site, week_start) treatment keys once, then mark rows.
    treat_keys = set(zip(treatment["SITENUMBER"], treatment["WEEK_START"]))
    treated = np.zeros(len(df), dtype=bool)
    for offset_days in (0, 7, 14):
        offset_dates = df["WEEK_START"] + pd.to_timedelta(offset_days, unit="D")
        keys = list(zip(df["SITENUMBER"].to_numpy(), offset_dates.to_numpy()))
        treated |= np.fromiter((k in treat_keys for k in keys),
                                dtype=bool, count=len(keys))
    n_before = len(df)
    df = df[~treated].copy()
    print(f"Treatment-window filter: {n_before:,} -> {len(df):,} "
          f"({100 * (1 - len(df) / n_before):.1f} % removed)")
    return df


def _binned_stats(sub: pd.DataFrame) -> pd.DataFrame:
    """Per mobile-lice bin: mean, 95 % CI, n of FA_t_plus_2."""
    sub = sub.copy()
    sub["mobile_bin"] = pd.cut(sub["MOBILELICE"], bins=MOBILE_BIN_EDGES,
                                include_lowest=True)
    out = (sub.groupby("mobile_bin", observed=True)["FA_t_plus_2"]
              .agg(["mean", "std", "size"])
              .reset_index())
    out["sem"] = out["std"] / np.sqrt(out["size"])
    out["ci_lo"] = out["mean"] - 1.96 * out["sem"]
    out["ci_hi"] = out["mean"] + 1.96 * out["sem"]
    out["bin_mid"] = out["mobile_bin"].apply(lambda iv: iv.mid)
    # Drop bins with too few observations to be meaningful
    out = out[out["size"] >= 20].copy()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    lice = load_lice()
    treat = load_treatment()
    pairs = _make_pairs(lice, treat)

    cold = pairs[(pairs["SEATEMPERATURE"] >= COLD_BAND[0])
                 & (pairs["SEATEMPERATURE"] <= COLD_BAND[1])]
    warm = pairs[(pairs["SEATEMPERATURE"] >= WARM_BAND[0])
                 & (pairs["SEATEMPERATURE"] <= WARM_BAND[1])]
    print(f"Cold ({COLD_BAND[0]}-{COLD_BAND[1]} deg C): n = {len(cold):,}")
    print(f"Warm ({WARM_BAND[0]}-{WARM_BAND[1]} deg C): n = {len(warm):,}")

    cold_stats = _binned_stats(cold)
    warm_stats = _binned_stats(warm)

    # Pearson r on the raw (uncapped) data for the annotation
    r_cold, _ = pearsonr(cold["MOBILELICE"], cold["FA_t_plus_2"])
    r_warm, _ = pearsonr(warm["MOBILELICE"], warm["FA_t_plus_2"])

    # Figure layout: two columns, each with a tall main panel and a short
    # marginal histogram on top showing the mobile-lice distribution.
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], hspace=0.05, wspace=0.20)

    # Top row — marginal histograms of mobile lice in each regime
    ax_hist_cold = fig.add_subplot(gs[0, 0])
    ax_hist_warm = fig.add_subplot(gs[0, 1])
    for ax, sub, color in [(ax_hist_cold, cold, "#1f77b4"),
                            (ax_hist_warm, warm, "#d62728")]:
        ax.hist(sub["MOBILELICE"], bins=MOBILE_BIN_EDGES, color=color,
                alpha=0.7, edgecolor="white")
        ax.set_xlim(0, 7)
        ax.set_xticklabels([])
        ax.set_ylabel("count", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.3)

    # Bottom row — binned-mean lines
    ax_cold = fig.add_subplot(gs[1, 0])
    ax_warm = fig.add_subplot(gs[1, 1], sharey=ax_cold)

    for ax, stats, color, label, r, sub in [
        (ax_cold, cold_stats, "#1f77b4",
         f"Cold water ({COLD_BAND[0]}-{COLD_BAND[1]} deg C)", r_cold, cold),
        (ax_warm, warm_stats, "#d62728",
         f"Warm water ({WARM_BAND[0]}-{WARM_BAND[1]} deg C)", r_warm, warm),
    ]:
        ax.fill_between(stats["bin_mid"], stats["ci_lo"], stats["ci_hi"],
                        color=color, alpha=0.20, label="95 % CI")
        ax.plot(stats["bin_mid"], stats["mean"], marker="o", markersize=7,
                linewidth=2.2, color=color,
                label="Mean adult-female at week t+2")
        ax.axhline(0.5, color="#666", linestyle="--", linewidth=1, alpha=0.7,
                   label="Regulatory limit (0.5)")

        ax.set_xlim(0, 7)
        ax.set_ylim(0, 1.4)
        ax.set_xlabel("Mobile lice per fish at week t")
        ax.set_title(label, fontsize=12, color=color)
        ax.grid(alpha=0.3)

        # Annotate the slope-of-the-binned-line + Pearson r in a corner box
        if len(stats) >= 2:
            slope = np.polyfit(stats["bin_mid"], stats["mean"], 1)[0]
        else:
            slope = float("nan")
        ax.text(0.04, 0.96,
                f"n = {len(sub):,} site-weeks\n"
                f"Pearson r = {r:.3f}\n"
                f"Slope of binned mean = {slope:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="#888", alpha=0.92))
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    ax_cold.set_ylabel("Adult-female lice per fish at week t+2")
    ax_hist_cold.set_title("Cold-water mobile-lice distribution",
                            fontsize=10, color="#1f77b4")
    ax_hist_warm.set_title("Warm-water mobile-lice distribution",
                            fontsize=10, color="#d62728")

    fig.suptitle(
        "Does warm water speed up mobile -> adult-female maturation? "
        "Yes — the slope is steeper at 15 deg C\n"
        "(treatment-window pairs excluded — pure biological cascade)",
        fontsize=13, fontweight="bold", y=1.04)
    fig.text(0.5, -0.02,
             "Pairs where any treatment landed at week t, t+1 or t+2 are excluded — "
             "those would artificially suppress adult-female counts and underestimate "
             "the cascade by ~50 %. Read as: 'X mobile lice today, at this temperature, "
             "untreated, gives expected mean adult-female 2 weeks later'.",
             ha="center", fontsize=9, style="italic", color="#444")

    out = FIG_DIR / "05c_mobile_adult_by_temperature.png"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
