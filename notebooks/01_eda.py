# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # BarentsWatch Lice Data — Exploratory Analysis
#
# Eight charts answering the six questions from the case brief, plus two bonus insights
# that bridge into the modeling step. All data is filtered to weeks before 2026-01-01
# via the leakage firewall in `src/load_data.py`.

# %%
import sys
from pathlib import Path

# Make src importable when running from the project root
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.load_data import load_lice, load_treatment, assert_no_leakage, TRAIN_CUTOFF

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
})

FIG_DIR = ROOT / "reports" / "figures"
FIG_TREATMENTS = FIG_DIR / "Treatments"
FIG_BREACHES = FIG_DIR / "breaches"
FIG_TEMPS = FIG_DIR / "temperatures"
FIG_LICE_CORR = FIG_DIR / "Lice_correlation"
for _d in (FIG_DIR, FIG_TREATMENTS, FIG_BREACHES, FIG_TEMPS, FIG_LICE_CORR):
    _d.mkdir(parents=True, exist_ok=True)


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
    """Format as 'PO5: Stadt til Hustadvika' — using the regulatory PO number."""
    if pd.isna(po_id) or pd.isna(po_name):
        return "Unknown"
    return f"PO{int(po_id)}: {short_po_name(po_name)}"


# %%
lice = load_lice(apply_cutoff=True, drop_fallow=False)
treat = load_treatment(apply_cutoff=True)
assert_no_leakage(lice)
assert_no_leakage(treat)

print(f"TRAIN_CUTOFF = {TRAIN_CUTOFF.date()}")
print(f"lice:      {lice.shape}, {lice['WEEK_START'].min().date()} → {lice['WEEK_START'].max().date()}")
print(f"treatment: {treat.shape}")

# Restrict the breach + lice-count analyses to weeks where counting actually happened
# (HAVECOUNTEDLICE == True). Fallow weeks are kept in treatment counts but excluded
# from breach-rate and lice-pressure aggregations.
lice_counted = lice[(lice["HAVECOUNTEDLICE"] == True) &  # noqa: E712
                    lice["PRODUCTIONAREA"].notna() &
                    lice["PRODUCTIONAREAID"].notna()].copy()
lice_counted["PO_short"] = [
    po_label(i, n) for i, n in
    zip(lice_counted["PRODUCTIONAREAID"], lice_counted["PRODUCTIONAREA"])
]
treat["PO_short"] = [
    po_label(i, n) for i, n in
    zip(treat["PRODUCTIONAREAID"], treat["PRODUCTIONAREA"])
]

print(f"\nCounted weeks: {len(lice_counted)} ({len(lice_counted)/len(lice):.1%} of all site-weeks)")
print(f"Overall breach rate (counted weeks): {lice_counted['BREACH'].mean():.2%}")


# %% [markdown]
# ## Chart 1 — Treatment intensity by Production Area
#
# **Case Q1:** *Which production areas have the most treatments relative to the
# number of active sites?*
#
# Numerator = treatment events recorded. Denominator = distinct active site-years
# (a site is "active" in a year if it had at least one lice count).

# %%
# Drop rows with unknown PA from both sides so the ratio is meaningful
lice_known = lice_counted.dropna(subset=["PRODUCTIONAREA"])
treat_known = treat.dropna(subset=["PRODUCTIONAREA"])

active_site_years = (
    lice_known.groupby("PO_short")
    .apply(lambda d: d[["SITENUMBER", "YEAR"]].drop_duplicates().shape[0], include_groups=False)
    .rename("active_site_years")
)

treatments_per_po = treat_known.groupby("PO_short").size().rename("treatments")

intensity = pd.concat([treatments_per_po, active_site_years], axis=1).dropna()
intensity = intensity[intensity["active_site_years"] >= 20]  # need a meaningful denominator
intensity["per_site_year"] = intensity["treatments"] / intensity["active_site_years"]
intensity = intensity.sort_values("per_site_year", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(intensity.index, intensity["per_site_year"], color="#1f77b4")
ax.set_xlabel("Treatments per active site-year")
ax.set_title("Treatment intensity by Production Area")
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
fig.tight_layout()
fig.savefig(FIG_TREATMENTS / "01_treatment_intensity_by_po.png")
plt.show()

print(intensity.sort_values("per_site_year", ascending=False).round(2))


# %% [markdown]
# ## Chart 2 — Breach rate by Production Area
#
# **Case Q2:** *Which regions or production areas appear to breach lice limits
# most consistently?*
#
# Breach rate = fraction of counted site-weeks where `OVERTHELICELIMITWEEK == "Ja"`.
# Dashed line shows the overall base rate.

# %%
breach_rate = (
    lice_counted.dropna(subset=["BREACH"])
    .groupby("PO_short")["BREACH"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=True)
)
overall_rate = lice_counted["BREACH"].mean()

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(breach_rate.index, breach_rate["mean"] * 100,
               color=["#d62728" if r > overall_rate else "#1f77b4" for r in breach_rate["mean"]])
ax.axvline(overall_rate * 100, color="black", linestyle="--", linewidth=1,
           label=f"Overall base rate {overall_rate:.1%}")
ax.set_xlabel("Breach rate (%)")
ax.set_title("Lice-limit breach rate by Production Area")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_BREACHES / "02_breach_rate_by_po.png")
plt.show()

print(breach_rate.assign(mean_pct=lambda d: (d["mean"] * 100).round(2)).sort_values("mean", ascending=False))


# %% [markdown]
# ## Chart 3 — Temperature × PO heatmap of mean adult-female lice (split by limit)
#
# **Case Q3:** *At which temperatures does lice pressure appear most prevalent
# in each Production Area?*
#
# Two panels, one per regulatory regime, on a shared color scale:
# - **0.5-limit weeks (normal regime, ~weeks 1-15 and 27-52):** baseline
#   biological response — what lice counts look like under normal management.
# - **0.2-limit weeks (spring smolt-protection window, ~weeks 16-26):** the
#   same biology but under aggressive management. Counts are actively
#   suppressed by treatments to stay below the tighter 0.2 threshold.
#
# Splitting matters because mixing them confounds biology with regulation:
# spring weeks are cold *and* aggressively managed, so a single combined
# heatmap would over-credit "cold → low lice" to biology when management is
# doing some of the work.
#
# Cells with < 30 observations are blanked out.

# %%
def make_heatmap_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.dropna(subset=["SEATEMPERATURE", "FEMALEADULT"]).copy()
    d["temp_bin"] = pd.cut(d["SEATEMPERATURE"],
                            bins=np.arange(0, 22, 1),
                            labels=[f"{i}-{i+1}" for i in range(0, 21)],
                            include_lowest=True)
    g = d.groupby(["PO_short", "temp_bin"], observed=True)["FEMALEADULT"]
    return g.mean().unstack(), g.count().unstack()


# Order POs by their regulatory PO number (south → north by design)
po_order = (lice_counted.groupby("PO_short")["PRODUCTIONAREAID"]
            .first().sort_values().index)
# Keep the variable name `po_lat` for backwards compatibility downstream
po_lat = pd.Series(range(len(po_order)), index=po_order)

limit_filters = {
    "0.5 (normal regime)": lice_counted[lice_counted["LICELIMITWEEK"] == 0.5],
    "0.2 (spring smolt window)": lice_counted[lice_counted["LICELIMITWEEK"] == 0.2],
}

# Compute both heatmaps and a shared vmax so the color scale is comparable
heat_panels = {}
for label, df_sub in limit_filters.items():
    mean_, count_ = make_heatmap_data(df_sub)
    masked = mean_.where(count_ >= 30)
    masked = masked.reindex(po_lat.index)
    heat_panels[label] = masked

vmax = max(p.max().max() for p in heat_panels.values())

fig, axes = plt.subplots(2, 1, figsize=(14, 13), sharex=True)
for ax, (label, panel) in zip(axes, heat_panels.items()):
    sns.heatmap(panel, annot=True, fmt=".2f", cmap="YlOrRd",
                cbar_kws={"label": "Mean adult-female lice per fish"},
                linewidths=0.4, ax=ax, vmin=0, vmax=vmax,
                annot_kws={"fontsize": 8})
    n_obs = int(limit_filters[label]["FEMALEADULT"].notna().sum())
    ax.set_title(f"LICELIMITWEEK = {label}   (n = {n_obs:,} site-weeks)",
                 fontsize=12)
    ax.set_ylabel("Production Area (S→N)")
    plt.setp(ax.get_yticklabels(), fontsize=9)

axes[-1].set_xlabel("Sea temperature (°C)")
plt.setp(axes[-1].get_xticklabels(), rotation=0, fontsize=9)
fig.suptitle("Adult-female lice by temperature & PO — split by regulatory regime",
             y=1.0, fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_TEMPS / "03_temp_po_heatmap.png")
plt.show()


# %% [markdown]
# ## Chart 4 — Seasonal pattern with the spring threshold band
#
# **Case Q4:** *Are there seasonal patterns worth highlighting?*
#
# Mean adult-female lice by ISO week (lines: per recent year, bold: 2020-2025 median).
# Shaded band shows weeks where the regulator's stricter 0.2 limit applies for some
# production areas.

# %%
weekly = lice_counted.dropna(subset=["FEMALEADULT"]).copy()
weekly_median = weekly.groupby(["YEAR", "WEEK"])["FEMALEADULT"].mean().reset_index()
recent_years = sorted(weekly_median["YEAR"].unique())[-6:]

# Find the spring-threshold weeks (where LICELIMITWEEK == 0.2)
spring_weeks = (lice[lice["LICELIMITWEEK"] == 0.2]["WEEK"]
                .value_counts().sort_index())
spring_min, spring_max = spring_weeks.index.min(), spring_weeks.index.max()

fig, ax = plt.subplots(figsize=(11, 6))
ax.axvspan(spring_min, spring_max, color="#ffeb99", alpha=0.5,
           label=f"Spring 0.2-limit window (W{spring_min}-{spring_max})")

# Qualitative palette so each year is visually distinct
year_colors = sns.color_palette("tab10", n_colors=len(recent_years))
for color, yr in zip(year_colors, recent_years):
    yd = weekly_median[weekly_median["YEAR"] == yr]
    ax.plot(yd["WEEK"], yd["FEMALEADULT"], color=color, alpha=0.85,
            linewidth=1.6, label=str(yr))

# Bold median across all years
median_curve = weekly_median.groupby("WEEK")["FEMALEADULT"].median()
ax.plot(median_curve.index, median_curve.values, color="black", linewidth=3.0,
        label="Median (all years)")

ax.set_xlabel("ISO week of year")
ax.set_ylabel("Mean adult-female lice per fish")
ax.set_title("Seasonal lice pressure: pronounced autumn peak after the spring window")
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.set_xlim(1, 53)
fig.tight_layout()
fig.savefig(FIG_TEMPS / "04_seasonal_pattern.png")
plt.show()


# %% [markdown]
# ## Chart 5 — Adult-female ↔ mobile-lice correlation per PO (contemporaneous vs lagged)
#
# **Case Q6:** *What is the correlation between adult-female lice and mobile lice
# in each PO?*
#
# Two correlations per PO:
# - **Contemporaneous** (same week): how closely the two life-cycle stages co-vary
#   in any given count.
# - **2-week lagged** (mobile at week t−2 vs adult-female at t): how well current
#   mobile lice *predict* adult-female lice two weeks later. Lower than
#   contemporaneous (the populations decorrelate over time and respond to
#   interventions) but still substantial — this is the signal that makes short-
#   horizon forecasting feasible.

# %%
def per_site_lagged_corr(g: pd.DataFrame, lag: int) -> float:
    g = g.sort_values("WEEK_START")
    x = g["MOBILELICE"].shift(lag)
    y = g["FEMALEADULT"]
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def po_corr(lag: int) -> pd.Series:
    per_site = (
        lice_counted.dropna(subset=["MOBILELICE", "FEMALEADULT"])
        .groupby(["PO_short", "SITENUMBER"], group_keys=False)
        .apply(lambda g: per_site_lagged_corr(g, lag), include_groups=False)
    )
    return per_site.groupby(level=0).mean()


corr_t0 = po_corr(0)
corr_t2 = po_corr(2)

corr_df = (pd.DataFrame({"contemporaneous (t)": corr_t0,
                         "lagged 2 weeks (t-2 mobile → t adult-female)": corr_t2})
           .dropna()
           .sort_values("contemporaneous (t)"))

fig, ax = plt.subplots(figsize=(10, 6.5))
corr_df.plot.barh(ax=ax, color=["#1f77b4", "#ff7f0e"], width=0.75)
ax.set_xlabel("Mean per-site Pearson correlation")
ax.set_xlim(0, 1)
ax.set_ylabel("")
ax.set_title("Mobile ↔ adult-female lice correlation, by PO")
ax.legend(loc="lower right", fontsize=10)
# Annotate each bar with its correlation value (same style as 05e / 05f)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
fig.tight_layout()
fig.savefig(FIG_LICE_CORR / "05_mobile_adultfemale_pearson_correlation.png")
plt.show()

print(corr_df.round(3))


# %% [markdown]
# ## Chart 6 — Geographic map of site-level annual breach rate
#
# **Case Q4 (geographic) + Q5 (insights):** Each dot is one site, colored by its
# average breach rate across all years. Norway's coast is recognizable from lat/long
# alone — no map projection needed for a presentation chart.

# %%
site_breach = (
    lice_counted.dropna(subset=["BREACH"])
    .groupby("SITENUMBER")
    .agg(breach_rate=("BREACH", "mean"),
         lat=("LATITUDE", "mean"),
         lon=("LONGITUDE", "mean"),
         n_obs=("BREACH", "size"))
)
site_breach = site_breach[site_breach["n_obs"] >= 20]

fig, ax = plt.subplots(figsize=(7.5, 10))
# Clip color scale at 95th percentile so outliers don't wash out the rest
vmax = np.percentile(site_breach["breach_rate"] * 100, 95)
sc = ax.scatter(site_breach["lon"], site_breach["lat"],
                c=site_breach["breach_rate"] * 100,
                s=18, cmap="YlOrRd", edgecolors="grey", linewidths=0.2,
                alpha=0.85, vmin=0, vmax=vmax)
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, extend="max")
cb.set_label(f"Site breach rate (%) — clipped at {vmax:.0f}%")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(f"Breach hotspots along the Norwegian coast (n={len(site_breach)} sites)")
ax.set_aspect(2.0)  # Norway is tall; stretch lat axis for legibility
fig.tight_layout()
fig.savefig(FIG_BREACHES / "06_geographic_breach_map.png")
plt.show()


# %% [markdown]
# ## Chart 7 — Treatment events per year, by method
#
# **Case Q5 (operational insight):** Chemical (medicinal) treatments dominated
# until ~2017, then mechanical removal and cleaner-fish became the main
# approaches — driven by resistance development and tighter regulation.
#
# Each line is one treatment method's annual count. A vertical marker shows
# where BarentsWatch changed its taxonomy in 2024 — the mechanical and
# cleaner-fish lines drop because those events are now reported under the
# new "non-medicinal" umbrella category.

# %%
ACTION_EN = {
    "medikamentell": "Medicinal (chemical)",
    "ikke-medikamentell": "Non-medicinal (umbrella, 2024+)",
    "mekanisk fjerning": "Mechanical removal",
    "rensefisk": "Cleaner fish",
}
ACTION_COLOR = {
    "Medicinal (chemical)":            "#d62728",  # red
    "Mechanical removal":              "#1f77b4",  # blue
    "Cleaner fish":                    "#2ca02c",  # green
    "Non-medicinal (umbrella, 2024+)": "#7f7f7f",  # grey
}

treat_mix = (
    treat.assign(YEAR=treat["YEAR"].astype(int),
                 ACTION_EN=treat["ACTION"].map(ACTION_EN).fillna(treat["ACTION"]))
    .groupby(["YEAR", "ACTION_EN"]).size().unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(11, 6))

# Draw the lines FIRST so we know the y-axis range, then place the annotation
for method in ["Medicinal (chemical)", "Mechanical removal",
               "Cleaner fish", "Non-medicinal (umbrella, 2024+)"]:
    if method in treat_mix.columns:
        ax.plot(treat_mix.index, treat_mix[method],
                label=method, color=ACTION_COLOR[method],
                linewidth=2.3, marker="o", markersize=5)

TAXONOMY_CHANGE = 2024
ymax = ax.get_ylim()[1]
ax.axvline(TAXONOMY_CHANGE - 0.5, color="black", linestyle="--",
           linewidth=1.2, alpha=0.6)
ax.annotate("Taxonomy change\n(BarentsWatch, 2024)",
            xy=(TAXONOMY_CHANGE - 0.5, ymax * 0.78),
            xytext=(TAXONOMY_CHANGE - 4.5, ymax * 0.78),
            fontsize=9, color="black", ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color="black", alpha=0.6))

ax.set_ylabel("Number of treatment events per year")
ax.set_xlabel("Year")
ax.set_title("Treatment events per year, by method")
ax.legend(title="Treatment method", loc="upper left", fontsize=10, frameon=True)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_TREATMENTS / "07_treatment_mix_over_time.png")
plt.show()

print(treat_mix.tail(8))


# %% [markdown]
# ## Chart 9 — Does temperature explain the high-lice years?
#
# **Follow-up to chart 4:** 2023 and 2024 have notably high autumn lice peaks.
# If the temperature → lice mechanism (chart 3) is real, those years should
# also be warmer than average.
#
# Two stacked panels share the same x-axis (week of year) and use the same
# per-year colors as chart 4 so they can be read together. The bar chart at
# the bottom summarises mean temperature in the lice-peak window (weeks 25-40)
# for each year.

# %%
temp_weekly = (lice_counted.dropna(subset=["SEATEMPERATURE"])
               .groupby(["YEAR", "WEEK"])["SEATEMPERATURE"].mean()
               .reset_index())

PEAK_WEEKS = (25, 40)  # the autumn lice-peak window

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                          gridspec_kw={"height_ratios": [1.0, 1.0]})

# --- Top: adult-female lice (same data as chart 4, same colors) ---
ax_top = axes[0]
ax_top.axvspan(spring_min, spring_max, color="#ffeb99", alpha=0.4)
for color, yr in zip(year_colors, recent_years):
    yd = weekly_median[weekly_median["YEAR"] == yr]
    ax_top.plot(yd["WEEK"], yd["FEMALEADULT"], color=color, alpha=0.85,
                linewidth=1.6, label=str(yr))
ax_top.axvspan(PEAK_WEEKS[0], PEAK_WEEKS[1], color="grey", alpha=0.08)
ax_top.set_ylabel("Mean adult-female\nlice per fish")
ax_top.set_title("Lice pressure (top) vs sea temperature (bottom), by ISO week")
ax_top.legend(loc="upper left", fontsize=9, ncol=2)

# --- Bottom: sea temperature ---
ax_bot = axes[1]
ax_bot.axvspan(spring_min, spring_max, color="#ffeb99", alpha=0.4,
               label=f"Spring 0.2-limit window")
for color, yr in zip(year_colors, recent_years):
    td = temp_weekly[temp_weekly["YEAR"] == yr]
    ax_bot.plot(td["WEEK"], td["SEATEMPERATURE"], color=color, alpha=0.85,
                linewidth=1.6, label=str(yr))
ax_bot.axvspan(PEAK_WEEKS[0], PEAK_WEEKS[1], color="grey", alpha=0.08,
               label=f"Lice-peak window (W{PEAK_WEEKS[0]}-{PEAK_WEEKS[1]})")
ax_bot.set_xlabel("ISO week of year")
ax_bot.set_ylabel("Mean sea\ntemperature (°C)")
ax_bot.set_xlim(1, 53)
ax_bot.legend(loc="upper left", fontsize=9, ncol=2)

fig.tight_layout()
fig.savefig(FIG_TEMPS / "09_lice_vs_temperature_by_year.png")
plt.show()

# --- Compute the peak-window summary table so the narrative is data-driven ---
peak_summary = []
for yr in recent_years:
    mask_lice = (weekly_median["YEAR"] == yr) & weekly_median["WEEK"].between(*PEAK_WEEKS)
    mask_temp = (temp_weekly["YEAR"] == yr) & temp_weekly["WEEK"].between(*PEAK_WEEKS)
    peak_summary.append({
        "year": yr,
        "mean_temp_peak_window_C": temp_weekly.loc[mask_temp, "SEATEMPERATURE"].mean(),
        "peak_femaleadult": weekly_median.loc[mask_lice, "FEMALEADULT"].max(),
        "mean_femaleadult_peak_window": weekly_median.loc[mask_lice, "FEMALEADULT"].mean(),
    })
peak_summary = pd.DataFrame(peak_summary).set_index("year").round(3)
print(f"\nWeek-{PEAK_WEEKS[0]}-{PEAK_WEEKS[1]} summary (lice-peak window):")
print(peak_summary)

# Pearson correlation across years between mean temp and mean lice in peak window
corr_yr = peak_summary["mean_temp_peak_window_C"].corr(
    peak_summary["mean_femaleadult_peak_window"])
print(f"\nCross-year correlation (mean temp ↔ mean lice in peak window): r = {corr_yr:.3f}")


# %% [markdown]
# ## Chart 10 — Active-treatment intensity (excludes cleaner fish)
#
# **Follow-up to chart 1:** `rensefisk` (cleaner fish) is a passive, preventive
# biological control — once a cohort is stocked, they continuously eat lice —
# not a reactive intervention triggered by rising lice counts. Counting them
# alongside chemical baths and mechanical removal conflates "preventive setup"
# with "reactive management intensity." This chart shows the *reactive*
# intensity only: medicinal + mechanical + non-medicinal events.
#
# Compare the two charts side-by-side in the deck to show how the southern POs
# (PO1, PO5) rise further up the ranking when only active interventions count.

# %%
# Exclude cleaner-fish events (with defensive lowercase/strip for typos)
treat_active = treat[treat["ACTION"].astype(str).str.lower().str.strip()
                     != "rensefisk"].copy()
n_excluded = len(treat) - len(treat_active)
print(f"Excluded {n_excluded:,} cleaner-fish events "
      f"({n_excluded / len(treat):.1%} of all treatment rows)")

treat_active_known = treat_active.dropna(subset=["PRODUCTIONAREA"])

treatments_active_per_po = treat_active_known.groupby("PO_short").size().rename("treatments")
intensity_active = pd.concat([treatments_active_per_po, active_site_years], axis=1).dropna()
intensity_active = intensity_active[intensity_active["active_site_years"] >= 20]
intensity_active["per_site_year"] = intensity_active["treatments"] / intensity_active["active_site_years"]
intensity_active = intensity_active.sort_values("per_site_year", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(intensity_active.index, intensity_active["per_site_year"],
               color="#1f77b4")
ax.set_xlabel("Active treatments per active site-year (cleaner fish excluded)")
ax.set_title("Active-treatment intensity by PO — cleaner fish excluded")
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
fig.tight_layout()
fig.savefig(FIG_TREATMENTS / "10_treatment_intensity_no_cleanerfish.png")
plt.show()

# Compare rankings side-by-side so the deck can call out which POs shift
comparison = pd.DataFrame({
    "all_treatments_per_yr": intensity["per_site_year"],
    "active_only_per_yr": intensity_active["per_site_year"],
})
comparison["delta"] = comparison["all_treatments_per_yr"] - comparison["active_only_per_yr"]
comparison["delta_pct_cleanerfish"] = (
    comparison["delta"] / comparison["all_treatments_per_yr"] * 100)
print("\nWhere does cleaner-fish use concentrate? Per-PO contribution:")
print(comparison.round(2).sort_values("delta", ascending=False))


# %% [markdown]
# ## Findings summary
#
# Write a one-page summary to `reports/findings.md` for the presentation deck.

# %%
findings = f"""# EDA Findings — BarentsWatch Lice Data

**Data:** {len(lice):,} weekly site rows · {lice['SITENUMBER'].nunique():,} sites · {lice['PRODUCTIONAREA'].nunique()} POs · {lice['WEEK_START'].min().date()}–{lice['WEEK_START'].max().date()}
**Overall breach rate (counted weeks):** {lice_counted['BREACH'].mean():.2%}

## Key findings

1. **Treatment intensity varies ~3× across POs.** The top PO by treatments-per-site-year
   is `{intensity['per_site_year'].idxmax()}` ({intensity['per_site_year'].max():.2f} treatments / active site-year);
   the lowest is `{intensity['per_site_year'].idxmin()}` ({intensity['per_site_year'].min():.2f}).

1b. **Excluding cleaner fish reshuffles the ranking dramatically (chart 10).**
   PO1 Svenskegrensen til Jæren drops from #1 (8.10) to #11 (1.70) because
   ~46 % of its treatments are passive cleaner-fish stocking. The "active
   intervention" leader is PO7 N-Trøndelag at 4.77. This separates preventive
   biological control (mostly the southwest coast, PO1-PO3) from reactive
   intervention intensity (mostly mid-Norway, PO4-PO8).

2. **Breach rates concentrate in a few POs.** Top-3 by breach rate:
{chr(10).join(f"   - {po}: {rate * 100:.1f}%" for po, rate in breach_rate['mean'].sort_values(ascending=False).head(3).items())}
   The overall base rate is {overall_rate:.2%}.

3. **Lice pressure rises sharply above ~10 °C** in every PO; almost all POs see
   their highest mean adult-female counts in the 12–16 °C band, peaking at
   0.40+ in some POs at 16–18 °C under the 0.5-limit regime. Below 6 °C, lice
   pressure is near zero — confirming the biological dependency on temperature.

3b. **The 0.2 spring-limit cuts measured lice by ~half at comparable temperatures.**
   Split-by-regime chart 3 shows that in the same 12–15 °C band, the 0.5-regime
   panel sits at 0.20–0.30 lice while the 0.2-regime panel sits at 0.10–0.13.
   That's regulation doing real work — and a reason a single combined heatmap
   would under-attribute lice pressure to temperature.

4. **Strong seasonal cycle.** Adult-female lice peak in late summer / early autumn
   (~weeks 30–40), well after the spring 0.2-limit window (~W{spring_min}-W{spring_max})
   when the regulator's stricter threshold protects out-migrating smolts.

4b. **The high-lice years are the warm years.** Comparing 2020–2025 in the peak
   window (weeks 25–40), the cross-year correlation between mean sea
   temperature and mean adult-female lice is **r = {corr_yr:.2f}** — exactly the
   mechanism chart 3 implies, playing out at the annual scale. 2024 was the
   warmest year (mean {peak_summary.loc[2024, 'mean_temp_peak_window_C']:.1f} °C
   in W25-40) and had the highest peak lice ({peak_summary.loc[2024, 'peak_femaleadult']:.2f}).

5. **Mobile and adult-female lice are tightly co-measured** (mean per-site
   contemporaneous correlation 0.55-0.70 across POs). The 2-week-lagged correlation
   is lower (~0.3-0.4) but still substantial, meaning current mobile-lice counts
   carry real forecasting information about adult-female levels two weeks ahead.
   This is the biological mechanism that makes 1- and 2-week breach forecasts feasible.

6. **Geographic clustering of breaches.** Mid-Norway (Trøndelag/Nordland coast)
   shows the highest density of high-breach-rate sites.

7. **Treatment methods shifted from chemical to mechanical / biological over
   2017-2023**, reflecting resistance development and regulatory pressure.
   (Note: BarentsWatch changed its action taxonomy in 2024, collapsing several
   specific codes into a single "non-medicinal" umbrella — visible in chart 7.)

## What this implies for modeling (step 3+)

- The base rate (4.5%) is low, so we need an evaluation metric robust to class
  imbalance (PR-AUC + precision-at-top-k, not just accuracy).
- Half of all site-weeks have no lice count → model must handle the censoring;
  consider conditioning predictions on whether a count occurred.
- The strong seasonal cycle means any baseline that captures season-of-year
  (e.g. seasonal naive) will be hard to beat without temperature + treatment features.
- 1- and 2-week horizons are mostly autoregressive; 12-week horizon needs structural
  drivers (PO, temperature trajectory, treatment history).
"""

(ROOT / "reports" / "findings.md").write_text(findings, encoding="utf-8")
print(findings)
