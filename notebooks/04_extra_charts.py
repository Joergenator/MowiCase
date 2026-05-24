# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Extra charts — supplementary maps + breakdowns for the deck
#
# Follow-up charts that didn't fit into the locked Step-2 EDA notebook but
# strengthen specific story beats for the presentation. Every chart still
# uses the leakage-firewalled loader (training data only, < 2026-01-01).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely import MultiPoint, concave_hull
from matplotlib.patches import Polygon as MplPolygon

from src.load_data import load_lice
from src.utils import po_label, short_po_name

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
FIG_DIR.mkdir(parents=True, exist_ok=True)

lice = load_lice()
lice_counted = lice.dropna(subset=["BREACH"])


# ---------------------------------------------------------------------------
# Map helpers — shared between the single-year and multi-year breach charts
# ---------------------------------------------------------------------------

DATA_CRS = ccrs.PlateCarree()
MAP_CRS = ccrs.Mercator(central_longitude=15)


def add_basemap(ax, extent):
    """Add Natural Earth ocean / land / coastline / borders to a cartopy axis."""
    ax.set_extent(extent, crs=DATA_CRS)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#dfeaf2", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#f3efe6", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.5, edgecolor="#666666", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   linewidth=0.4, edgecolor="#888888", linestyle=":",
                   zorder=2)


# Compute one polygon per PO from the actual site locations. Concave hull keeps
# the polygons hugging the coast instead of cutting across open sea (which a
# convex hull would do, e.g. for PO12 Vest-Finnmark's strung-out archipelago).
_po_sites = (
    lice.dropna(subset=["LATITUDE", "LONGITUDE", "PRODUCTIONAREAID"])
    .groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA"])[["LONGITUDE", "LATITUDE"]]
    .agg(list)
    .reset_index()
)

PO_POLYGONS = {}
for _, row in _po_sites.iterrows():
    pts = list(zip(row["LONGITUDE"], row["LATITUDE"]))
    if len(pts) < 3:
        continue
    mp = MultiPoint(pts)
    # ratio=0.4 hugs the points reasonably; 0 = tightest, 1 = convex hull
    hull = concave_hull(mp, ratio=0.4)
    # Buffer slightly so the polygon includes the dots themselves, not just
    # the inter-site convex envelope. ~0.15° ≈ 10-15 km.
    if hull is not None and not hull.is_empty:
        PO_POLYGONS[int(row["PRODUCTIONAREAID"])] = {
            "name": row["PRODUCTIONAREA"],
            "polygon": hull.buffer(0.15),
        }

# Stable color per PO. 13 POs -> tab20 gives enough distinct hues.
_cmap = plt.get_cmap("tab20")
PO_COLORS = {po_id: _cmap(i % 20) for i, po_id in enumerate(sorted(PO_POLYGONS))}


def add_po_zones(ax, alpha=0.18, draw_labels=True, label_fontsize=7):
    """Overlay each PO as a light-alpha polygon on a cartopy axis."""
    for po_id, info in PO_POLYGONS.items():
        poly = info["polygon"]
        color = PO_COLORS[po_id]
        polygons = (list(poly.geoms) if poly.geom_type == "MultiPolygon"
                    else [poly])
        for p in polygons:
            xs, ys = p.exterior.xy
            patch = MplPolygon(list(zip(xs, ys)),
                               facecolor=color, alpha=alpha,
                               edgecolor=color, linewidth=0.8,
                               transform=DATA_CRS, zorder=2.5)
            ax.add_patch(patch)
        if draw_labels:
            cx, cy = poly.centroid.x, poly.centroid.y
            ax.text(cx, cy, f"PO{po_id}", transform=DATA_CRS,
                    fontsize=label_fontsize, fontweight="bold",
                    ha="center", va="center", color="#333333",
                    zorder=2.7,
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", alpha=0.7,
                              edgecolor="none"))

# %% [markdown]
# ## Chart of breaches in 2024
#
# **Case Q3 ("repeated breaches") + Q4 (geography):** All-time breach counts
# stack 14 years of data on one map and the picture gets muddy — most active
# sites cross any low threshold eventually. Restricting to a single recent year
# (2024 — the warmest year in the record, and the year with the highest peak
# lice; see chart 9) gives a sharp, current-state map.
#
# Each site is plotted by its mean lat/lon. Sites that were counted in 2024 but
# had **zero breach weeks** appear in faded gray as a backdrop, so the reader
# can read both "where the industry operated last year" and "where breaches
# happened." Sites with at least one 2024 breach are colored by their 2024
# breach count.

# %%
YEAR = 2024

lice_year = lice_counted[lice_counted["WEEK_START"].dt.year == YEAR]

site_year = (
    lice_year
    .groupby("SITENUMBER")
    .agg(breach_weeks=("BREACH", "sum"),
         counted_weeks=("BREACH", "size"),
         lat=("LATITUDE", "mean"),
         lon=("LONGITUDE", "mean"),
         PRODUCTIONAREA=("PRODUCTIONAREA", "first"))
    .dropna(subset=["lat", "lon"])
)
# nullable-bool sum -> float; cast for clean labels
site_year["breach_weeks"] = site_year["breach_weeks"].astype(int)

breached = site_year[site_year["breach_weeks"] >= 1]
zero = site_year[site_year["breach_weeks"] == 0]

print(f"{YEAR}: {len(site_year):,} sites counted; "
      f"{len(breached):,} ({len(breached) / len(site_year):.1%}) breached at "
      f"least once.")
print(f"Among breached sites - max breach weeks: "
      f"{breached['breach_weeks'].max()}, "
      f"median: {breached['breach_weeks'].median():.0f}")

# Map extent from data + small margin
_lat_min = min(zero["lat"].min(), breached["lat"].min()) - 0.3
_lat_max = max(zero["lat"].max(), breached["lat"].max()) + 0.3
_lon_min = min(zero["lon"].min(), breached["lon"].min()) - 0.5
_lon_max = max(zero["lon"].max(), breached["lon"].max()) + 0.5

fig, ax = plt.subplots(figsize=(9, 11),
                       subplot_kw={"projection": MAP_CRS})
add_basemap(ax, [_lon_min, _lon_max, _lat_min, _lat_max])
add_po_zones(ax)

# Backdrop: sites active in 2024 with zero breaches
ax.scatter(zero["lon"], zero["lat"],
           s=14, color="dimgray", alpha=0.55, edgecolors="none",
           transform=DATA_CRS, zorder=3,
           label=f"0 breach weeks (n={len(zero):,})")

# Foreground: sites with ≥ 1 breach, colored by count
vmax = max(2, np.percentile(breached["breach_weeks"], 95))
sc = ax.scatter(breached["lon"], breached["lat"],
                c=breached["breach_weeks"],
                s=42, cmap="YlOrRd", edgecolors="grey", linewidths=0.3,
                alpha=0.9, vmin=1, vmax=vmax,
                transform=DATA_CRS, zorder=4,
                label=f"≥ 1 breach week (n={len(breached):,})")
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, extend="max")
cb.set_label(f"Breach weeks in {YEAR} (clipped at {vmax:.0f})")

gl = ax.gridlines(crs=DATA_CRS, draw_labels=True,
                  linewidth=0.3, color="grey", alpha=0.4)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 9}
gl.ylabel_style = {"size": 9}

ax.set_title(f"Chart of breaches in {YEAR}")
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
fig.tight_layout()
fig.savefig(FIG_DIR / f"chart_of_breaches_{YEAR}.png")
plt.show()

# Per-PO breakdown for the deck
print(f"\nBreach weeks in {YEAR}, by PO (sites with >= 1 breach):")
print((breached.groupby("PRODUCTIONAREA")
       .agg(n_sites=("breach_weeks", "size"),
            total_breach_weeks=("breach_weeks", "sum"))
       .sort_values("total_breach_weeks", ascending=False)
       .to_string()))


# %% [markdown]
# ## Chart of breaches 2020-2025 (year-by-year panels, 2×3 layout)
#
# Same recipe as the 2024 chart but one panel per year, so the reader can see
# the year-over-year shift in where breaches happen.
#
# 2026 is **deliberately excluded**: the case rule is "no 2026 for training or
# feature engineering" and we also treat the 21,301 Jan-May 2026 rows as a
# fully-unseen final validation set. Looking at them now — even for a chart —
# would taint that. So this chart uses the firewalled loader (training data
# only, < 2026-01-01) and the year range is capped at 2025.

# %%
lice_counted_fw = lice_counted  # already firewalled by load_lice(); just alias

YEARS = list(range(2020, 2026))  # 2020..2025 inclusive (6 years, 2×3 grid)

# Pre-compute per-(site, year) breach counts so we can pick a common vmax
per_site_year = (
    lice_counted_fw.assign(YEAR_=lice_counted_fw["WEEK_START"].dt.year)
    .query("YEAR_ in @YEARS")
    .groupby(["YEAR_", "SITENUMBER"])
    .agg(breach_weeks=("BREACH", "sum"),
         lat=("LATITUDE", "mean"),
         lon=("LONGITUDE", "mean"))
    .dropna(subset=["lat", "lon"])
    .reset_index()
)
per_site_year["breach_weeks"] = per_site_year["breach_weeks"].astype(int)

# Common color scale across years: clip at 95th percentile of breaching sites
breached_all = per_site_year[per_site_year["breach_weeks"] >= 1]
vmax_shared = max(2, int(np.percentile(breached_all["breach_weeks"], 95)))

# Common axis limits so panels are spatially aligned
lon_min, lon_max = per_site_year["lon"].min() - 0.5, per_site_year["lon"].max() + 0.5
lat_min, lat_max = per_site_year["lat"].min() - 0.3, per_site_year["lat"].max() + 0.3

fig, axes = plt.subplots(2, 3, figsize=(15, 12),
                         subplot_kw={"projection": MAP_CRS},
                         gridspec_kw={"wspace": 0.06, "hspace": 0.12})
axes_flat = axes.flatten()

sc = None
for ax, yr in zip(axes_flat, YEARS):
    yr_df = per_site_year[per_site_year["YEAR_"] == yr]
    zero = yr_df[yr_df["breach_weeks"] == 0]
    breached = yr_df[yr_df["breach_weeks"] >= 1]

    add_basemap(ax, [lon_min, lon_max, lat_min, lat_max])
    # Labels off here so 6 panels don't get cluttered; PO zones still visible
    add_po_zones(ax, alpha=0.18, draw_labels=False)

    ax.scatter(zero["lon"], zero["lat"],
               s=8, color="dimgray", alpha=0.45, edgecolors="none",
               transform=DATA_CRS, zorder=3)
    sc = ax.scatter(breached["lon"], breached["lat"],
                    c=breached["breach_weeks"],
                    s=26, cmap="YlOrRd", edgecolors="grey", linewidths=0.25,
                    alpha=0.9, vmin=1, vmax=vmax_shared,
                    transform=DATA_CRS, zorder=4)

    gl = ax.gridlines(crs=DATA_CRS, draw_labels=True,
                      linewidth=0.3, color="grey", alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    ax.set_title(f"{yr}  —  n={len(breached):,} breached", fontsize=12)

# One shared colorbar on the right
cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02, extend="max")
cb.set_label(f"Breach weeks in year (clipped at {vmax_shared})")

fig.suptitle("Chart of breaches 2020-2025 (one panel per year)",
             fontsize=15, fontweight="bold", y=0.96)
fig.savefig(FIG_DIR / "chart_of_breaches_2020_2025.png")
plt.show()


# %% [markdown]
# ## Sjøtemperatur i PO 9-12 (Nord-Norge), 2024
#
# **Temperaturoversikt for de fire nordligste produksjonsområdene i 2024:**
# PO9 Vestfjorden og Vesterålen, PO10 Andøya til Senja, PO11 Kvaløya til Loppa,
# PO12 Vest-Finnmark. Hver linje viser ukentlig snitt-temperatur på tvers av
# alle sites i området. Skraverte bånd viser 10-90 %-spredningen mellom sites.
#
# 2024 var det varmeste året i hele perioden (snitt 14.08 °C i uke 25-40, se
# chart 9) — den nordlige delen av kysten følger samme sesongprofil som
# sør-Norge, men er kjøligere og toppen kommer noen uker senere.

# %%
NORTH_POS = [9, 10, 11, 12]
lice_2024 = lice[lice["WEEK_START"].dt.year == 2024].copy()
lice_north = lice_2024[lice_2024["PRODUCTIONAREAID"].isin(NORTH_POS)]

# Aggregate weekly per PO: mean, p10, p90
weekly = (
    lice_north.dropna(subset=["SEATEMPERATURE"])
    .assign(iso_week=lice_north["WEEK_START"].dt.isocalendar().week.astype(int))
    .groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA", "iso_week"])
    .agg(temp_mean=("SEATEMPERATURE", "mean"),
         temp_p10=("SEATEMPERATURE", lambda s: s.quantile(0.10)),
         temp_p90=("SEATEMPERATURE", lambda s: s.quantile(0.90)),
         n_obs=("SEATEMPERATURE", "size"))
    .reset_index()
)

# Drop weeks with very few observations (avoid noisy endpoints)
weekly = weekly[weekly["n_obs"] >= 3]

PO_COLORS = {9: "#1f77b4", 10: "#2ca02c", 11: "#ff7f0e", 12: "#d62728"}

fig, ax = plt.subplots(figsize=(13, 7))

for po_id in NORTH_POS:
    sub = weekly[weekly["PRODUCTIONAREAID"] == po_id].sort_values("iso_week")
    if sub.empty:
        continue
    po_name = sub["PRODUCTIONAREA"].iloc[0]
    label = po_label(po_id, po_name)
    color = PO_COLORS[po_id]
    ax.fill_between(sub["iso_week"], sub["temp_p10"], sub["temp_p90"],
                    color=color, alpha=0.12, linewidth=0)
    ax.plot(sub["iso_week"], sub["temp_mean"],
            color=color, linewidth=2.2, marker="o", markersize=4,
            label=label)

ax.axhline(8, color="grey", linestyle="--", linewidth=1, alpha=0.6)
ax.text(1.5, 8.2, "8 °C (cleaner-fish viability)", fontsize=9, color="grey")

ax.set_xlabel("ISO uke (2024)")
ax.set_ylabel("Sjøtemperatur (°C)")
ax.set_title("Sjøtemperatur i PO 9-12 (Nord-Norge), 2024 — ukentlig snitt per område")
ax.set_xlim(1, 52)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "temp_po9_12_2024.png")
plt.show()

# Year summary by PO — useful numbers for the deck
print("\nSjøtemperatur 2024 — PO 9-12:")
summary_temp = (
    lice_north.dropna(subset=["SEATEMPERATURE"])
    .groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA"])
    .agg(mean_temp=("SEATEMPERATURE", "mean"),
         peak_temp=("SEATEMPERATURE", "max"),
         peak_week=("SEATEMPERATURE", "idxmax"),
         n_obs=("SEATEMPERATURE", "size"),
         n_sites=("SITENUMBER", "nunique"))
)
summary_temp["peak_week"] = (
    lice_north.loc[summary_temp["peak_week"], "WEEK_START"]
    .dt.isocalendar().week.values
)
print(summary_temp.round(2).to_string())


# %% [markdown]
# ## Sjøtemperatur i PO 9-12, 2019-2024 (6 år, 3×2 layout)
#
# Én panel per år 2019-2024 i 3-rader × 2-kolonner format. Lar leseren se
# hele baseline-perioden 2019-2023 sammen med det uvanlig varme 2024 i samme
# grid — den varme outlieren blir tydelig i kontrast til de fem foregående
# årene. 8 °C-linjen (rensefisk-overlevelse) holdes som referanse på alle
# paneler.

# %%
HIST_YEARS = list(range(2019, 2025))  # 2019..2024 inclusive (6 panels for 3×2)

lice_north_hist = lice[
    (lice["WEEK_START"].dt.year.isin(HIST_YEARS))
    & (lice["PRODUCTIONAREAID"].isin(NORTH_POS))
].copy()

PO_LINE_COLORS = {9: "#1f77b4", 10: "#2ca02c", 11: "#ff7f0e", 12: "#d62728"}

# Pre-compute weekly aggregations once per (year, PO)
weekly_hist = (
    lice_north_hist.dropna(subset=["SEATEMPERATURE"])
    .assign(year=lice_north_hist["WEEK_START"].dt.year,
            iso_week=lice_north_hist["WEEK_START"].dt.isocalendar().week.astype(int))
    .groupby(["year", "PRODUCTIONAREAID", "PRODUCTIONAREA", "iso_week"])
    .agg(temp_mean=("SEATEMPERATURE", "mean"),
         temp_p10=("SEATEMPERATURE", lambda s: s.quantile(0.10)),
         temp_p90=("SEATEMPERATURE", lambda s: s.quantile(0.90)),
         n_obs=("SEATEMPERATURE", "size"))
    .reset_index()
)
weekly_hist = weekly_hist[weekly_hist["n_obs"] >= 3]

fig, axes = plt.subplots(3, 2, figsize=(14, 13),
                         sharey=True, sharex=True,
                         gridspec_kw={"wspace": 0.05, "hspace": 0.20})
axes_flat = axes.flatten()

for ax, yr in zip(axes_flat, HIST_YEARS):
    yr_df = weekly_hist[weekly_hist["year"] == yr]
    for po_id in NORTH_POS:
        sub = yr_df[yr_df["PRODUCTIONAREAID"] == po_id].sort_values("iso_week")
        if sub.empty:
            continue
        po_name = sub["PRODUCTIONAREA"].iloc[0]
        color = PO_LINE_COLORS[po_id]
        ax.fill_between(sub["iso_week"], sub["temp_p10"], sub["temp_p90"],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(sub["iso_week"], sub["temp_mean"],
                color=color, linewidth=2.0, marker="o", markersize=3,
                label=po_label(po_id, po_name) if ax is axes_flat[0] else None)

    ax.axhline(8, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    # Mark 2024 panel with a subtle accent so its outlier status is obvious
    title_color = "#b3261e" if yr == 2024 else "#1c1c1c"
    title_weight = "bold" if yr == 2024 else "normal"
    ax.set_title(f"{yr}" + ("  (warmest in record)" if yr == 2024 else ""),
                 fontsize=12, color=title_color, fontweight=title_weight)
    ax.set_xlim(1, 52)
    ax.grid(True, alpha=0.3)

# Axis labels only on outer rows / columns
for ax in axes[:, 0]:
    ax.set_ylabel("Sjøtemperatur (°C)")
for ax in axes[-1, :]:
    ax.set_xlabel("ISO uke")

axes_flat[0].legend(loc="upper left", fontsize=9, framealpha=0.95)

fig.suptitle("Sjøtemperatur i PO 9-12 (Nord-Norge), 2019-2024 — ukentlig snitt per område",
             fontsize=14, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig(FIG_DIR / "temp_po9_12_2019_2024.png")
plt.show()

# Year-by-year peak summary for the deck
print("\nÅrlig snitt + p95 (uke 25-35) for PO 9-12, 2019-2024:")
peak_window_summary = (
    lice_north_hist.dropna(subset=["SEATEMPERATURE"])
    .assign(year=lice_north_hist["WEEK_START"].dt.year,
            iso_week=lice_north_hist["WEEK_START"].dt.isocalendar().week.astype(int))
    .query("25 <= iso_week <= 35")
    .groupby(["year", "PRODUCTIONAREAID", "PRODUCTIONAREA"])
    .agg(peak_window_mean=("SEATEMPERATURE", "mean"),
         peak_week_p95=("SEATEMPERATURE", lambda s: s.quantile(0.95)))
    .round(2)
)
print(peak_window_summary.to_string())


# %% [markdown]
# ## Temperatur og lus i PO 1-2 vs PO 3-4, 2024 (2×2)
#
# Sør-Norge har en lengre varm sesong enn nord (chart-en over). Dette 2×2-
# grid-et viser:
# - Topp-rad: ukentlig **sjøtemperatur** per PO i 2024.
# - Bunn-rad: ukentlig **FEMALEADULT lus per fisk** per PO i 2024.
# - Venstre kolonne: PO1 (Svenskegrensen til Jæren) + PO2 (Ryfylket).
# - Høyre kolonne: PO3 (Karmøy til Sotra) + PO4 (Nordhordaland til Stadt).
#
# Lese-rytme: vertikalt for "temperaturen driver lus" innenfor en region,
# horisontalt for nord-vs-sør-sammenlikning på samme tid.

# %%
SOUTH_POS_PAIRS = [
    ("PO 1-2  (Svenskegrensen til Jæren, Ryfylket)", [1, 2]),
    ("PO 3-4  (Karmøy til Sotra, Nordhordaland til Stadt)", [3, 4]),
]

PO_PAIR_COLORS = {
    1: "#1f77b4",  # PO1 blue
    2: "#2ca02c",  # PO2 green
    3: "#ff7f0e",  # PO3 orange
    4: "#d62728",  # PO4 red
}


def weekly_agg(df, value_col, group_cols):
    """Weekly aggregate (mean, p10, p90, n_obs) per group, dropping nulls."""
    sub = df.dropna(subset=[value_col]).copy()
    sub["iso_week"] = sub["WEEK_START"].dt.isocalendar().week.astype(int)
    agg = (sub.groupby(group_cols + ["iso_week"])
              .agg(value_mean=(value_col, "mean"),
                   value_p10=(value_col, lambda s: s.quantile(0.10)),
                   value_p90=(value_col, lambda s: s.quantile(0.90)),
                   n_obs=(value_col, "size"))
              .reset_index())
    return agg[agg["n_obs"] >= 3]


lice_2024_south = lice[
    (lice["WEEK_START"].dt.year == 2024)
    & (lice["PRODUCTIONAREAID"].isin([1, 2, 3, 4]))
].copy()

fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                         sharex=True,
                         gridspec_kw={"wspace": 0.18, "hspace": 0.25})

# Temperature row (shared y across the two temp panels)
temp_ax_pair = axes[0, :]
# Lice row (shared y across the two lice panels)
lice_ax_pair = axes[1, :]
temp_ax_pair[1].sharey(temp_ax_pair[0])
lice_ax_pair[1].sharey(lice_ax_pair[0])

# ---- Top row: temperature ----
for ax, (title, po_ids) in zip(temp_ax_pair, SOUTH_POS_PAIRS):
    df_pair = lice_2024_south[lice_2024_south["PRODUCTIONAREAID"].isin(po_ids)]
    agg = weekly_agg(df_pair, "SEATEMPERATURE",
                     ["PRODUCTIONAREAID", "PRODUCTIONAREA"])
    for po_id in po_ids:
        sub = agg[agg["PRODUCTIONAREAID"] == po_id].sort_values("iso_week")
        if sub.empty:
            continue
        color = PO_PAIR_COLORS[po_id]
        ax.fill_between(sub["iso_week"], sub["value_p10"], sub["value_p90"],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(sub["iso_week"], sub["value_mean"],
                color=color, linewidth=2.0, marker="o", markersize=3,
                label=po_label(po_id, sub["PRODUCTIONAREA"].iloc[0]))
    ax.axhline(8, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Sjøtemperatur (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_xlim(1, 52)

# ---- Bottom row: lice (FEMALEADULT) ----
for ax, (title, po_ids) in zip(lice_ax_pair, SOUTH_POS_PAIRS):
    df_pair = lice_2024_south[lice_2024_south["PRODUCTIONAREAID"].isin(po_ids)]
    agg = weekly_agg(df_pair, "FEMALEADULT",
                     ["PRODUCTIONAREAID", "PRODUCTIONAREA"])
    for po_id in po_ids:
        sub = agg[agg["PRODUCTIONAREAID"] == po_id].sort_values("iso_week")
        if sub.empty:
            continue
        color = PO_PAIR_COLORS[po_id]
        ax.fill_between(sub["iso_week"], sub["value_p10"], sub["value_p90"],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(sub["iso_week"], sub["value_mean"],
                color=color, linewidth=2.0, marker="o", markersize=3,
                label=po_label(po_id, sub["PRODUCTIONAREA"].iloc[0]))
    # Regulatory thresholds: 0.5 normal, 0.2 in spring window (W16-22)
    ax.axhline(0.5, color="#cc0000", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(51.5, 0.51, "0.5 limit", fontsize=8, color="#cc0000",
            ha="right", va="bottom")
    ax.axhline(0.2, color="#cc0000", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(51.5, 0.21, "0.2 spring limit", fontsize=8, color="#cc0000",
            ha="right", va="bottom")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("FEMALEADULT lus / fisk")
    ax.set_xlabel("ISO uke (2024)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_xlim(1, 52)

fig.suptitle("Sør-Norge 2024 — temperatur (topp) og lus (bunn) per PO-par",
             fontsize=14, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig(FIG_DIR / "temp_lice_po1_4_2024.png")
plt.show()

# Print summary for the deck
print("\nÅrssnitt 2024 — PO 1-4:")
summary_south = (
    lice_2024_south.dropna(subset=["SEATEMPERATURE", "FEMALEADULT"], how="all")
    .groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA"])
    .agg(mean_temp_C=("SEATEMPERATURE", "mean"),
         peak_temp_C=("SEATEMPERATURE", "max"),
         mean_femaleadult=("FEMALEADULT", "mean"),
         peak_femaleadult=("FEMALEADULT", "max"),
         n_sites=("SITENUMBER", "nunique"))
    .round(2)
)
print(summary_south.to_string())

print("\nYear-by-year summary:")
summary = (per_site_year.groupby("YEAR_")
           .agg(sites_counted=("SITENUMBER", "nunique"),
                sites_breached=("breach_weeks", lambda s: (s >= 1).sum()),
                total_breach_weeks=("breach_weeks", "sum"),
                max_at_site=("breach_weeks", "max")))
summary["pct_breached"] = (summary["sites_breached"]
                           / summary["sites_counted"] * 100).round(1)
print(summary)
