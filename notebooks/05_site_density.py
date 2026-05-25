# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lokaliteter — antall og tetthet per PO
#
# Egen notebook for spørsmål om struktur i lokalitetslandskapet, separert fra
# `04_extra_charts.py` så vi slipper å re-kjøre alle breach-/temp-figurene
# hver gang vi vil legge til en ny analyse.
#
# Bruker den firewall-beskyttede loaderen (`apply_cutoff=True`, < 2026-01-01).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree

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

FIG_DIR = ROOT / "reports" / "figures" / "Site-forhold"
FIG_DIR.mkdir(parents=True, exist_ok=True)

lice = load_lice()
EARTH_R_KM = 6371.0088


# %% [markdown]
# ## Chart 1 — Antall aktive lokaliteter per PO i 2024
#
# "Aktiv" = lokalitet med minst én ukentlig observasjon i 2024 (fra `vlice`).
# Dette teller alle steder hvor det ble rapportert lus-tall eller status (også
# fallow-uker), så det reflekterer hvor mange anlegg som *eksisterte og var
# overvåket* — ikke hvor mange som hadde fisk i sjøen til enhver tid.

# %%
YEAR = 2024
lice_2024 = lice[lice["WEEK_START"].dt.year == YEAR]

active_per_po = (
    lice_2024.dropna(subset=["PRODUCTIONAREAID"])
    .groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA"])["SITENUMBER"]
    .nunique()
    .rename("n_active_sites")
    .reset_index()
    .sort_values("n_active_sites", ascending=True)
)
active_per_po["label"] = active_per_po.apply(
    lambda r: po_label(r["PRODUCTIONAREAID"], r["PRODUCTIONAREA"]), axis=1
)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(active_per_po["label"], active_per_po["n_active_sites"],
               color="#1f77b4")
ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
ax.set_xlabel(f"Antall aktive lokaliteter i {YEAR}")
ax.set_title(f"Aktive lokaliteter per produksjonsområde, {YEAR}")
ax.set_xlim(0, active_per_po["n_active_sites"].max() * 1.12)
fig.tight_layout()
fig.savefig(FIG_DIR / f"active_sites_per_po_{YEAR}.png")
plt.show()

total_active = active_per_po["n_active_sites"].sum()
print(f"\nTotalt antall aktive lokaliteter i {YEAR}: {total_active:,}")
print(f"Snitt per PO: {active_per_po['n_active_sites'].mean():.1f}")
print(f"\nTopp 3 PO etter antall lokaliteter:")
print(active_per_po.sort_values("n_active_sites", ascending=False)
      .head(3)[["label", "n_active_sites"]].to_string(index=False))


# %% [markdown]
# ## Chart 2 — Klyngedannelse: hvilke PO har lokalitetene tettest samlet?
#
# **Spørsmål:** Hvor i Norge ligger oppdrettsanlegg tettest sammen?
# Klynger betyr potensielt høyere smittepress mellom anlegg via vannmasser.
#
# **Mål:** For hver lokalitet teller vi antall **andre** lokaliteter innenfor
# 5 km radius (haversine på lat/lon). PO-grensene ignoreres — naboer på tvers
# av PO-grenser teller med (lus bryr seg ikke om administrative grenser).
# Snittet beregnes på (år, lokalitet)-nivå over 2020-2025 og aggregeres per PO.
#
# To paneler:
#   1. Stolpediagram: snitt naboer-innen-5km per PO (sortert, mest tett øverst).
#   2. Kart av Norge: hver lokalitet farget etter sin egen naboscore.

# %%
from src.map_utils import add_basemap, get_crs

RADIUS_KM = 5.0


def neighbors_within(lat: np.ndarray, lon: np.ndarray, radius_km: float) -> np.ndarray:
    """Per point, count *other* points within radius_km using haversine."""
    if len(lat) < 2:
        return np.zeros(len(lat), dtype=int)
    coords_rad = np.radians(np.column_stack([lat, lon]))
    tree = BallTree(coords_rad, metric="haversine")
    radius_rad = radius_km / EARTH_R_KM
    counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True)
    # Subtract 1 because each point is its own nearest "neighbor"
    return counts - 1


def build_clustering_chart(years: range, lice_df: pd.DataFrame) -> None:
    """Compute per-PO clustering metric and render the two-panel chart.

    For every year in `years`, count each site's neighbors within RADIUS_KM,
    then average across (year, site) within each PO. Renders a bar ranking
    plus a Norway-coastline map of per-site density, saved as
    `site_clustering_per_po_{ystart}_{yend}.png` in FIG_DIR.
    """
    per_year_records = []
    for yr in years:
        lice_y = lice_df[lice_df["WEEK_START"].dt.year == yr]
        sites_y = (lice_y.dropna(subset=["LATITUDE", "LONGITUDE", "PRODUCTIONAREAID"])
                         .groupby(["SITENUMBER", "PRODUCTIONAREAID", "PRODUCTIONAREA"])
                         .agg(lat=("LATITUDE", "mean"), lon=("LONGITUDE", "mean"))
                         .reset_index())
        if len(sites_y) == 0:
            continue
        sites_y["n_neighbors_5km"] = neighbors_within(
            sites_y["lat"].values, sites_y["lon"].values, RADIUS_KM,
        )
        sites_y["YEAR"] = yr
        per_year_records.append(sites_y)

    per_year = pd.concat(per_year_records, ignore_index=True)

    po_summary = (
        per_year.groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA"])
        .agg(mean_neighbors=("n_neighbors_5km", "mean"),
             median_neighbors=("n_neighbors_5km", "median"),
             site_years=("SITENUMBER", "size"),
             unique_sites=("SITENUMBER", "nunique"))
        .reset_index()
        .sort_values("mean_neighbors", ascending=True)
    )
    po_summary["label"] = po_summary.apply(
        lambda r: po_label(r["PRODUCTIONAREAID"], r["PRODUCTIONAREA"]), axis=1)

    site_summary = (
        per_year.groupby(["SITENUMBER", "PRODUCTIONAREAID", "PRODUCTIONAREA"])
        .agg(mean_neighbors=("n_neighbors_5km", "mean"),
             lat=("lat", "mean"), lon=("lon", "mean"))
        .reset_index()
    )

    ystart, yend = min(years), max(years)
    print(f"\nKlyngedannelse: snitt naboer innen {RADIUS_KM:.0f} km per PO, "
          f"snitt over {ystart}-{yend}:")
    print(po_summary.sort_values("mean_neighbors", ascending=False)
          [["label", "mean_neighbors", "median_neighbors", "unique_sites"]]
          .round(2).to_string(index=False))

    DATA_CRS, MAP_CRS = get_crs()

    fig = plt.figure(figsize=(16, 10))
    ax_bar = fig.add_subplot(1, 2, 1)
    ax_map = fig.add_subplot(1, 2, 2, projection=MAP_CRS)

    bar_colors = ["#d62728" if v >= po_summary["mean_neighbors"].median() else "#1f77b4"
                  for v in po_summary["mean_neighbors"]]
    bars = ax_bar.barh(po_summary["label"], po_summary["mean_neighbors"],
                       color=bar_colors)
    ax_bar.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax_bar.set_xlabel(f"Snitt antall naboer innen {RADIUS_KM:.0f} km per lokalitet")
    ax_bar.set_title(f"Klyngedannelse per PO  ({ystart}-{yend} snitt)")
    ax_bar.set_xlim(0, po_summary["mean_neighbors"].max() * 1.15)

    extent = (
        site_summary["lon"].min() - 1.0, site_summary["lon"].max() + 1.0,
        site_summary["lat"].min() - 0.5, site_summary["lat"].max() + 0.5,
    )
    add_basemap(ax_map, extent)

    vmax = np.percentile(site_summary["mean_neighbors"], 95)
    sc = ax_map.scatter(
        site_summary["lon"], site_summary["lat"],
        c=site_summary["mean_neighbors"], cmap="YlOrRd",
        s=22, alpha=0.9, edgecolors="grey", linewidths=0.25,
        vmin=0, vmax=vmax,
        transform=DATA_CRS, zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax_map, shrink=0.6, pad=0.02, extend="max")
    cb.set_label(f"Naboer innen {RADIUS_KM:.0f} km (snitt {ystart}-{yend}), "
                 f"fargeskala klippet ved p95={vmax:.0f}")
    ax_map.set_title(f"Hvor klyngene faktisk ligger  (n={len(site_summary)} lokaliteter)")

    fig.suptitle(
        f"Hvilke produksjonsområder har tettest klynger av lokaliteter?  "
        f"({ystart}-{yend})",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"site_clustering_per_po_{ystart}_{yend}.png")
    plt.show()


# Recent operational snapshot — matches the chart we already had
build_clustering_chart(range(2020, 2026), lice)

# Full-history view — every year in the cleaned dataset
build_clustering_chart(range(2012, 2026), lice)
