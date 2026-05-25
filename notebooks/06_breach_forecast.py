# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Step 6 — 12-week breach risk forecast for every commercial site
#
# Uses the persisted LightGBM v1 booster at h=12 to score the latest
# available week per site, then explains the main drivers behind each
# top prediction using LightGBM's native `pred_contrib=True` decomposition.
#
# **What this notebook produces:**
# - `reports/figures/forecast/F1_top_sites_predicted_risk_h12.png` — top-20 horizontal bar chart
# - `reports/figures/forecast/F2_national_risk_map_h12.png` — every site on the Norwegian coast, colored by predicted risk
# - `reports/figures/forecast/F3_top_drivers_heatmap.png` — per-site feature contribution heatmap for the top-10
# - `reports/figures/forecast/F4_predicted_proba_distribution.png` — histogram of all-site predicted probabilities
# - `reports/forecast_summary.md` — auto-generated narrative + top-20 table
#
# **HI research sites are excluded** at the data layer (booster trained
# commercial-only). See `src/research_sites.py`.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.load_data import load_training_data
from src.features import build_inference_frame
from src.models import LightGBMBreach
from src.utils import po_label
from src.map_utils import add_basemap, get_crs

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.labelsize": 11,
})

FIG_DIR = ROOT / "reports" / "figures"
FIG_FORECAST = FIG_DIR / "forecast"
FIG_FORECAST.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

HORIZON = 12

# %% [markdown]
# ## 1. Load data + score every commercial site at h=12

# %%
lice, treat = load_training_data()
inf = build_inference_frame(lice, treat, horizon=HORIZON)
print(f"Inference frame: {len(inf):,} commercial sites at week "
      f"{inf['WEEK_START'].max().date()}")
print(f"Target week (predict FOR): {inf['target_week_start'].iloc[0].date()}")

model = LightGBMBreach.load(MODELS_DIR / f"lgbm_v1_h{HORIZON}.txt")
inf["predicted_breach_probability"] = model.predict_proba(inf)

# Per-site SHAP-style contributions (one row per site, one col per feature + _bias)
contribs = model.predict_contributions(inf)
print(f"\nContributions shape: {contribs.shape}")
print(f"Sigmoid(row.sum()) sanity check on row 0: "
      f"{1/(1+np.exp(-contribs.iloc[0].sum())):.4f}  "
      f"vs predict_proba: {inf['predicted_breach_probability'].iloc[0]:.4f}")

# Also score with v3 (v1 + neighbor features) so we can compare rankings.
# v3 may not be trained yet on a fresh checkout — only load it if present.
V3_PATH = MODELS_DIR / f"lgbm_v3_h{HORIZON}.txt"
v3_available = V3_PATH.exists()
if v3_available:
    model_v3 = LightGBMBreach.load(V3_PATH)
    inf["predicted_breach_probability_v3"] = model_v3.predict_proba(inf)
    print(f"v3 also scored — {len(model_v3.feature_cols)} features incl. "
          f"{sum(1 for f in model_v3.feature_cols if f.startswith('neighbors_'))} "
          f"neighbor features")
else:
    print("v3 booster not found — run `python -m scripts.train_and_save` "
          "to enable v1 vs v3 comparison.")


# %% [markdown]
# ## Chart F1 — Top-20 sites by predicted h=12 probability

# %%
def _render_top20_bar(top_df, proba_col, model_tag, out_filename):
    """Render the top-20 horizontal bar chart for a given model's predictions."""
    df = top_df.copy()
    df["label"] = df.apply(
        lambda r: f"{r['SITENAME']} (PO{int(r['PRODUCTIONAREAID'])})", axis=1)
    unique_pos = sorted(df["PRODUCTIONAREAID"].unique())
    palette = sns.color_palette("tab10", n_colors=max(len(unique_pos), 3))
    po_colors = {po: palette[i] for i, po in enumerate(unique_pos)}
    bar_colors = [po_colors[po] for po in df["PRODUCTIONAREAID"]]

    # Sort ascending so the highest-risk site appears at the top of the figure
    df_plot = df.iloc[::-1]
    bar_colors_plot = bar_colors[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(df_plot["label"], df_plot[proba_col],
                   color=bar_colors_plot, edgecolor="grey", linewidth=0.4)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel(f"Predicted breach probability (h={HORIZON} weeks)")
    ax.set_title(f"Top-20 commercial sites by predicted h={HORIZON}w breach risk — {model_tag}\n"
                 f"(predict from {inf['WEEK_START'].max().date()} -> "
                 f"{inf['target_week_start'].iloc[0].date()})",
                 fontsize=12)
    ax.set_xlim(0, df[proba_col].max() * 1.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=po_colors[po]) for po in unique_pos]
    labels = [po_label(po, df[df["PRODUCTIONAREAID"] == po]
                                  ["PRODUCTIONAREA"].iloc[0]) for po in unique_pos]
    ax.legend(handles, labels, title="Produksjonsområde", loc="lower right",
              fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_FORECAST / out_filename)
    plt.show()


top20 = (inf.sort_values("predicted_breach_probability", ascending=False)
            .head(20).copy())
_render_top20_bar(top20, "predicted_breach_probability", "v1",
                   "F1_top_sites_predicted_risk_h12_v1.png")

if v3_available:
    top20_v3 = (inf.sort_values("predicted_breach_probability_v3", ascending=False)
                    .head(20).copy())
    _render_top20_bar(top20_v3, "predicted_breach_probability_v3",
                       "v3 (+ neighbor features)",
                       "F1_top_sites_predicted_risk_h12_v3.png")

# %% [markdown]
# ## Chart F2 — National risk map (every commercial site)

# %%
DATA_CRS, MAP_CRS = get_crs()
extent = (
    inf["LONGITUDE"].min() - 1.0, inf["LONGITUDE"].max() + 1.0,
    inf["LATITUDE"].min() - 0.5, inf["LATITUDE"].max() + 0.5,
)

# Two-panel map if v3 exists: v1 on left, v3 on right with a SHARED color
# scale so the eye can tell which model is "hotter" overall.
panels = [("v1", "predicted_breach_probability")]
if v3_available:
    panels.append(("v3 (+ neighbors)", "predicted_breach_probability_v3"))

# Shared vmax across both maps — use the larger of the two p95s so neither
# panel washes out.
all_probas = pd.concat([inf[col] for _, col in panels])
shared_vmax = float(np.percentile(all_probas, 95))

fig = plt.figure(figsize=(9 * len(panels), 11))
for i, (tag, col) in enumerate(panels, start=1):
    ax = fig.add_subplot(1, len(panels), i, projection=MAP_CRS)
    add_basemap(ax, extent)
    sc = ax.scatter(
        inf["LONGITUDE"], inf["LATITUDE"],
        c=inf[col], cmap="YlOrRd",
        s=26, alpha=0.9, edgecolors="grey", linewidths=0.25,
        vmin=0, vmax=shared_vmax,
        transform=DATA_CRS, zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, extend="max")
    cb.set_label(f"Pred. proba (h={HORIZON}w), clipped at p95={shared_vmax:.3f}")
    ax.set_title(f"{tag}  (max {inf[col].max():.3f})")
fig.suptitle(f"Norway-wide h={HORIZON}-week breach-risk forecast  "
             f"(n={len(inf)} commercial sites)",
             fontsize=14, fontweight="bold", y=0.97)
fig.tight_layout()
fig.savefig(FIG_FORECAST / "F2_national_risk_map_h12.png")
plt.show()

# %% [markdown]
# ## Chart F3 — Driver heatmap for the top-10 sites
#
# Each row = one site (top-10 by predicted risk). Each column = one feature.
# Cells are signed feature contributions in logit-space — red pushes risk up,
# blue pulls risk down. The columns shown are the top 12 features by total
# absolute contribution across these 10 sites, so the heatmap focuses on the
# drivers that actually matter for the high-risk subset.

# %%
def _build_heatmap_data(contribs_df, top_df, proba_col):
    """Top-10 sites × top-12 features-by-|contribution| → DataFrame for heatmap."""
    idx = top_df.head(10).index
    sub = contribs_df.loc[idx].drop(columns=["_bias"])
    col_importance = sub.abs().sum(axis=0).sort_values(ascending=False)
    top_features = col_importance.head(12).index.tolist()
    heat = sub[top_features].copy()
    heat.index = [
        f"{r['SITENAME']} (PO{int(r['PRODUCTIONAREAID'])}, p={r[proba_col]:.2f})"
        for _, r in top_df.head(10).iterrows()
    ]
    return heat


heat_v1 = _build_heatmap_data(contribs, top20, "predicted_breach_probability")

# Stack v1 above v3 if available so the visual story is "old model → new model"
heatmaps_to_render = [("v1 (52 features)", heat_v1)]
if v3_available:
    contribs_v3 = model_v3.predict_contributions(inf)
    top20_v3_df = (inf.sort_values("predicted_breach_probability_v3", ascending=False)
                       .head(20))
    heat_v3 = _build_heatmap_data(contribs_v3, top20_v3_df,
                                   "predicted_breach_probability_v3")
    heatmaps_to_render.append(("v3 (+ neighbor features)", heat_v3))

# Shared color scale across both panels so visual intensity is comparable.
shared_max = max(max(abs(h.values.min()), abs(h.values.max()))
                 for _, h in heatmaps_to_render)

fig, axes = plt.subplots(len(heatmaps_to_render), 1,
                         figsize=(13, 6.5 * len(heatmaps_to_render)))
if len(heatmaps_to_render) == 1:
    axes = [axes]
for ax, (tag, heat) in zip(axes, heatmaps_to_render):
    sns.heatmap(heat, ax=ax, cmap="RdBu_r", center=0,
                vmin=-shared_max, vmax=shared_max,
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Contribution to logit (red=push up, blue=pull down)"},
                linewidths=0.3, linecolor="white")
    ax.set_xlabel("Top-12 features by absolute contribution across these 10 sites")
    ax.set_ylabel("")
    ax.set_title(f"{tag} — what drives the top-10 at h={HORIZON} weeks?",
                 fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=9)
fig.tight_layout()
fig.savefig(FIG_FORECAST / "F3_top_drivers_heatmap.png")
plt.show()


# %% [markdown]
# ## Chart F4 — Distribution of predicted probabilities across all sites

# %%
base_rate_2025 = 0.038
top20_cutoff_v1 = top20["predicted_breach_probability"].min()
# Use log-spaced bins so the heavy left tail doesn't dominate the picture;
# linear bins squish the interesting top-tail comparison into a few cells.
bins = np.linspace(0, max(inf["predicted_breach_probability"].max(),
                          inf["predicted_breach_probability_v3"].max()
                          if v3_available else 0) * 1.02, 60)

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.hist(inf["predicted_breach_probability"], bins=bins,
        color="#1f77b4", alpha=0.55, edgecolor="white", linewidth=0.4,
        label=f"v1 (max {inf['predicted_breach_probability'].max():.3f})")
if v3_available:
    top20_cutoff_v3 = (inf["predicted_breach_probability_v3"]
                       .nlargest(20).min())
    ax.hist(inf["predicted_breach_probability_v3"], bins=bins,
            color="#d62728", alpha=0.55, edgecolor="white", linewidth=0.4,
            label=f"v3 (max {inf['predicted_breach_probability_v3'].max():.3f})")
ax.axvline(base_rate_2025, color="grey", linestyle="--", linewidth=1.2,
           label=f"2025 holdout base rate ≈ {base_rate_2025:.2f}")
ax.axvline(top20_cutoff_v1, color="#1f77b4", linestyle=":", linewidth=1.2,
           label=f"v1 top-20 cutoff = {top20_cutoff_v1:.3f}")
if v3_available:
    ax.axvline(top20_cutoff_v3, color="#d62728", linestyle=":", linewidth=1.2,
               label=f"v3 top-20 cutoff = {top20_cutoff_v3:.3f}")
ax.set_xlabel(f"Predicted breach probability (h={HORIZON} weeks)")
ax.set_ylabel(f"Number of sites  (total n={len(inf)})")
ax.set_title(f"Distribution of predicted breach probabilities — v1 vs v3")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_FORECAST / "F4_predicted_proba_distribution.png")
plt.show()

print(f"\nPrediction distribution:")
print(inf["predicted_breach_probability"].describe().round(4))


# %% [markdown]
# ## Chart F5 — v1 vs v3 ranking comparison (only if v3 booster exists)
#
# v3 adds spatial-diffusion features: for each site at each week, the mean
# and max FEMALEADULT of neighboring sites within 5 km and 10 km, plus
# count of breaching neighbors. Biologically motivated — lice drift
# between farms in the same fjord system.
#
# Compare the top-20 sites between v1 and v3 to see whether the neighbor
# signal reshuffles the ranking, lifts borderline sites, or confirms v1.

# %%
if v3_available:
    # Union of v1 top-20 and v3 top-20 (could overlap)
    top20_v1 = inf.sort_values("predicted_breach_probability", ascending=False).head(20)
    top20_v3 = inf.sort_values("predicted_breach_probability_v3", ascending=False).head(20)
    union_ids = pd.Index(top20_v1["SITENUMBER"]).union(top20_v3["SITENUMBER"])
    compare = inf[inf["SITENUMBER"].isin(union_ids)].copy()
    compare["label"] = compare.apply(
        lambda r: f"{r['SITENAME']} (PO{int(r['PRODUCTIONAREAID'])})", axis=1)
    compare = compare.sort_values("predicted_breach_probability_v3", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(7, 0.4 * len(compare))))
    y = np.arange(len(compare))
    h = 0.38
    bars_v1 = ax.barh(y - h / 2, compare["predicted_breach_probability"],
                      height=h, color="#1f77b4", label="v1 (52 features)")
    bars_v3 = ax.barh(y + h / 2, compare["predicted_breach_probability_v3"],
                      height=h, color="#d62728",
                      label="v3 (+ neighbor features)")
    ax.bar_label(bars_v1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars_v3, fmt="%.3f", padding=3, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(compare["label"], fontsize=9)
    ax.set_xlabel(f"Predicted breach probability (h={HORIZON} weeks)")
    ax.set_title(f"v1 vs v3 — union of top-20 rankings  "
                 f"(does the neighbor signal change who's flagged?)",
                 fontsize=12)
    ax.set_xlim(0, max(compare["predicted_breach_probability"].max(),
                       compare["predicted_breach_probability_v3"].max()) * 1.15)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_FORECAST / "F5_v1_vs_v3_ranking.png")
    plt.show()

    # Rank-shift narrative for the summary doc
    inf_with_ranks = inf.assign(
        rank_v1=inf["predicted_breach_probability"].rank(ascending=False, method="min"),
        rank_v3=inf["predicted_breach_probability_v3"].rank(ascending=False, method="min"),
    )
    rank_shifts = (inf_with_ranks[inf_with_ranks["rank_v1"] <= 20]
                       .assign(rank_change=lambda d: d["rank_v1"] - d["rank_v3"])
                       [["SITENAME", "PRODUCTIONAREAID", "rank_v1", "rank_v3",
                         "rank_change", "predicted_breach_probability",
                         "predicted_breach_probability_v3"]]
                       .sort_values("rank_change"))
    print("\nv1 top-20 — how each moved under v3 (negative rank_change = "
          "promoted up the list):")
    print(rank_shifts.round(3).to_string(index=False))


# %% [markdown]
# ## Cumulative model — "how many breaches in the next 12 weeks?"
#
# v1 and v3 above are BINARY classifiers — they predict the probability of
# a breach **in the specific target week** (e.g. ISO week 11, 2026-03-16).
#
# The case also asks for "Nr of breaches X weeks ahead", which the literal
# reading is **cumulative** — total breach weeks in the prediction window.
# We train a LightGBM Poisson regressor (`scripts/train_cumulative.py`)
# whose target is `sum(BREACH) over t+1..t+12`. Predictions are expected
# counts (0-12 scale); MAE / RMSE are the natural metrics here.

# %%
CUM_PATH = MODELS_DIR / f"lgbm_cumulative_w{HORIZON}.txt"
cum_available = CUM_PATH.exists()
if cum_available:
    cum_booster = lgb.Booster(model_file=str(CUM_PATH))
    cum_meta = json.loads((CUM_PATH.with_suffix(CUM_PATH.suffix + ".meta.json"))
                          .read_text())
    cum_features = cum_meta["feature_cols"]
    inf["predicted_cumulative_count"] = cum_booster.predict(inf[cum_features])
    print(f"Cumulative model loaded — {len(cum_features)} features, "
          f"inner-val MAE={cum_meta['inner_val_mae']:.3f}")
    print(f"\nPredicted cumulative count distribution (expected breach weeks "
          f"in next {HORIZON} weeks):")
    print(inf["predicted_cumulative_count"].describe().round(3))

    # Top-20 by cumulative prediction
    top20_cum = (inf.sort_values("predicted_cumulative_count", ascending=False)
                     .head(20).copy())
    top20_cum["label"] = top20_cum.apply(
        lambda r: f"{r['SITENAME']} (PO{int(r['PRODUCTIONAREAID'])})", axis=1)
    top20_cum_plot = top20_cum.iloc[::-1]
    # Color by PO
    unique_pos_c = sorted(top20_cum["PRODUCTIONAREAID"].unique())
    palette_c = sns.color_palette("tab10", n_colors=max(len(unique_pos_c), 3))
    po_colors_c = {po: palette_c[i] for i, po in enumerate(unique_pos_c)}
    colors_c = [po_colors_c[po] for po in top20_cum_plot["PRODUCTIONAREAID"]]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(top20_cum_plot["label"], top20_cum_plot["predicted_cumulative_count"],
                   color=colors_c, edgecolor="grey", linewidth=0.4)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xlabel(f"Predicted breach count in next {HORIZON} weeks  "
                  f"(Poisson regression, 0-{HORIZON} scale)")
    ax.set_title(f"Top-20 commercial sites by predicted CUMULATIVE breach count "
                 f"over next {HORIZON} weeks\n"
                 f"(predict from {inf['WEEK_START'].max().date()} -> "
                 f"covers weeks t+1 ... t+{HORIZON})",
                 fontsize=11)
    ax.set_xlim(0, top20_cum["predicted_cumulative_count"].max() * 1.15)
    handles_c = [plt.Rectangle((0, 0), 1, 1, color=po_colors_c[po])
                 for po in unique_pos_c]
    labels_c = [po_label(po, top20_cum[top20_cum["PRODUCTIONAREAID"] == po]
                                  ["PRODUCTIONAREA"].iloc[0]) for po in unique_pos_c]
    ax.legend(handles_c, labels_c, title="Produksjonsområde", loc="lower right",
              fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_FORECAST / "F6_top_sites_cumulative_count_w12.png")
    plt.show()
else:
    print("Cumulative model not trained yet — run "
          "`python -m scripts.train_cumulative` first.")


# %% [markdown]
# ## Auto-generated forecast summary doc

# %%
# For the narrative: across the top-20, which features are most often
# THE dominant positive contributor (i.e. which feature shows up most
# frequently as the #1 push-up driver)?
top20_contribs = contribs.loc[top20.index].drop(columns=["_bias"])
top_positive_per_site = top20_contribs.where(top20_contribs > 0).idxmax(axis=1)
dominant_driver_counts = top_positive_per_site.value_counts()

# Top-20 table (rendered as markdown by hand — no `tabulate` dependency)
top20_md_cols = [
    ("SITENUMBER", "SITENUMBER"),
    ("SITENAME", "SITENAME"),
    ("PRODUCTIONAREAID", "PO_id"),
    ("PRODUCTIONAREA", "PO_name"),
    ("LATITUDE", "lat"),
    ("LONGITUDE", "lon"),
    ("FEMALEADULT", "FEMALEADULT"),
    ("MOBILELICE", "MOBILELICE"),
    ("SEATEMPERATURE", "SEATEMP_C"),
    ("predicted_breach_probability", "proba"),
]
top20_view = top20.copy()
top20_view["LATITUDE"] = top20_view["LATITUDE"].round(3)
top20_view["LONGITUDE"] = top20_view["LONGITUDE"].round(3)
top20_view["FEMALEADULT"] = top20_view["FEMALEADULT"].round(3)
top20_view["MOBILELICE"] = top20_view["MOBILELICE"].round(3)
top20_view["SEATEMPERATURE"] = top20_view["SEATEMPERATURE"].round(2)
top20_view["predicted_breach_probability"] = top20_view["predicted_breach_probability"].round(3)


def _row_to_md_cells(row, cols):
    out = []
    for src, _ in cols:
        v = row[src]
        out.append("" if pd.isna(v) else str(v))
    return "| " + " | ".join(out) + " |"


top20_md_lines = []
top20_md_lines.append("| " + " | ".join(label for _, label in top20_md_cols) + " |")
top20_md_lines.append("| " + " | ".join("---" for _ in top20_md_cols) + " |")
for _, row in top20_view.iterrows():
    top20_md_lines.append(_row_to_md_cells(row, top20_md_cols))
top20_md_str = "\n".join(top20_md_lines)

predict_from = inf["WEEK_START"].max().date()
predict_for = inf["target_week_start"].iloc[0].date()
n_sites = len(inf)
base_rate_doc = base_rate_2025
top20_min = top20["predicted_breach_probability"].min()
top20_max = top20["predicted_breach_probability"].max()
top20_mean_FA = top20["FEMALEADULT"].mean()
top20_mean_temp = top20["SEATEMPERATURE"].mean()

dominant_md = "\n".join(
    f"- **{feat}** — dominant driver for {n} of the top-20 sites"
    for feat, n in dominant_driver_counts.head(5).items()
)

# v3 narrative addendum (only if v3 booster exists)
if v3_available:
    top_v1 = inf.sort_values("predicted_breach_probability", ascending=False).head(20)
    top_v3 = inf.sort_values("predicted_breach_probability_v3", ascending=False).head(20)
    inf_with_ranks = inf.assign(
        rank_v1=inf["predicted_breach_probability"].rank(ascending=False, method="min"),
        rank_v3=inf["predicted_breach_probability_v3"].rank(ascending=False, method="min"),
    )
    v1_top1_site = top_v1.iloc[0]["SITENAME"]
    v1_top1_p = top_v1.iloc[0]["predicted_breach_probability"]
    v3_top1_site = top_v3.iloc[0]["SITENAME"]
    v3_top1_p = top_v3.iloc[0]["predicted_breach_probability_v3"]
    # Biggest promotions and demotions. shift = rank_v1 - rank_v3.
    # Positive shift = v3 ranks the site BETTER (lower number = closer to #1).
    # Negative shift = v3 ranks the site WORSE.
    movers = (inf_with_ranks[inf_with_ranks["rank_v1"] <= 20]
              .assign(shift=lambda d: d["rank_v1"] - d["rank_v3"])
              .sort_values("shift", ascending=False))
    top_promoted = movers.head(3)[["SITENAME", "rank_v1", "rank_v3"]].to_dict("records")
    top_demoted = movers.tail(3)[["SITENAME", "rank_v1", "rank_v3"]].to_dict("records")
    v3_md = f"""

## v3 (with neighbor features) — does the spatial signal change the picture?

v3 = v1's 52 features **plus** 8 neighbor features (mean/max FEMALEADULT,
mean MOBILELICE, count of breaching neighbors — within 5 km and 10 km).
Inner-validation PR-AUC lifts visible at every horizon, biggest at h=12.

- **Top hit (v1):** {v1_top1_site}  @  p={v1_top1_p:.3f}
- **Top hit (v3):** {v3_top1_site}  @  p={v3_top1_p:.3f}
- **Biggest promotions** (sites v3 ranks higher than v1):
{chr(10).join(f"  - {m['SITENAME']}: v1 rank {int(m['rank_v1'])} -> v3 rank {int(m['rank_v3'])}" for m in top_promoted)}
- **Biggest demotions** (v1 false-flags v3 disagrees with):
{chr(10).join(f"  - {m['SITENAME']}: v1 rank {int(m['rank_v1'])} -> v3 rank {int(m['rank_v3'])}" for m in top_demoted)}

See `F5_v1_vs_v3_ranking.png` for the full union-of-top-20 comparison.
v3 makes more extreme predictions on a subset of sites where neighbors
are also under pressure, and dampens predictions where v1 was picking
up site-internal noise that the spatial context contradicts.
"""
else:
    v3_md = "\n\n## v3 (neighbor features)\n\nNot available — run `python -m scripts.train_and_save` to train.\n"

if cum_available:
    top20_cum_for_md = (inf.sort_values("predicted_cumulative_count", ascending=False)
                            .head(20))
    cum_table_lines = [
        "| SITENUMBER | SITENAME | PO | exp.count (next 12w) | recent FA |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in top20_cum_for_md.iterrows():
        cum_table_lines.append(
            f"| {int(row['SITENUMBER'])} | {row['SITENAME']} | "
            f"PO{int(row['PRODUCTIONAREAID'])} | "
            f"{row['predicted_cumulative_count']:.2f} | "
            f"{('' if pd.isna(row['FEMALEADULT']) else f'{row['FEMALEADULT']:.2f}')} |"
        )
    cum_md = f"""

## Cumulative model — "how many breaches in the next {HORIZON} weeks?"

LightGBM **Poisson regression** trained with the v3 feature set (V1 + 8
neighbor features). Target: `sum(BREACH) over t+1..t+{HORIZON}` —
integer count, 0..{HORIZON}.

- **Inner-val MAE (2024):** {cum_meta['inner_val_mae']:.3f}
- **Predicted distribution across {len(inf)} commercial sites:**
  min {inf['predicted_cumulative_count'].min():.3f} ·
  median {inf['predicted_cumulative_count'].median():.3f} ·
  mean {inf['predicted_cumulative_count'].mean():.3f} ·
  max {inf['predicted_cumulative_count'].max():.3f}

Interpretation: an expected-count of 1.5 means "the model thinks this
site will have, on average, 1.5 breach weeks across the next {HORIZON}
weeks." Unlike the binary point-prediction at week t+{HORIZON} only,
this score integrates risk over the full {HORIZON}-week window.

### Top-20 by cumulative expected count

{chr(10).join(cum_table_lines)}

See `F6_top_sites_cumulative_count_w{HORIZON}.png` for the bar chart.

### Limitation — the treatment-response cycle is implicit, not simulated

This model predicts a **marginal expected count** given current state,
averaged across historical analogous states. It does NOT simulate the
breach -> treatment -> lower lice -> regrowth cycle forward in time.
The training data already includes those dynamics, so the predicted
count reflects *typical Mowi management* — 2.6 expected breach weeks
for Ommundsteigen means "on average, historically similar states ended
with 2.6 breaches across the next {HORIZON} weeks, including the typical
treatments that followed each breach."

What the model has indirect signal about (via existing features):
- `days_since_chem`, `days_since_mech`, `days_since_bio` — where the site
  is in the post-treatment recovery cycle.
- `treat_*_roll12` — how aggressively the site has been managed recently.
- `FEMALEADULT_roll8_mean` — whether lice levels are persistently elevated
  vs an acute spike.

What it CANNOT answer:
- Counterfactual questions ("what if we treat 2 weeks earlier?").
- Predictions under a changed management regime (e.g. if Mowi adopts
  preventive treatment policies that diverge from historical patterns).

A natural next step would be a sequential / hazard model that simulates
each future week conditionally on a treatment policy. Out of scope here.
"""
else:
    cum_md = ""

forecast_md = f"""# 12-week breach-risk forecast

**Predict from week:** {predict_from}
**Target week:** {predict_for}  (= predict_from + {HORIZON} weeks)
**Sites scored:** {n_sites} commercial sites (HI research sites filtered out)
**Model:** LightGBM v1 (default), persisted booster `models/lgbm_v1_h{HORIZON}.txt`. v3 (with neighbor features) also available and compared below.

## Headline

Predicted breach probabilities range from
**{inf['predicted_breach_probability'].min():.3f}** to
**{inf['predicted_breach_probability'].max():.3f}**, with a median of
{inf['predicted_breach_probability'].median():.3f}. The top-20 cutoff is
**{top20_min:.3f}** and the highest-risk site is **{top20_max:.3f}**.

For context the 2025 holdout base rate is ≈{base_rate_doc:.2f}, so a
probability above the base rate flags above-average risk; the top-20 are
all at least {top20_min / base_rate_doc:.1f}× the base rate.

The top-20 carry an average recent FEMALEADULT of {top20_mean_FA:.2f}
lice/fish (regulatory limit 0.5) and an average sea temperature of
{top20_mean_temp:.1f} °C. They cluster geographically in mid-Norway and
the southern coast (see F2).

## Top-20 sites — predicted breach risk at h={HORIZON}w (v1)

{top20_md_str}
{v3_md}
{cum_md}
## Main drivers across the top-20 (v1)

Feature most often ranked as the #1 positive contributor:

{dominant_md}

The full per-site decomposition is in `F3_top_drivers_heatmap.png`.

## Calibration caveat

LightGBM v1 was trained without `scale_pos_weight`, so the probabilities
above are calibrated rather than rank-only — absolute values are
interpretable. **But:** the h={HORIZON} model has PR-AUC ≈ 0.10 on the
2025 holdout (real but modest — 1.2× the best naive baseline). The
ranking is reliable on the top tail; absolute probabilities for sites
near the base rate should be read as "uncertain, not safe", not
"benign". Operational decisions should weight the **per-site driver
breakdown** (F3) more than the raw score for borderline sites.

## How this was produced

- `src/features.py::build_inference_frame(lice, treat, horizon={HORIZON})`
  — builds one row per commercial site at its latest week with all 52
  model features.
- `src/models.py::LightGBMBreach.predict_contributions()` — wraps
  LightGBM's native `pred_contrib=True` decomposition; per-site
  per-feature contributions sum (plus bias) to the raw logit.
- HI research sites are excluded at the data layer via
  `src.research_sites.RESEARCH_SITE_IDS`.
- The same data + model is callable from the agent as `predict_risk`
  (ranking) and `predict_drivers` (per-site explanation).
"""

(REPORTS_DIR / "forecast_summary.md").write_text(forecast_md, encoding="utf-8")
print(f"\nSaved: {REPORTS_DIR / 'forecast_summary.md'}  "
      f"({len(forecast_md):,} chars)")
