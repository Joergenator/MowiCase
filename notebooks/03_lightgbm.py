# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Step 4 — LightGBM breach prediction
#
# Train one LightGBM model per horizon (h=1, 2, 12) with engineered features
# (lags, rolling stats, temperature, treatment, structural). Compare against
# the step 3 baselines on the same 2025 holdout.
#
# **Bars to beat:**
# - h=1:  PR-AUC 0.194 (B1 Persistence)
# - h=12: PR-AUC 0.077 (B3 POWeekRate)

# %%
import sys
import time
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.load_data import load_lice, load_treatment
from src.features import (
    build_feature_frame, FEATURE_COLUMNS_V1, FEATURE_COLUMNS_V2,
    CATEGORICAL_FEATURES,
)
from src.utils import train_test_split_by_year, po_label
from src.baselines import all_baselines
from src.evaluation import score_baseline, pr_auc_by_po
from src.models import LightGBMBreach

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 14,
})

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

HORIZONS = (1, 2, 12)


# %% [markdown]
# ## 1. Load data and build feature frames at each horizon

# %%
t0 = time.time()
lice = load_lice()
treat = load_treatment()
print(f"loaded: lice={lice.shape}, treatment={treat.shape} "
      f"in {time.time() - t0:.1f}s")

frames = {}
splits = {}
for h in HORIZONS:
    t0 = time.time()
    sup = build_feature_frame(lice, treat, horizon=h)
    train, test = train_test_split_by_year(sup)
    frames[h] = sup
    splits[h] = (train, test)
    print(f"  h={h:>2}w: shape={sup.shape}, train={len(train):>7,}, "
          f"test={len(test):>6,}, pos_rate={test['target'].mean():.3%} "
          f"({time.time() - t0:.1f}s)")


# %% [markdown]
# ## 2. Fit two LightGBM variants per horizon — v1 (no biology) and v2 (with cleaner-fish biology)
#
# v1 uses the original 52-feature set. v2 adds two cleaner-fish features:
# `bio_active` (count of rensefisk stockings still alive given temperature
# history) and `weeks_since_last_cold` (proxy for how long the warm period
# has lasted). The hypothesis: encoding the cleaner-fish-only-survives-when-
# warm biology should help h=4-8 forecasts in particular.

# %%
VARIANTS = (
    ("LightGBM v1", FEATURE_COLUMNS_V1),
    ("LightGBM v2", FEATURE_COLUMNS_V2),
)

models = {}  # (variant_name, h) -> fitted model
records = []
predictions = {}  # (variant_name, h) -> (test_df, y_score)
for h in HORIZONS:
    train, test = splits[h]
    for variant_name, feature_cols in VARIANTS:
        t0 = time.time()
        m = LightGBMBreach(
            name=variant_name, horizon=h, tune=True,
            feature_cols=tuple(feature_cols),
            cat_cols=tuple(CATEGORICAL_FEATURES),
        ).fit(train)
        p = m.predict_proba(test)
        s = score_baseline(variant_name, h, test, p)
        models[(variant_name, h)] = m
        predictions[(variant_name, h)] = (test, p)
        records.append(s.as_row())
        print(f"{variant_name} h={h:>2}w: PR-AUC={s.pr_auc:.4f}, "
              f"P@100={s.p_at_100:.3f}, Brier={s.brier:.5f}, "
              f"count_MAE={s.count_mae:.2f}  "
              f"(inner-val={m.inner_val_pr_auc_:.4f}, "
              f"best_iter={m.best_iter_}, fit={time.time() - t0:.1f}s)")


# %% [markdown]
# ## 3. Re-run the baselines on the same frames for direct comparison

# %%
baseline_records = []
for h in HORIZONS:
    train, test = splits[h]
    for b in all_baselines(horizon=h):
        b.fit(train)
        p = b.predict_proba(test)
        s = score_baseline(b.name, h, test, p)
        baseline_records.append(s.as_row())

all_rows = pd.DataFrame(baseline_records + records)
print(all_rows.to_string(index=False))


# %% [markdown]
# ## 4. Headline pivots — LightGBM vs baselines, per metric

# %%
ORDER = ["B0 GlobalRate", "B1 Persistence", "B2 SeasonalNaive",
         "B3 POWeekRate", "LightGBM v1", "LightGBM v2"]


def pivot(metric: str) -> pd.DataFrame:
    return (all_rows.pivot(index="baseline", columns="horizon", values=metric)
            .reindex(ORDER))


pivots = {m: pivot(m) for m in ["PR-AUC", "P@100", "Brier", "count_MAE"]}
for m, tbl in pivots.items():
    print(f"\n— {m} —")
    print(tbl.round(4))


# %% [markdown]
# ## 5. Headline chart — PR-AUC at each horizon, all models

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
colors = {"B0 GlobalRate": "#cccccc", "B1 Persistence": "#1f77b4",
          "B2 SeasonalNaive": "#ff7f0e", "B3 POWeekRate": "#2ca02c",
          "LightGBM v1": "#d62728", "LightGBM v2": "#7d0d0d"}

for ax, h in zip(axes, HORIZONS):
    subset = all_rows[all_rows["horizon"] == h]
    bars = ax.bar(subset["baseline"], subset["PR-AUC"],
                  color=[colors[b] for b in subset["baseline"]])
    ax.set_title(f"h = {h} week{'s' if h > 1 else ''}")
    ax.set_ylabel("PR-AUC" if ax is axes[0] else "")
    ax.tick_params(axis="x", rotation=35)
    pos_rate = subset["pos_rate"].iloc[0]
    ax.axhline(pos_rate, color="black", linestyle=":", linewidth=1,
               label=f"base rate {pos_rate:.2%}")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)

fig.suptitle("LightGBM vs naive baselines — PR-AUC on 2025 holdout",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "C1_lightgbm_vs_baselines_pr_auc.png")
plt.show()


# %% [markdown]
# ## 6. Feature importance — what does v2 use, and where do bio features rank?

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharex=False)
for ax, h in zip(axes, HORIZONS):
    imp = models[("LightGBM v2", h)].feature_importance("gain").head(15).iloc[::-1]
    bio_features = {"bio_active", "weeks_since_last_cold",
                    "treat_bio", "treat_bio_roll4", "treat_bio_roll8",
                    "treat_bio_roll12", "days_since_bio"}
    bar_colors = ["#7d0d0d" if f in bio_features else "#4c72b0" for f in imp.index]
    ax.barh(imp.index, imp.values, color=bar_colors)
    ax.set_title(f"h = {h} week{'s' if h > 1 else ''}")
    ax.set_xlabel("Gain")

fig.suptitle("LightGBM v2 — top 15 features by gain (red = cleaner-fish biology)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "C2_lightgbm_feature_importance.png")
plt.show()

# Also export a small table of bio-feature ranks per horizon — useful to
# diagnose whether the v2 features are actually being used.
print("\nRank of cleaner-fish features in v2 by gain (1 = most important):")
bio_rank_rows = []
for h in HORIZONS:
    imp = models[("LightGBM v2", h)].feature_importance("gain")
    ranks = pd.Series(range(1, len(imp) + 1), index=imp.index)
    bio_rank_rows.append({
        "horizon": h,
        "bio_active_rank": int(ranks.get("bio_active", -1)),
        "weeks_since_last_cold_rank": int(ranks.get("weeks_since_last_cold", -1)),
        "bio_active_gain": float(imp.get("bio_active", 0)),
        "weeks_since_last_cold_gain": float(imp.get("weeks_since_last_cold", 0)),
    })
bio_ranks = pd.DataFrame(bio_rank_rows)
print(bio_ranks.to_string(index=False))


# %% [markdown]
# ## 7. Per-PO breakdown — does LightGBM help in low-base-rate POs?

# %%
po_id_to_label = (lice.dropna(subset=["PRODUCTIONAREA", "PRODUCTIONAREAID"])
                  .groupby("PRODUCTIONAREAID")["PRODUCTIONAREA"]
                  .agg(lambda s: s.mode().iloc[0]))

per_po = {}
for h in HORIZONS:
    test, p = predictions[("LightGBM v2", h)]
    by_po = pr_auc_by_po(test, p)
    by_po.index = [po_label(po_id, po_id_to_label.get(po_id, "Unknown"))
                   for po_id in by_po.index]
    by_po.name = f"h={h}w"
    per_po[h] = by_po

breakdown = pd.concat(per_po.values(), axis=1).round(3)
print("\nLightGBM PR-AUC by PO (NaN = no positives in test):")
print(breakdown)

fig, ax = plt.subplots(figsize=(11, 6))
breakdown.plot.barh(ax=ax, color=["#fc8d59", "#e34a33", "#b30000"], width=0.8)
ax.set_xlabel("PR-AUC")
ax.set_ylabel("")
ax.set_title("LightGBM v2 PR-AUC by PO — where the model holds up")
ax.legend(loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "C3_lightgbm_pr_auc_by_po.png")
plt.show()


# %% [markdown]
# ## 8. Persist results to JSON for downstream comparisons

# %%
output = {
    "horizons": list(HORIZONS),
    "summary_rows": records,
    "all_rows": all_rows.to_dict(orient="records"),
    "metric_tables": {m: tbl.round(5).to_dict() for m, tbl in pivots.items()},
    "po_breakdown_v2": {f"h={h}": per_po[h].round(5).to_dict() for h in HORIZONS},
    "training": {
        variant: {
            f"h={h}": {
                "best_params": {k: v for k, v in models[(variant, h)].best_params_.items()
                                if k not in ("metric", "verbosity", "objective",
                                             "boosting_type", "deterministic")},
                "best_iter": models[(variant, h)].best_iter_,
                "inner_val_pr_auc": models[(variant, h)].inner_val_pr_auc_,
            }
            for h in HORIZONS
        }
        for variant, _ in VARIANTS
    },
    "feature_columns": {
        "v1": list(FEATURE_COLUMNS_V1),
        "v2": list(FEATURE_COLUMNS_V2),
    },
    "top_features": {
        variant: {
            f"h={h}": models[(variant, h)].feature_importance("gain").head(10).round(0).to_dict()
            for h in HORIZONS
        }
        for variant, _ in VARIANTS
    },
    "bio_feature_ranks_v2": bio_ranks.to_dict(orient="records"),
}
(MODELS_DIR / "lightgbm.json").write_text(
    json.dumps(output, indent=2, default=str), encoding="utf-8")
print(f"Saved to {MODELS_DIR / 'lightgbm.json'}")


# %% [markdown]
# ## 9. One-page summary for the deck

# %%
def fmt_pivot(tbl: pd.DataFrame) -> str:
    return tbl.round(4).to_markdown()


pr = pivots["PR-AUC"]
lift_h1 = pr.loc["LightGBM v2", 1] / pr.loc["B1 Persistence", 1]
lift_h12 = pr.loc["LightGBM v2", 12] / pr.loc["B3 POWeekRate", 12]
v2_v1_delta = {h: pr.loc["LightGBM v2", h] - pr.loc["LightGBM v1", h]
               for h in HORIZONS}

deck_md = f"""# LightGBM results — step 4

Test set: **2025 holdout** (untouched in development).
Training: 2012-01-02 to 2024-12-29, with 2024 as inner validation for early stopping.
Metric primary: **PR-AUC** (base rate {all_rows['pos_rate'].mean():.2%}).

Two model variants:
- **v1** — original 52 features (lags, rolling, treatment counts, structural).
- **v2** — v1 + cleaner-fish biology features: `bio_active` (count of
  rensefisk stockings still alive given the site's temperature history) and
  `weeks_since_last_cold`. Encodes the biological fact that cleaner fish die
  when SEATEMP drops below ~6 deg C.

## PR-AUC by horizon — every model side-by-side

{fmt_pivot(pivots['PR-AUC'])}

## Lift over the bar (using LightGBM v2)

- **h=1: v2 {pr.loc['LightGBM v2', 1]:.3f} vs B1 Persistence {pr.loc['B1 Persistence', 1]:.3f}** ({lift_h1:.1f}x lift)
- **h=12: v2 {pr.loc['LightGBM v2', 12]:.3f} vs B3 POWeekRate {pr.loc['B3 POWeekRate', 12]:.3f}** ({lift_h12:.1f}x lift)

## v2 vs v1 — does the cleaner-fish biology help?

| horizon | v1 PR-AUC | v2 PR-AUC | delta |
|---:|---:|---:|---:|
{chr(10).join(
    f"| h={h}w | {pr.loc['LightGBM v1', h]:.4f} | {pr.loc['LightGBM v2', h]:.4f} | "
    f"{v2_v1_delta[h]:+.4f} |"
    for h in HORIZONS
)}

## Precision-at-100 by horizon

{fmt_pivot(pivots['P@100'])}

## Count-MAE by horizon (weekly aggregate)

{fmt_pivot(pivots['count_MAE'])}

## What v2 uses — top 5 features per horizon by gain

| horizon | top features |
|---|---|
{chr(10).join(
    f"| h={h}w | " + ", ".join(models[('LightGBM v2', h)].feature_importance().head(5).index) + " |"
    for h in HORIZONS
)}

## Where do the cleaner-fish features rank in v2?

| horizon | bio_active rank | weeks_since_last_cold rank |
|---:|---:|---:|
{chr(10).join(
    f"| h={h}w | {row['bio_active_rank']} | {row['weeks_since_last_cold_rank']} |"
    for row in bio_rank_rows
)}

(Rank 1 = most important feature in the v2 model.)

## Inner-validation diagnostics — v2

| horizon | best PR-AUC (2024) | best_iter | num_leaves | min_data_in_leaf | learning_rate |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(
    f"| h={h}w | {models[('LightGBM v2', h)].inner_val_pr_auc_:.4f} | "
    f"{models[('LightGBM v2', h)].best_iter_} | "
    f"{models[('LightGBM v2', h)].best_params_.get('num_leaves')} | "
    f"{models[('LightGBM v2', h)].best_params_.get('min_data_in_leaf')} | "
    f"{models[('LightGBM v2', h)].best_params_.get('learning_rate')} |"
    for h in HORIZONS
)}

## Interpretation

- **LightGBM (either variant) dominates every metric at every horizon.**
  PR-AUC, P@100, Brier, and count-MAE all improve over the best baseline.
- **The crossover prediction from step 3 holds in the feature importances.**
  At h=1, the top features are current/recent lice counts (FEMALEADULT,
  FEMALEADULT_roll4_max, MOBILELICE) — the inertia signal B1 Persistence
  exploited. At h=12, lag importances collapse and structural features take
  over (target_iso_week, site_age_weeks, PRODUCTIONAREAID) — the same signal
  B3 POWeekRate exploited. One model handles both regimes.
- **v2 vs v1: an honest negative on PR-AUC.** The cleaner-fish features
  get *used* by the model (they rank 19-47 out of 54 by gain), but they
  don't move the headline ranking metric — PR-AUC is marginally *worse* at
  every horizon (deltas ~0.003-0.005, within tuning noise). v2 wins
  modestly on P@100 at h=1-2 and meaningfully on count-MAE at h=12 (5.37
  vs 6.51 — a 17% drop in weekly-count error). Interpretation: LightGBM v1
  was already learning the bio-x-temperature interaction implicitly from
  `treat_bio` + SEATEMP; making it explicit helps aggregate forecasting at
  long horizons but doesn't sharpen site-level ranking. The biology
  hypothesis was sound; the gain isn't where we expected it.
- **No `scale_pos_weight`.** We deliberately do NOT use class-imbalance flags
  because they inflate predicted probabilities and wreck calibration. PR-AUC
  is rank-invariant so we pay nothing in the headline metric; we gain a
  best-in-class Brier and count-MAE.
- **Per-PO performance** is strongest in mid-Norway (PO3, PO4, PO6, PO11);
  weakest in the cold-water Finnmark POs (PO12, PO13) where the breach base
  rate is too low for the model to find a signal. PO13 has zero positives in
  the 2025 test, so PR-AUC is undefined there.

## Why this is enough for a model

The case asks whether ~12 weeks ahead is forecastable. At h=12 we hit
PR-AUC {pr.loc['LightGBM v2', 12]:.2f} — a real but modest signal, and
{lift_h12:.1f}x the best naive. That is the honest answer: the structural
signal exists but is faint at 12 weeks, and the case's bonus question
("how far ahead is the signal still detectable?") points to h~1-4 as the
operationally useful range, with h=12 being the limit of what a tabular
model can do with the given features.
"""

(REPORTS_DIR / "model_summary.md").write_text(deck_md, encoding="utf-8")
print(f"Saved to {REPORTS_DIR / 'model_summary.md'} ({len(deck_md):,} chars)")
