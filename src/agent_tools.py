"""Tools the LLM agent can call.

Each function returns a JSON-serializable dict so the result can be fed back
into the Anthropic SDK tool_result block without further conversion.

Design notes
------------
- Named tools (trending_pos, repeat_offenders, treatment_intensity,
  pre_breach_signature, predict_risk) cover the 5 case demo questions with
  reliable, defensible logic the LLM never has to reinvent.
- `run_sql` + `describe_schema` form the escape hatch for arbitrary follow-up
  questions.
- All data flows through `src.load_data.load_training_data()` — the 2026
  leakage firewall therefore holds for the agent automatically.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent_db import describe_schema, run_sql
from src.features import FEATURE_COLUMNS_V1, build_inference_frame
from src.load_data import load_training_data
from src.models import LightGBMBreach
from src.research_sites import (
    RESEARCH_SITE_IDS, SOURCE_NOTE,
    list_research_sites as _list_research_sites,
)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Lazy data + model loading — first call pays the cost, later calls are free
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_training_data()


@lru_cache(maxsize=8)
def _load_model(horizon: int) -> LightGBMBreach:
    return LightGBMBreach.load(MODELS_DIR / f"lgbm_v1_h{horizon}.txt")


def _df_to_records(df: pd.DataFrame, limit: int = 50) -> list[dict]:
    """Convert a DataFrame to JSON-safe records, truncating to `limit` rows."""
    if len(df) > limit:
        df = df.head(limit)
    # Cast Timestamps to ISO strings so json.dumps doesn't choke.
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.replace({np.nan: None}).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Q1: trending_pos — which areas show increasing lice pressure
# ---------------------------------------------------------------------------

def trending_pos(weeks: int = 8) -> dict:
    """Slope of weekly mean FEMALEADULT (and breach rate) over the last `weeks`.

    Computed PER PO from the most recent `weeks` weeks of training data.
    Positive slope = rising lice pressure.
    """
    lice, _ = _data()
    end = lice["WEEK_START"].max()
    start = end - pd.Timedelta(weeks=weeks - 1)
    recent = lice[(lice["WEEK_START"] >= start) & lice["HAVECOUNTEDLICE"].fillna(False)].copy()

    grouped = (recent.groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA", "WEEK_START"])
                     .agg(mean_femaleadult=("FEMALEADULT", "mean"),
                          breach_rate=("BREACH", lambda s: s.dropna().mean()))
                     .reset_index())

    results = []
    for (po_id, po_name), g in grouped.groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA"]):
        if len(g) < 3:
            continue
        x = ((g["WEEK_START"] - g["WEEK_START"].min()).dt.days / 7).to_numpy()
        y_lice = g["mean_femaleadult"].to_numpy()
        y_br = g["breach_rate"].fillna(0).to_numpy()
        slope_lice = float(np.polyfit(x, y_lice, 1)[0]) if np.ptp(x) > 0 else 0.0
        slope_br = float(np.polyfit(x, y_br, 1)[0]) if np.ptp(x) > 0 else 0.0
        results.append({
            "production_area_id": int(po_id),
            "production_area": po_name,
            "weeks_observed": int(len(g)),
            "latest_mean_femaleadult": float(g["mean_femaleadult"].iloc[-1]),
            "latest_breach_rate": float(g["breach_rate"].iloc[-1] or 0),
            "slope_femaleadult_per_week": slope_lice,
            "slope_breach_rate_per_week": slope_br,
        })
    results.sort(key=lambda r: r["slope_femaleadult_per_week"], reverse=True)
    return {
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "weeks": weeks,
        "rows": results,
    }


# ---------------------------------------------------------------------------
# Q2: repeat_offenders — sites with the most breaches
# ---------------------------------------------------------------------------

def repeat_offenders(min_breaches: int = 5,
                     since_year: int = 2020,
                     top_n: int = 20) -> dict:
    """Sites with the most breach weeks since `since_year`.

    HI research sites are excluded at the data layer (see
    src.load_data.load_lice's `commercial_only` argument). Nothing to opt
    back in here — call `list_research_sites` if you want the registry.
    """
    lice, _ = _data()
    df = lice[(lice["YEAR"] >= since_year) & (lice["BREACH"] == True)].copy()  # noqa: E712
    agg = (df.groupby(["SITENUMBER", "SITENAME", "PRODUCTIONAREAID", "PRODUCTIONAREA"])
             .agg(breach_weeks=("BREACH", "size"),
                  first_breach=("WEEK_START", "min"),
                  last_breach=("WEEK_START", "max"))
             .reset_index())
    agg = agg[agg["breach_weeks"] >= min_breaches]
    agg = agg.sort_values("breach_weeks", ascending=False).head(top_n)
    agg["first_breach"] = agg["first_breach"].dt.strftime("%Y-%m-%d")
    agg["last_breach"] = agg["last_breach"].dt.strftime("%Y-%m-%d")
    return {
        "since_year": since_year,
        "min_breaches": min_breaches,
        "commercial_only": True,
        "research_sites_excluded": len(RESEARCH_SITE_IDS),
        "rows": agg.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Q3: treatment_intensity — treatments per active site by PO
# ---------------------------------------------------------------------------

def treatment_intensity(year: int | None = None,
                        exclude_cleanerfish: bool = True) -> dict:
    """Treatments per active site per year, grouped by PO.

    `exclude_cleanerfish=True` reproduces EDA chart 10 — active-intervention
    intensity excluding passive `rensefisk` stocking. Without that exclusion,
    POs that lean heavily on cleaner-fish appear artificially high.
    """
    lice, treat = _data()
    if year is not None:
        treat = treat[treat["YEAR"] == year]
        lice_y = lice[lice["YEAR"] == year]
    else:
        lice_y = lice

    if exclude_cleanerfish:
        treat = treat[treat["ACTION"] != "rensefisk"]

    # active site = at least one HAVECOUNTEDLICE=True week in scope
    active_per_po = (lice_y[lice_y["HAVECOUNTEDLICE"] == True]  # noqa: E712
                     .groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA"])["SITENUMBER"]
                     .nunique().rename("active_sites").reset_index())
    treat_per_po = (treat.groupby(["PRODUCTIONAREAID", "PRODUCTIONAREA"])
                         .size().rename("treatment_events").reset_index())

    merged = active_per_po.merge(treat_per_po, on=["PRODUCTIONAREAID", "PRODUCTIONAREA"], how="left")
    merged["treatment_events"] = merged["treatment_events"].fillna(0).astype(int)
    merged["treatments_per_active_site"] = (
        merged["treatment_events"] / merged["active_sites"].replace(0, np.nan)
    )
    merged = merged.sort_values("treatments_per_active_site", ascending=False, na_position="last")
    return {
        "year": year,
        "exclude_cleanerfish": exclude_cleanerfish,
        "rows": merged.round(2).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Q4: pre_breach_signature — what's different in the weeks before a breach
# ---------------------------------------------------------------------------

def pre_breach_signature(weeks_before: int = 4) -> dict:
    """Mean FEMALEADULT / MOBILELICE / SEATEMP in the `weeks_before` weeks
    preceding a breach, compared to the all-time baseline.

    The lead signal is the same one the LightGBM model exploits at h=1: lice
    counts tend to rise for several weeks before crossing the regulatory limit.
    """
    lice, _ = _data()
    df = lice.sort_values(["SITENUMBER", "WEEK_START"]).copy()
    breach_mask = (df["BREACH"] == True).fillna(False)  # noqa: E712
    grouped = df.groupby("SITENUMBER", sort=False)

    # For each row, was there a breach in the next `weeks_before` rows (same site)?
    # BREACH is a nullable bool; cast each shifted Series to plain bool via fillna.
    #
    # Cross-site safety: groupby.shift(-k) shifts WITHIN each site only — when a
    # site's last row has no row k positions ahead within its own group, the
    # shifted value is NaN (becomes False after fillna). So site B's first
    # breach never leaks back as a "future breach" for site A's last weeks.
    # .to_numpy() preserves positional alignment because df was sorted by
    # (SITENUMBER, WEEK_START) before the groupby.
    is_pre_breach = np.zeros(len(df), dtype=bool)
    for k in range(1, weeks_before + 1):
        is_pre_breach |= grouped["BREACH"].shift(-k).fillna(False).to_numpy(dtype=bool)

    df["pre_breach"] = is_pre_breach
    # Only consider rows that are themselves *not* a breach (we want the run-up)
    pre = df[df["pre_breach"] & ~breach_mask & df["HAVECOUNTEDLICE"].fillna(False)]
    baseline = df[~breach_mask & df["HAVECOUNTEDLICE"].fillna(False)]

    def _means(frame: pd.DataFrame) -> dict:
        return {
            "n_rows": int(len(frame)),
            "femaleadult": float(frame["FEMALEADULT"].mean()),
            "mobilelice": float(frame["MOBILELICE"].mean()),
            "seatemp": float(frame["SEATEMPERATURE"].mean()),
        }

    return {
        "weeks_before": weeks_before,
        "pre_breach": _means(pre),
        "baseline_non_breach": _means(baseline),
    }


# ---------------------------------------------------------------------------
# Q5: predict_risk — LightGBM forward prediction
# ---------------------------------------------------------------------------

VALID_HORIZONS = (1, 2, 12)


def predict_risk(horizon: int = 12, top_n: int = 20) -> dict:
    """Score the latest available week per site with the LightGBM v1 model.

    `horizon` ∈ {1, 2, 12} — picks the matching booster.
    Returns the top `top_n` sites by predicted breach probability, along with
    the contextual fields a human reviewer wants to see (PO, lat/lon, recent
    FEMALEADULT, SEATEMP).

    HI research sites are excluded at the data layer — the booster itself
    was trained on commercial-only data via `load_training_data()`.
    """
    if horizon not in VALID_HORIZONS:
        raise ValueError(f"horizon must be one of {VALID_HORIZONS}, got {horizon}")

    lice, treat = _data()
    inf = build_inference_frame(lice, treat, horizon=horizon)
    model = _load_model(horizon)
    proba = model.predict_proba(inf)
    inf = inf.assign(predicted_breach_probability=proba)

    display_cols = [
        "SITENUMBER", "SITENAME", "PRODUCTIONAREAID", "PRODUCTIONAREA",
        "LATITUDE", "LONGITUDE", "WEEK_START", "target_week_start",
        "FEMALEADULT", "MOBILELICE", "SEATEMPERATURE",
        "predicted_breach_probability",
    ]
    out = (inf[display_cols]
              .sort_values("predicted_breach_probability", ascending=False)
              .head(top_n)
              .round({"predicted_breach_probability": 4,
                      "FEMALEADULT": 3, "MOBILELICE": 3, "SEATEMPERATURE": 2,
                      "LATITUDE": 4, "LONGITUDE": 4}))
    return {
        "horizon_weeks": horizon,
        "model_version": "lightgbm_v1",
        "predict_from_week": str(inf["WEEK_START"].max().date()),
        "predict_for_week": str(out["target_week_start"].iloc[0].date()),
        "n_sites_scored": int(len(inf)),
        "commercial_only": True,
        "research_sites_excluded": len(RESEARCH_SITE_IDS),
        "top_sites": _df_to_records(out, limit=top_n),
    }


# ---------------------------------------------------------------------------
# Escape hatch + schema helper
# ---------------------------------------------------------------------------

def sql(query: str, max_rows: int = 200) -> dict:
    """Run a read-only SQL query over vlice / vtreatment."""
    df = run_sql(query, max_rows=max_rows)
    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "rows": _df_to_records(df, limit=max_rows),
        "truncated": len(df) == max_rows,
    }


def schema() -> dict:
    """Return the column dictionary for vlice + vtreatment."""
    return describe_schema()


# ---------------------------------------------------------------------------
# EDA narrative — analyst briefing on what we already learned from the data
# ---------------------------------------------------------------------------

FINDINGS_PATH = MODELS_DIR.parent / "reports" / "findings.md"


def list_research_sites() -> dict:
    """Return the registry of Havforskningsinstituttet (HI) research sites.

    These sites are excluded by default from `predict_risk` and
    `repeat_offenders` because their lice patterns reflect study protocols
    rather than commercial operation.
    """
    return {
        "source": SOURCE_NOTE,
        "n_sites": len(RESEARCH_SITE_IDS),
        "sites": _list_research_sites(),
    }


def read_findings() -> dict:
    """Return the full EDA findings document.

    The agent is given a 3-sentence summary in its system prompt; this tool
    surfaces the full narrative (numbered findings on treatment intensity,
    temperature-lice relationship, seasonality, geography, etc.) when the
    agent decides it needs the detail. The findings were derived from
    pre-2026 training data only — leakage discipline holds.
    """
    return {
        "source": str(FINDINGS_PATH.relative_to(MODELS_DIR.parent)),
        "content": FINDINGS_PATH.read_text(encoding="utf-8"),
    }


# ---------------------------------------------------------------------------
# Tool registry — used by src.agent to build the Anthropic tool_use list
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "trending_pos": trending_pos,
    "repeat_offenders": repeat_offenders,
    "treatment_intensity": treatment_intensity,
    "pre_breach_signature": pre_breach_signature,
    "predict_risk": predict_risk,
    "sql": sql,
    "schema": schema,
    "read_findings": read_findings,
    "list_research_sites": list_research_sites,
}
