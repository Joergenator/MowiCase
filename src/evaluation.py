"""Evaluation utilities for breach-prediction baselines and models.

Two evaluation lenses:

1. **Site-level** — each prediction is one site-week. Metrics:
   - PR-AUC (average precision): the right summary for a 4.5%-positive
     problem; accuracy and ROC-AUC are misleading at this imbalance.
   - Precision-at-top-K (P@K): of the K most-at-risk sites, what fraction
     actually breach? Operationally relevant: "if Mowi can only inspect K
     sites this week, how many would be real catches?"
   - Brier score: how well-calibrated are the probabilities?

2. **Count-level** — aggregate predicted probabilities per target week
   to get an expected count, compare against the actual number of breaches.
   - MAE on the count: how far off is the weekly total prediction?

Per-PO breakdown shows where models work and where they don't.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


# ---------------------------------------------------------------------------
# Site-level metrics
# ---------------------------------------------------------------------------

def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision among the top-k scoring rows. If fewer rows than k, use all."""
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    # argsort ascending; we want descending — slice from end
    top_idx = np.argsort(y_score)[-k:]
    return float(y_true[top_idx].mean())


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(brier_score_loss(y_true, y_score))


# ---------------------------------------------------------------------------
# Count-level metric
# ---------------------------------------------------------------------------

def count_mae(df: pd.DataFrame, y_score: np.ndarray) -> float:
    """Aggregate per target week and compare predicted-count vs actual-count.

    `df` must include `target_week_start` (the prediction's target week)
    and `target` (the true 0/1).
    """
    out = df[["target_week_start", "target"]].copy()
    out["y_score"] = y_score
    weekly = out.groupby("target_week_start").agg(
        predicted=("y_score", "sum"),
        actual=("target", "sum"),
    )
    return float((weekly["predicted"] - weekly["actual"]).abs().mean())


# ---------------------------------------------------------------------------
# One-row scorecard
# ---------------------------------------------------------------------------

@dataclass
class Score:
    baseline: str
    horizon: int
    n_test: int
    pos_rate: float
    pr_auc: float
    p_at_100: float
    brier: float
    count_mae: float

    def as_row(self) -> dict:
        return {
            "baseline": self.baseline,
            "horizon": self.horizon,
            "n_test": self.n_test,
            "pos_rate": round(self.pos_rate, 4),
            "PR-AUC": round(self.pr_auc, 4),
            "P@100": round(self.p_at_100, 4),
            "Brier": round(self.brier, 5),
            "count_MAE": round(self.count_mae, 2),
        }


def score_baseline(name: str, horizon: int, test_df: pd.DataFrame,
                   y_score: np.ndarray) -> Score:
    y_true = test_df["target"].astype(int).to_numpy()
    return Score(
        baseline=name,
        horizon=horizon,
        n_test=len(test_df),
        pos_rate=float(y_true.mean()),
        pr_auc=pr_auc(y_true, y_score),
        p_at_100=precision_at_k(y_true, y_score, k=100),
        brier=brier(y_true, y_score),
        count_mae=count_mae(test_df, y_score),
    )


# ---------------------------------------------------------------------------
# Per-PO breakdown
# ---------------------------------------------------------------------------

def pr_auc_by_po(test_df: pd.DataFrame, y_score: np.ndarray,
                  po_col: str = "PRODUCTIONAREAID") -> pd.Series:
    """Return PR-AUC per PO. NaN where the PO has no positives in the test."""
    df = test_df[[po_col, "target"]].copy()
    df["y_score"] = y_score
    out = {}
    for po, g in df.groupby(po_col):
        out[po] = pr_auc(g["target"].astype(int).to_numpy(), g["y_score"].to_numpy())
    return pd.Series(out, name="pr_auc").sort_index()
