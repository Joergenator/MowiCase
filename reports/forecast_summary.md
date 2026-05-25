# 12-week breach-risk forecast

**Predict from week:** 2025-12-22
**Target week:** 2026-03-16  (= predict_from + 12 weeks)
**Sites scored:** 1083 commercial sites (HI research sites filtered out)
**Model:** LightGBM v1 (default), persisted booster `models/lgbm_v1_h12.txt`. v3 (with neighbor features) also available and compared below.

## Headline

Predicted breach probabilities range from
**0.002** to
**0.107**, with a median of
0.006. The top-20 cutoff is
**0.043** and the highest-risk site is **0.107**.

For context the 2025 holdout base rate is ≈0.04, so a
probability above the base rate flags above-average risk; the top-20 are
all at least 1.1× the base rate.

The top-20 carry an average recent FEMALEADULT of 0.26
lice/fish (regulatory limit 0.5) and an average sea temperature of
8.5 °C. They cluster geographically in mid-Norway and
the southern coast (see F2).

## Top-20 sites — predicted breach risk at h=12w (v1)

| SITENUMBER | SITENAME | PO_id | PO_name | lat | lon | FEMALEADULT | MOBILELICE | SEATEMP_C | proba |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12071 | Ommundsteigen | 2 | Ryfylket | 59.551 | 6.513 | 0.17 | 0.45 | 7.74 | 0.107 |
| 11625 | Langholmen | 3 | Karmøy til Sotra | 60.215 | 5.063 | 0.19 | 1.64 | 12.5 | 0.087 |
| 13225 | Kilaneset | 2 | Ryfylket | 59.313 | 6.267 | 0.4 | 4.83 | 9.21 | 0.086 |
| 15460 | Laholmen | 12 | Vest-Finnmark | 70.894 | 25.557 | 0.0 | 0.17 | 5.14 | 0.079 |
| 37717 | Vassgåsholmen | 6 | Nordmøre og Sør-Trøndelag | 63.592 | 9.34 | 0.09 | 0.21 | 8.84 | 0.076 |
| 45231 | Slønga | 8 | Helgeland til Bodø | 66.55 | 12.576 | 0.14 | 0.52 | 7.35 | 0.069 |
| 35377 | Kveitskjeret | 6 | Nordmøre og Sør-Trøndelag | 63.825 | 8.445 | 0.98 | 0.42 | 9.49 | 0.065 |
| 45122 | Lausund | 5 | Stadt til Hustadvika | 62.598 | 6.18 | 0.15 | 0.45 | 8.3 | 0.062 |
| 32437 | Urda | 6 | Nordmøre og Sør-Trøndelag | 63.001 | 8.327 | 0.07 | 0.04 | 8.37 | 0.058 |
| 45206 | Lyngholmane | 4 | Nordhordaland til Stadt | 61.193 | 4.852 | 0.45 | 3.91 | 9.31 | 0.057 |
| 45117 | Knubben | 7 | Nord-Trøndelag med Bindal | 64.558 | 10.798 |  |  |  | 0.053 |
| 36877 | Munkskjæra | 6 | Nordmøre og Sør-Trøndelag | 63.82 | 8.409 |  |  |  | 0.052 |
| 30137 | Edøya II | 6 | Nordmøre og Sør-Trøndelag | 63.658 | 8.684 | 0.05 | 1.53 | 8.9 | 0.051 |
| 45211 | Saltskår | 4 | Nordhordaland til Stadt | 61.193 | 4.815 | 0.45 | 5.52 | 9.09 | 0.05 |
| 45078 | Almbakkevika | 4 | Nordhordaland til Stadt | 61.62 | 5.272 |  |  |  | 0.05 |
| 38797 | Oløya N | 6 | Nordmøre og Sør-Trøndelag | 63.853 | 8.591 | 0.31 | 0.78 | 9.3 | 0.047 |
| 10838 | Slettnesfjord | 12 | Vest-Finnmark | 70.627 | 23.11 | 0.26 | 0.55 | 6.91 | 0.044 |
| 12890 | Høybuvika | 6 | Nordmøre og Sør-Trøndelag | 62.957 | 8.049 |  |  |  | 0.044 |
| 13084 | Kjørem | 6 | Nordmøre og Sør-Trøndelag | 63.892 | 9.983 | 0.1 | 0.11 | 7.3 | 0.044 |
| 39997 | Helligvær Ø | 9 | Vestfjorden og Vesterålen | 67.422 | 14.032 | 0.32 | 0.45 | 9.0 | 0.043 |


## v3 (with neighbor features) — does the spatial signal change the picture?

v3 = v1's 52 features **plus** 8 neighbor features (mean/max FEMALEADULT,
mean MOBILELICE, count of breaching neighbors — within 5 km and 10 km).
Inner-validation PR-AUC lifts visible at every horizon, biggest at h=12.

- **Top hit (v1):** Ommundsteigen  @  p=0.107
- **Top hit (v3):** Ommundsteigen  @  p=0.373
- **Biggest promotions** (sites v3 ranks higher than v1):
  - Urda: v1 rank 9 -> v3 rank 2
  - Helligvær Ø: v1 rank 20 -> v3 rank 15
  - Lyngholmane: v1 rank 10 -> v3 rank 6
- **Biggest demotions** (v1 false-flags v3 disagrees with):
  - Knubben: v1 rank 11 -> v3 rank 47
  - Vassgåsholmen: v1 rank 5 -> v3 rank 90
  - Edøya II: v1 rank 13 -> v3 rank 400

See `F5_v1_vs_v3_ranking.png` for the full union-of-top-20 comparison.
v3 makes more extreme predictions on a subset of sites where neighbors
are also under pressure, and dampens predictions where v1 was picking
up site-internal noise that the spatial context contradicts.

## Main drivers across the top-20 (v1)

Feature most often ranked as the #1 positive contributor:

- **FEMALEADULT_roll12_max** — dominant driver for 8 of the top-20 sites
- **site_age_weeks** — dominant driver for 4 of the top-20 sites
- **days_since_chem** — dominant driver for 3 of the top-20 sites
- **PRODUCTIONAREAID** — dominant driver for 2 of the top-20 sites
- **FEMALEADULT_roll12_mean** — dominant driver for 2 of the top-20 sites

The full per-site decomposition is in `F3_top_drivers_heatmap.png`.

## Calibration caveat

LightGBM v1 was trained without `scale_pos_weight`, so the probabilities
above are calibrated rather than rank-only — absolute values are
interpretable. **But:** the h=12 model has PR-AUC ≈ 0.10 on the
2025 holdout (real but modest — 1.2× the best naive baseline). The
ranking is reliable on the top tail; absolute probabilities for sites
near the base rate should be read as "uncertain, not safe", not
"benign". Operational decisions should weight the **per-site driver
breakdown** (F3) more than the raw score for borderline sites.

## How this was produced

- `src/features.py::build_inference_frame(lice, treat, horizon=12)`
  — builds one row per commercial site at its latest week with all 52
  model features.
- `src/models.py::LightGBMBreach.predict_contributions()` — wraps
  LightGBM's native `pred_contrib=True` decomposition; per-site
  per-feature contributions sum (plus bias) to the raw logit.
- HI research sites are excluded at the data layer via
  `src.research_sites.RESEARCH_SITE_IDS`.
- The same data + model is callable from the agent as `predict_risk`
  (ranking) and `predict_drivers` (per-site explanation).
