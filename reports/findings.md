# EDA Findings — BarentsWatch Lice Data

**Data:** 783,477 weekly site rows · 1,674 sites · 13 POs · 2012-01-02–2025-12-22
**Overall breach rate (counted weeks):** 4.55%

## Key findings

1. **Treatment intensity varies ~3× across POs.** The top PO by treatments-per-site-year
   is `PO1: Svenskegrensen til Jæren` (8.10 treatments / active site-year);
   the lowest is `PO13: Øst-Finnmark` (0.64).

1b. **Excluding cleaner fish reshuffles the ranking dramatically (chart 10).**
   PO1 Svenskegrensen til Jæren drops from #1 (8.10) to #11 (1.70) because
   ~46 % of its treatments are passive cleaner-fish stocking. The "active
   intervention" leader is PO7 N-Trøndelag at 4.77. This separates preventive
   biological control (mostly the southwest coast, PO1-PO3) from reactive
   intervention intensity (mostly mid-Norway, PO4-PO8).

2. **Breach rates concentrate in a few POs.** Top-3 by breach rate:
   - PO5: Stadt til Hustadvika: 6.2%
   - PO3: Karmøy til Sotra: 5.8%
   - PO4: Nordhord. til Stadt: 5.3%
   The overall base rate is 4.55%.

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
   (~weeks 30–40), well after the spring 0.2-limit window (~W16-W26)
   when the regulator's stricter threshold protects out-migrating smolts.

4b. **The high-lice years are the warm years.** Comparing 2020–2025 in the peak
   window (weeks 25–40), the cross-year correlation between mean sea
   temperature and mean adult-female lice is **r = 0.84** — exactly the
   mechanism chart 3 implies, playing out at the annual scale. 2024 was the
   warmest year (mean 14.1 °C
   in W25-40) and had the highest peak lice (0.31).

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
