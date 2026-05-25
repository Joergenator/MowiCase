# Baseline results — step 3

Test set: **2025 holdout** (untouched in development).
Training: 2012-01-02 → 2024-12-29.
Metric primary: **PR-AUC** (base rate 3.69%, accuracy is useless here).

## PR-AUC by horizon

| baseline         |      1 |      2 |     12 |
|:-----------------|-------:|-------:|-------:|
| B0 GlobalRate    | 0.035  | 0.0354 | 0.0403 |
| B1 Persistence   | 0.1897 | 0.0796 | 0.0466 |
| B2 SeasonalNaive | 0.0383 | 0.0387 | 0.0445 |
| B3 POWeekRate    | 0.0701 | 0.071  | 0.0757 |

## Precision-at-100 by horizon

| baseline         |    1 |    2 |   12 |
|:-----------------|-----:|-----:|-----:|
| B0 GlobalRate    | 0.03 | 0.03 | 0.03 |
| B1 Persistence   | 0.38 | 0.14 | 0.12 |
| B2 SeasonalNaive | 0.04 | 0.07 | 0.05 |
| B3 POWeekRate    | 0.13 | 0.14 | 0.13 |

## Count-MAE by horizon (weekly aggregate)

| baseline         |     1 |     2 |    12 |
|:-----------------|------:|------:|------:|
| B0 GlobalRate    | 13.49 | 13.32 | 12.28 |
| B1 Persistence   |  5.52 |  8.47 | 16.72 |
| B2 SeasonalNaive |  7.81 |  7.76 |  7.37 |
| B3 POWeekRate    |  8.04 |  7.87 |  7.55 |

## Strongest baseline per horizon

|   horizon | baseline       |   PR-AUC |   P@100 |   count_MAE |
|----------:|:---------------|---------:|--------:|------------:|
|         1 | B1 Persistence |   0.1897 |    0.38 |        5.52 |
|         2 | B1 Persistence |   0.0796 |    0.14 |        8.47 |
|        12 | B3 POWeekRate  |   0.0757 |    0.13 |        7.55 |

## What this tells us for step 4

- **At h=1** the bar to beat is **B1 Persistence** (PR-AUC 0.190).
  Recent lice counts dominate — any model that doesn't use lag features will fail here.
- **At h=12** the bar to beat is **B3 POWeekRate** (PR-AUC 0.076).
  Persistence breaks down over 12 weeks, so PO + seasonality become the dominant signal.
  Beating this requires temperature and treatment features.
- The **per-PO breakdown** shows baselines are blind in the Finnmark POs
  (insufficient positives) — a model that handles low-base-rate POs better
  will be a real improvement.
