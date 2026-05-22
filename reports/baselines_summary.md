# Baseline results — step 3

Test set: **2025 holdout** (untouched in development).
Training: 2012-01-02 → 2024-12-29.
Metric primary: **PR-AUC** (base rate 3.78%, accuracy is useless here).

## PR-AUC by horizon

| baseline         |      1 |      2 |     12 |
|:-----------------|-------:|-------:|-------:|
| B0 GlobalRate    | 0.0359 | 0.0363 | 0.0413 |
| B1 Persistence   | 0.1943 | 0.085  | 0.0478 |
| B2 SeasonalNaive | 0.0392 | 0.0396 | 0.0454 |
| B3 POWeekRate    | 0.0712 | 0.0721 | 0.0772 |

## Precision-at-100 by horizon

| baseline         |    1 |    2 |   12 |
|:-----------------|-----:|-----:|-----:|
| B0 GlobalRate    | 0.03 | 0.03 | 0.03 |
| B1 Persistence   | 0.34 | 0.21 | 0.16 |
| B2 SeasonalNaive | 0.01 | 0.03 | 0.11 |
| B3 POWeekRate    | 0.12 | 0.13 | 0.12 |

## Count-MAE by horizon (weekly aggregate)

| baseline         |     1 |     2 |    12 |
|:-----------------|------:|------:|------:|
| B0 GlobalRate    | 13.61 | 13.46 | 12.43 |
| B1 Persistence   |  5.54 |  8.59 | 17    |
| B2 SeasonalNaive |  7.75 |  7.73 |  7.31 |
| B3 POWeekRate    |  8.02 |  7.85 |  7.42 |

## Strongest baseline per horizon

|   horizon | baseline       |   PR-AUC |   P@100 |   count_MAE |
|----------:|:---------------|---------:|--------:|------------:|
|         1 | B1 Persistence |   0.1943 |    0.34 |        5.54 |
|         2 | B1 Persistence |   0.085  |    0.21 |        8.59 |
|        12 | B3 POWeekRate  |   0.0772 |    0.12 |        7.42 |

## What this tells us for step 4

- **At h=1** the bar to beat is **B1 Persistence** (PR-AUC 0.194).
  Recent lice counts dominate — any model that doesn't use lag features will fail here.
- **At h=12** the bar to beat is **B3 POWeekRate** (PR-AUC 0.077).
  Persistence breaks down over 12 weeks, so PO + seasonality become the dominant signal.
  Beating this requires temperature and treatment features.
- The **per-PO breakdown** shows baselines are blind in the Finnmark POs
  (insufficient positives) — a model that handles low-base-rate POs better
  will be a real improvement.
