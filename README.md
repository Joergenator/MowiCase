# Mowi BarentsWatch Lice Data Challenge

Take-home case: explore BarentsWatch lice and treatment data, build predictive models for weekly lice-limit breaches at site level, and create an LLM-based natural-language QA agent over the data.

See [PLAN.md](PLAN.md) for the sequential work plan and [case.txt](case.txt) for the original brief.

## Data

Two datasets from BarentsWatch (public):

- `vlice.csv` — weekly lice counts at site level
- `vtreatment.csv` — weekly treatment records at site level

Download both files from the OneDrive link in [case.txt](case.txt) and place them in `data/raw/`. The raw CSVs are **not** committed to this repo.

## Project structure

```
.
├── data/
│   └── raw/              # vlice.csv, vtreatment.csv (gitignored)
├── src/                  # reusable modules
│   ├── load_data.py      # data loading with hard 2026 cutoff
│   └── ...
├── notebooks/            # exploratory analysis, modeling experiments
├── models/               # trained model artefacts (gitignored)
├── reports/              # figures, presentation deck
├── PLAN.md               # work plan
└── README.md
```

## Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Leakage discipline

**No data from 2026 onward is used for training or feature engineering.** This is enforced by `src/load_data.py`, which exposes a `load_training_data()` function that filters all rows to dates < 2026-01-01 and asserts the cutoff at runtime. See the leakage check section below for how to verify.

### Verifying no 2026 data was used

```powershell
py src/check_leakage.py
```

## Reproducing the analysis

(to be filled in as work progresses)
