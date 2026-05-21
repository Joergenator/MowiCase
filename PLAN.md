# Mowi BarentsWatch Lice Data Challenge — Plan

Sequential plan for the take-home case. Work one step at a time.

## Constraints to respect throughout

- **No 2026 data** may be used for training or feature engineering. This is a hard rule and the single easiest place to lose credibility.
- Reviewed **relative to time spent** — depth and judgment matter more than checking every box.
- 20-min presentation + 10-min Q&A — everything built must fit a story we can tell in 20 minutes.
- GitHub deployment is a plus — reproducibility and proof of leakage discipline.
- Frame conclusions in operational terms (when to treat, where to monitor), not ML metrics alone.

## Tooling decisions

- **Python** with pandas, scikit-learn, LightGBM, matplotlib/seaborn
- **DuckDB** as the analytics SQL layer (reads CSVs directly, fast aggregations, shared backend between EDA and LLM agent)
- **Anthropic SDK** for the LLM agent (tool use, no LangChain)
- **Git + GitHub** from commit 1

---

## 1. Foundation

- Initialize Git repo with clean structure: `data/`, `notebooks/`, `src/`, `models/`, `README.md`, `.gitignore`
- Push to GitHub from the first commit
- Download both CSVs (`vlice`, `vtreatment`) into `data/raw/`
- Write `load_data.py` with a hard 2026-01-01 cutoff for training data — this is the leakage firewall everything else depends on
- First peek at the data: column names, row counts, date ranges, missingness

## 2. Exploratory analysis

- Confirm the breach definition from the BarentsWatch docs (which lice metric, which threshold, spring rules)
- Answer each EDA question from the case spec with one clear visual:
  - PO-level treatment intensity (treatments per active site)
  - Which POs breach lice limits most consistently
  - Temperature vs. lice pressure per PO
  - Seasonal and geographic patterns
  - Adult-female ↔ mobile-lice correlation per PO (with lags — they're biologically connected through the life cycle)
- Aim for ~5–8 strong charts, not a tour of the data

## 3. Modeling — baselines first

- Define the target precisely: weekly breach (0/1) and breach count per site
- Build two trivial baselines:
  - Persistence ("next week = this week")
  - Seasonal naive (same week last year)
- Set up time-series cross-validation with expanding window; hold out all of 2025 as untouched validation
- Write a unit test that asserts no feature uses data from after the prediction time

## 4. Modeling — main models

- One gradient boosting model (LightGBM), trained as direct forecasts for 1, 2, and 12 weeks ahead
- Features:
  - Lagged lice counts (adult female, mobile, fixed)
  - Lagged treatments
  - Temperature
  - Latitude / longitude
  - Neighboring-site lice pressure
  - Seasonal encodings (week of year, sin/cos)
  - Previous breach history
- Evaluate against the baselines:
  - MAE / RMSE for counts
  - Accuracy / F1 / precision-recall for binary breach
- Error analysis by PO — where does the model fail, and why?

## 5. LLM agent

- Load both CSVs into DuckDB
- Build a small set of tool functions the LLM can call (or let it write SQL directly against DuckDB)
- Use Anthropic SDK tool use, no LangChain
- Hard-test the example questions from the case spec so the demo is bulletproof:
  - Which areas currently show increasing lice pressure?
  - Which sites have had repeated breaches?
  - Which production areas have the highest treatment intensity?
  - What patterns are visible before breaches occur?
  - Which sites appear most at risk in the coming weeks?

## 6. Bonus — 12-week risk for all sites

- Run the trained model forward from the latest available week
- Rank sites by predicted breach risk
- Use SHAP or LightGBM feature importance to explain the top drivers per site

## 7. Bonus — Stata → Python port

- Translate the SGR salmon growth model (Handeland et al. 2008) to Python
- Use numpy for the daily Euler integration, matplotlib for the temperature-curve plot
- Match the original plot structure (lines per temperature, optimum at 14°C highlighted)

## 8. Presentation

- 20-min deck:
  1. Problem framing — what's a breach, why it matters operationally
  2. Key data findings (the 5–8 charts from step 2)
  3. Model results vs. baselines, with honest error analysis
  4. Live agent demo
  5. Production deployment story — how this would run in operations at Mowi
- Push final repo with a README that lets the reviewer reproduce everything end-to-end
- Include a leakage-check section in the README showing how the 2026 cutoff is enforced

---

**Current status:** waiting for CSVs to be downloaded into `data/raw/` before starting step 1.
