"""Systematic data-quality audit. Run before trusting the data for modelling.

Usage: py src/audit_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

from src.load_data import load_lice, load_treatment


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# Load with cutoff applied (mirrors how every consumer sees the data)
lice = load_lice(apply_cutoff=True)
treat = load_treatment(apply_cutoff=True)

print(f"lice:      {lice.shape}")
print(f"treatment: {treat.shape}")


# ---------------------------------------------------------------------------
# 1. Duplicate detection
# ---------------------------------------------------------------------------
header("1. Duplicates on the natural key (SITENUMBER, YEAR, WEEK)")
dup_key = lice.duplicated(subset=["SITENUMBER", "YEAR", "WEEK"], keep=False)
print(f"  lice rows with duplicate key: {dup_key.sum()}")
if dup_key.any():
    sample = (lice[dup_key]
              .sort_values(["SITENUMBER", "YEAR", "WEEK"])
              .head(8)
              [["SITENUMBER", "SITENAME", "YEAR", "WEEK",
                "FEMALEADULT", "MOBILELICE", "HAVECOUNTEDLICE", "BREACH"]])
    print("  sample of duplicated keys:")
    print(sample.to_string(index=False))

# treatment: many treatments can occur the same week at one site — duplicates expected
treat_dup_key = treat.duplicated(subset=["SITENUMBER", "YEAR", "WEEK", "ACTION"], keep=False)
print(f"\n  treatment rows with duplicate (site, week, ACTION): {treat_dup_key.sum()}")


# ---------------------------------------------------------------------------
# 2. NaT week-start (bad year/week combos that iso_week_to_date couldn't parse)
# ---------------------------------------------------------------------------
header("2. NaT WEEK_START values")
print(f"  lice NaT WEEK_START: {lice['WEEK_START'].isna().sum()}")
print(f"  treatment NaT WEEK_START: {treat['WEEK_START'].isna().sum()}")


# ---------------------------------------------------------------------------
# 3. HAVECOUNTEDLICE vs FEMALEADULT consistency
# ---------------------------------------------------------------------------
header("3. HAVECOUNTEDLICE consistency with lice-count columns")
mask_say_counted = lice["HAVECOUNTEDLICE"] == True
mask_say_not = lice["HAVECOUNTEDLICE"] == False

inconsistent_a = (mask_say_counted & lice["FEMALEADULT"].isna()).sum()
inconsistent_b = (mask_say_not & lice["FEMALEADULT"].notna()).sum()

print(f"  HAVECOUNTEDLICE=Ja but FEMALEADULT is null:   {inconsistent_a}")
print(f"  HAVECOUNTEDLICE=Nei but FEMALEADULT is not null: {inconsistent_b}")


# ---------------------------------------------------------------------------
# 4. BREACH flag vs FEMALEADULT > LICELIMITWEEK
# ---------------------------------------------------------------------------
header("4. BREACH flag vs derived breach (FEMALEADULT > LICELIMITWEEK)")
sub = lice.dropna(subset=["FEMALEADULT", "LICELIMITWEEK", "BREACH"]).copy()
derived = sub["FEMALEADULT"] > sub["LICELIMITWEEK"]
disagree = (derived != sub["BREACH"])
print(f"  Rows where source flag disagrees with FEMALEADULT > LICELIMITWEEK: "
      f"{disagree.sum()} / {len(sub)} ({disagree.mean():.3%})")
if disagree.any():
    sample = sub[disagree].head(8)[["SITENAME", "YEAR", "WEEK",
                                     "FEMALEADULT", "LICELIMITWEEK", "BREACH"]]
    print("  sample of disagreements:")
    print(sample.to_string(index=False))


# ---------------------------------------------------------------------------
# 5. Site → PO stability over time
# ---------------------------------------------------------------------------
header("5. Sites assigned to more than one PO over their history")
po_per_site = (lice.dropna(subset=["PRODUCTIONAREA"])
               .groupby("SITENUMBER")["PRODUCTIONAREA"].nunique())
unstable = po_per_site[po_per_site > 1]
print(f"  Sites whose PO changes over time: {len(unstable)} of {po_per_site.size}")
if len(unstable):
    examples = unstable.head(5).index.tolist()
    for s in examples:
        pos = lice[lice["SITENUMBER"] == s]["PRODUCTIONAREA"].dropna().unique()
        print(f"    site {s}: {list(pos)}")


# ---------------------------------------------------------------------------
# 6. Week 53 prevalence (chart 4 narrative depends on this not being weird)
# ---------------------------------------------------------------------------
header("6. ISO week 53 — how much data is there?")
w53 = lice[lice["WEEK"] == 53]
print(f"  week-53 lice rows: {len(w53)} ({len(w53)/len(lice):.2%} of total)")
print(f"  unique years with week 53: {sorted(w53['YEAR'].unique().tolist())}")


# ---------------------------------------------------------------------------
# 7. Remaining outliers / sentinel values
# ---------------------------------------------------------------------------
header("7. Possible sentinel values still hiding in numeric columns")
for col in ["FEMALEADULT", "MOBILELICE", "PERSISTENTLICE", "SEATEMPERATURE"]:
    s = lice[col].dropna()
    if len(s) == 0:
        continue
    # Common sentinels: 99, 999, -1, -999
    sentinels = [-999, -1, 99, 999]
    counts = {v: int((s == v).sum()) for v in sentinels if (s == v).any()}
    print(f"  {col:20s}  min={s.min():.2f}  max={s.max():.2f}  "
          f"q99={s.quantile(0.99):.2f}  sentinel hits={counts}")


# ---------------------------------------------------------------------------
# 8. Whitespace / casing in string columns
# ---------------------------------------------------------------------------
header("8. Whitespace and case anomalies in key string columns")
for col in ["PRODUCTIONAREA", "COUNTY", "SITENAME"]:
    s = lice[col].dropna().astype(str)
    with_space = ((s != s.str.strip()).sum())
    print(f"  {col:18s}  leading/trailing whitespace: {with_space}")

# OVERTHELICELIMITWEEK is what BREACH was built from — any unexpected casing?
ovr = lice["OVERTHELICELIMITWEEK"].dropna().astype(str).str.strip().unique()
print(f"\n  OVERTHELICELIMITWEEK unique values (stripped): {sorted(ovr)}")
liko = lice["LIKELYNOFISH"].dropna().unique()
print(f"  LIKELYNOFISH unique values: {sorted([str(v) for v in liko])}")


# ---------------------------------------------------------------------------
# 9. Sites with very few observations
# ---------------------------------------------------------------------------
header("9. Sites with very few weeks in the data")
weeks_per_site = lice.groupby("SITENUMBER").size()
print(f"  median weeks/site: {int(weeks_per_site.median())}")
print(f"  sites with <  10 weeks: {(weeks_per_site < 10).sum()}")
print(f"  sites with <  52 weeks (< 1 year): {(weeks_per_site < 52).sum()}")
print(f"  sites with < 104 weeks (< 2 years): {(weeks_per_site < 104).sum()}")


# ---------------------------------------------------------------------------
# 10. Same site, varying lat/lon (a site that "moved")
# ---------------------------------------------------------------------------
header("10. Sites with non-trivial position drift (>0.05° = ~5 km)")
def position_drift(g: pd.DataFrame) -> float:
    lat_range = g["LATITUDE"].max() - g["LATITUDE"].min()
    lon_range = g["LONGITUDE"].max() - g["LONGITUDE"].min()
    return max(lat_range, lon_range)

drift = lice.dropna(subset=["LATITUDE", "LONGITUDE"]).groupby("SITENUMBER").apply(
    position_drift, include_groups=False)
big_drift = drift[drift > 0.05]
print(f"  sites with >0.05° drift: {len(big_drift)} / {len(drift)}")
if len(big_drift):
    print(f"  max drift: {big_drift.max():.3f}° at site {big_drift.idxmax()}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE — review findings above")
print("=" * 70)
