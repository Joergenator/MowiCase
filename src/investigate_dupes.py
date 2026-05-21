"""Investigate the duplicate rows found by the audit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.load_data import load_lice, load_treatment

lice = load_lice(apply_cutoff=True)
treat = load_treatment(apply_cutoff=True)

print("=== LICE DUPLICATES ===\n")

# Are duplicates on the natural key also exact duplicates on all data columns?
key = ["SITENUMBER", "YEAR", "WEEK"]
# When using include_groups=False, the group keys are dropped from the frame
data_cols = [c for c in lice.columns if c not in ["WEEK_START"] + key]

# Mark dup groups
lice["_dup_group"] = lice.groupby(key, dropna=False).cumcount()
dup_mask = lice.duplicated(subset=key, keep=False)
dups = lice[dup_mask].sort_values(key + ["_dup_group"])

# How many distinct value-rows per (site, year, week)?
distinct_per_key = (lice[dup_mask]
                    .groupby(key)
                    .apply(lambda g: g[data_cols].drop_duplicates().shape[0],
                           include_groups=False))

print(f"Total duplicate rows (on key): {dup_mask.sum()}")
print(f"Distinct (site,year,week) groups with duplicates: {distinct_per_key.size}")
print(f"  - groups where ALL columns identical (true dupes): {(distinct_per_key == 1).sum()}")
print(f"  - groups where some values differ (legitimate?): {(distinct_per_key > 1).sum()}")

# Show 5 examples of groups where values DIFFER (if any)
diff_keys = distinct_per_key[distinct_per_key > 1].head(5).index
for k in diff_keys:
    site, year, week = k
    rows = lice[(lice["SITENUMBER"] == site) &
                (lice["YEAR"] == year) & (lice["WEEK"] == week)]
    print(f"\nDIFFERING values at site={site}, year={year}, week={week}:")
    print(rows[["FEMALEADULT", "MOBILELICE", "PERSISTENTLICE",
                "HAVECOUNTEDLICE", "BREACH", "LIKELYNOFISH",
                "SEATEMPERATURE", "PRODUCTIONAREA"]].to_string())

print("\n=== TREATMENT DUPLICATES ===\n")

treat_key = ["SITENUMBER", "YEAR", "WEEK", "ACTION"]
treat_dup_mask = treat.duplicated(subset=treat_key, keep=False)

# Are they identical on all other columns?
treat_data_cols = [c for c in treat.columns if c not in ["WEEK_START"] + treat_key]
distinct_per_treat_key = (treat[treat_dup_mask]
                          .groupby(treat_key)
                          .apply(lambda g: g[treat_data_cols].drop_duplicates().shape[0],
                                 include_groups=False))
print(f"Total treatment dup rows: {treat_dup_mask.sum()}")
print(f"Distinct (site,year,week,action) groups with dups: {distinct_per_treat_key.size}")
print(f"  - groups where ALL columns identical: {(distinct_per_treat_key == 1).sum()}")
print(f"  - groups where some values differ: {(distinct_per_treat_key > 1).sum()}")

# Show 5 examples where treatment values differ
diff_treat_keys = distinct_per_treat_key[distinct_per_treat_key > 1].head(5).index
for k in diff_treat_keys:
    site, year, week, action = k
    rows = treat[(treat["SITENUMBER"] == site) & (treat["YEAR"] == year) &
                 (treat["WEEK"] == week) & (treat["ACTION"] == action)]
    print(f"\nDIFFERING values at site={site}, year={year}, week={week}, action={action!r}:")
    print(rows[["TYPEOFTREATMENT", "ACTIVEINGREDIENT", "CLEANERFISH",
                "SPECIESID", "SCOPE"]].to_string())


print("\n=== POSITION-DRIFT SITES ===\n")

import pandas as pd

def position_drift_detail(g):
    return pd.Series({
        "lat_min": g["LATITUDE"].min(),
        "lat_max": g["LATITUDE"].max(),
        "lon_min": g["LONGITUDE"].min(),
        "lon_max": g["LONGITUDE"].max(),
        "n_weeks": len(g),
        "n_distinct_locations": g[["LATITUDE", "LONGITUDE"]].drop_duplicates().shape[0],
    })

drift = (lice.dropna(subset=["LATITUDE", "LONGITUDE"])
         .groupby("SITENUMBER").apply(position_drift_detail, include_groups=False))
drift["max_drift_deg"] = drift[["lat_max", "lon_max"]].max(axis=1) - drift[["lat_min", "lon_min"]].min(axis=1)
big = drift[(drift["lat_max"] - drift["lat_min"] > 0.05) |
            (drift["lon_max"] - drift["lon_min"] > 0.05)]
print(big.sort_values("n_distinct_locations", ascending=False))


print("\n=== PERSISTENTLICE = 99 ROW ===\n")
weird = lice[lice["PERSISTENTLICE"] == 99]
print(weird[["SITENAME", "YEAR", "WEEK", "FEMALEADULT", "MOBILELICE",
             "PERSISTENTLICE", "BREACH", "PRODUCTIONAREA"]].to_string())
