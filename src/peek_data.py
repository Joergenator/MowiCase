"""One-off script: schema, date range, row counts, missingness for both CSVs."""
import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

for name in ["vlice.csv", "vtreatment.csv"]:
    path = RAW / name
    print(f"\n{'=' * 70}\n{name}  ({path.stat().st_size / 1e6:.1f} MB)\n{'=' * 70}")

    # Use utf-8-sig to strip BOM if present
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    print(f"\nShape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissingness (%):\n{(df.isna().mean() * 100).round(2)}")
    print(f"\nDescribe (numeric):\n{df.describe(include='number').T}")
    print(f"\nDescribe (object):\n{df.describe(include='object').T}")
