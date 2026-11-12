
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables1"

print("=" * 65)
print("YFU BAYESIAN DATA AUDIT")
print("=" * 65)

print(f"\nProject directory: {ROOT}")
print(f"Tables directory: {TABLES}")
print(f"Directory exists: {TABLES.exists()}")

if not TABLES.exists():
    raise FileNotFoundError(TABLES)

# Find relevant saved results.
files = sorted(
    f for f in TABLES.rglob("*")
    if f.is_file() and (
        "yfu" in f.name.lower()
        or "gamma" in f.name.lower()
        or "bayes" in f.name.lower()
        or "lambda" in f.name.lower()
    )
)

print(f"\nFound {len(files)} candidate files:\n")

for f in files:
    print(f.relative_to(ROOT))

# Inspect the known raw count file.
raw_path = TABLES / "YFU_arithmetic_raw_spike_counts.csv"

if raw_path.exists():
    df = pd.read_csv(raw_path)

    print("\n" + "=" * 65)
    print("YFU RAW SPIKE COUNTS")
    print("=" * 65)
    print("Shape:", df.shape)
    print("First 15 columns:", df.columns[:15].tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
else:
    print("\nRaw spike-count file not found at expected path.")

print("\nAudit complete.")
