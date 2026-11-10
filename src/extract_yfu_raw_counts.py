
from pathlib import Path
import csv
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]

SPIKE_FILE = ROOT / "data/raw/YFU/arithmetic/spikes.mat"
PRES_FILE = ROOT / "tables1/YFU_arithmetic_presentations.csv"
OUTPUT_FILE = ROOT / "tables1/YFU_arithmetic_raw_spike_counts.csv"

with open(PRES_FILE, newline="") as f:
    presentations = list(csv.DictReader(f))

with h5py.File(SPIKE_FILE, "r") as f:
    ir = f["spikes/ir"][:]
    jc = f["spikes/jc"][:]

n_neurons = 78
n_presentations = len(presentations)
n_columns = len(jc) - 1

X = np.zeros(
    (n_presentations, n_neurons),
    dtype=np.int32
)

# Each column represents a time index.
# jc stores the starting position of each column
# in the sparse row-index array ir.
for i, row in enumerate(presentations):
    onset = float(row["onset"])

    # Match the original strict comparison:
    # onset + 50 <= spike_index < onset + 950
    start = int(np.ceil(onset + 50))
    end = int(np.ceil(onset + 950))

    if start < 0 or end > n_columns:
        raise ValueError(
            f"Presentation {i}: window outside spike matrix"
        )

    # Gather sparse entries in the selected time columns.
    rows = ir[jc[start]:jc[end]]

    X[i] = np.bincount(
        rows.astype(np.int64),
        minlength=n_neurons
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [f"neuron_{j}" for j in range(n_neurons)]
    )
    writer.writerows(X)

print("=== EXTRACTION COMPLETE ===")
print("Shape:", X.shape)
print("Total counted spikes:", X.sum())
print("Minimum count:", X.min())
print("Maximum count:", X.max())
print("Saved:", OUTPUT_FILE)
