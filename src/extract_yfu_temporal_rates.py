
from pathlib import Path
import csv
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]

SPIKE_FILE = ROOT / "data/raw/YFU/arithmetic/spikes.mat"
PRES_FILE = ROOT / "tables1/YFU_arithmetic_presentations.csv"

ORIGINAL_FILE = ROOT / "tables1/YFU_arithmetic_raw_spike_counts.csv"

COUNTS_FILE = ROOT / "tables1/YFU_arithmetic_temporal_counts_100ms.csv"
RATES_FILE = ROOT / "tables1/YFU_arithmetic_temporal_rates_100ms.csv"

N_NEURONS = 78
N_BINS = 9
BIN_MS = 100
WINDOW_START = 50

# ---------------------------------------------
# 1. Load presentations
# ---------------------------------------------

with open(PRES_FILE, newline="") as f:
    presentations = list(csv.DictReader(f))

n_presentations = len(presentations)

# ---------------------------------------------
# 2. Load sparse spike information
# ---------------------------------------------

with h5py.File(SPIKE_FILE, "r") as f:
    ir = f["spikes/ir"][:]
    jc = f["spikes/jc"][:]

n_columns = len(jc) - 1

# ---------------------------------------------
# 3. Extract 100 ms temporal spike counts
# ---------------------------------------------

temporal_counts = np.zeros(
    (n_presentations, N_NEURONS, N_BINS),
    dtype=np.int32
)

for i, row in enumerate(presentations):

    onset = float(row["onset"])

    for b in range(N_BINS):

        start = int(np.ceil(
            onset + WINDOW_START + b * BIN_MS
        ))

        end = int(np.ceil(
            onset + WINDOW_START + (b + 1) * BIN_MS
        ))

        if start < 0 or end > n_columns:
            raise ValueError(
                f"Presentation {i}, bin {b}: "
                "window outside spike matrix"
            )

        rows = ir[jc[start]:jc[end]]

        temporal_counts[i, :, b] = np.bincount(
            rows.astype(np.int64),
            minlength=N_NEURONS
        )

# ---------------------------------------------
# 4. Convert counts to firing rates (Hz)
# ---------------------------------------------

temporal_rates = temporal_counts / (BIN_MS / 1000)

# Flatten: presentations x (neurons * bins)
X_counts = temporal_counts.reshape(
    n_presentations, N_NEURONS * N_BINS
)

X_rates = temporal_rates.reshape(
    n_presentations, N_NEURONS * N_BINS
)

columns = [
    f"neuron_{j}_bin_{b+1}"
    for j in range(N_NEURONS)
    for b in range(N_BINS)
]

# ---------------------------------------------
# 5. Save CSV files
# ---------------------------------------------

COUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)

for path, data in [
    (COUNTS_FILE, X_counts),
    (RATES_FILE, X_rates)
]:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(data)

# ---------------------------------------------
# 6. Validate against original 900 ms counts
# ---------------------------------------------

with open(ORIGINAL_FILE, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    original_counts = np.array(
        [[int(value) for value in row] for row in reader],
        dtype=np.int32
    )

recovered_counts = temporal_counts.sum(axis=2)

if not np.array_equal(recovered_counts, original_counts):
    raise AssertionError(
        "Temporal counts do not match original counts!"
    )

# ---------------------------------------------
# 7. Summary
# ---------------------------------------------

print("=== TEMPORAL EXTRACTION COMPLETE ===")
print("Presentations:", n_presentations)
print("Neurons:", N_NEURONS)
print("Time bins:", N_BINS)
print("Bin duration:", BIN_MS, "ms")

print("Temporal tensor:", temporal_counts.shape)
print("Flattened counts:", X_counts.shape)
print("Flattened rates:", X_rates.shape)

print(
    "Original 900 ms counts recovered:",
    np.array_equal(recovered_counts, original_counts)
)

print("Total spikes:", temporal_counts.sum())
print("Mean firing rate (Hz):", temporal_rates.mean())

print("\nSaved:")
print(COUNTS_FILE)
print(RATES_FILE)
