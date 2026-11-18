from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sorted_data"

SUBJECTS = {
    "YFM": ("spikesArithmetic.mat", "spikesArithmetic"),
    "YFS": ("spikes.mat", "spikes"),
    "YFT": ("spikes.mat", "spikes"),
}

MTL_REGIONS = {"hpc", "ent", "amy", "para-hpc"}
BIN_SIZES = [60, 75, 90, 100, 150, 180, 225, 300, 450, 900]
GAMMAS = [0.2, 0.5, 0.8]


def load_subject(subject):
    filename, spike_key = SUBJECTS[subject]
    folder = DATA / subject / "arithmetic"

    # Reproduce notebook cell 17: presentation selection.
    df = pd.read_csv(folder / "photoBehavEvents.csv")
    df = df[df["toExclude"] == 0].copy()

    if subject in {"YFR", "YFS"}:
        df["operand1_onset"] = df["tCue1"]
        df["operand2_onset"] = df["tCue3"]
    else:
        df["operand1_onset"] = np.where(
            df["operationFirst"] == 1, df["tCue2"], df["tCue1"]
        )
        df["operand2_onset"] = np.where(
            df["operationFirst"] == 1, df["tCue3"], df["tCue2"]
        )

    op1 = df[["trial", "cue1", "operand1_onset"]].copy()
    op1.columns = ["trial", "number", "onset"]
    op1["operand"] = 1

    op2 = df[["trial", "cue2", "operand2_onset"]].copy()
    op2.columns = ["trial", "number", "onset"]
    op2["operand"] = 2

    presentations = pd.concat([op1, op2], ignore_index=True)
    presentations = presentations[
        presentations["number"].between(1, 9)
    ].copy()
    presentations = presentations.sort_values(
        ["onset", "trial", "operand"]
    ).reset_index(drop=True)

    if presentations["onset"].isna().any():
        raise ValueError(f"{subject}: missing presentation onsets")

    # Reproduce notebook cells 21 and 127.
    with h5py.File(folder / filename, "r") as f:
        refs = f["regionsVect"][:].ravel()
        regions = [
            "".join(chr(int(x)) for x in f[ref][:].ravel())
            .strip()
            .lower()
            for ref in refs
        ]

        indices = np.flatnonzero(
            np.isin(np.asarray(regions), list(MTL_REGIONS))
        )

        group = f[spike_key]
        data = group["data"][:]
        ir = group["ir"][:].astype(np.int64)
        jc = group["jc"][:].astype(np.int64)

        n_units = int(group.attrs["MATLAB_sparse"])
        n_samples = len(jc) - 1

    spikes = csc_matrix(
        (data, ir, jc), shape=(n_units, n_samples)
    )[indices, :].tocsr()

    starts = np.floor(
        presentations["onset"].to_numpy(dtype=float) + 50
    ).astype(int)

    if np.any(starts < 0) or np.any(starts + 900 > n_samples):
        raise ValueError(f"{subject}: presentation window out of bounds")

    y = presentations["number"].to_numpy(dtype=int)
    trials = presentations["trial"].to_numpy()

    return spikes, starts, y, trials, indices


def temporal_counts(spikes, starts, bin_size):
    if 900 % bin_size:
        raise ValueError("Bin size must divide 900 ms exactly")

    n_bins = 900 // bin_size
    X = np.zeros(
        (len(starts), spikes.shape[0], n_bins),
        dtype=np.int32,
    )

    for i, start in enumerate(starts):
        for b in range(n_bins):
            left = start + b * bin_size
            right = left + bin_size
            X[i, :, b] = np.asarray(
                spikes[:, left:right].sum(axis=1)
            ).ravel()

    return X


if __name__ == "__main__":
    for subject in SUBJECTS:
        spikes, starts, y, trials, indices = load_subject(subject)
        X = temporal_counts(spikes, starts, 100)

        print(
            f"{subject}: presentations={len(y)}, "
            f"MTL neurons={len(indices)}, "
            f"100ms shape={X.shape}, "
            f"unique trials={len(np.unique(trials))}",
            flush=True,
        )
