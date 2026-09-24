from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt


# MTL regions used in the paper
MTL_REGIONS = {"hpc", "ent", "amy", "para-hpc"}


def analyze_neuron(
    subject,
    neuron,
    root=None,
    plot=True,
    verbose=True,
):
    """
    Compute arithmetic-task firing-rate tuning for one neuron.

    Parameters
    ----------
    subject : str
        Subject ID, e.g. "YFU".

    neuron : int
        MATLAB/paper neuron row number.
        Example: YFU.hpc.43 -> neuron=43.
        Internally converted to Python row 42.

    root : str or Path, optional
        Root directory of the repository.
        If None, assumes the current working directory is a notebook
        inside notebooks1/.

    plot : bool
        Whether to plot the mean firing-rate tuning curve.

    verbose : bool
        Whether to print subject/neuron information.

    Returns
    -------
    dict
        Contains:
        - subject
        - matlab_row
        - python_row
        - region
        - total_neurons
        - region_counts
        - mtl_region_counts
        - presentations
        - tuning
        - spike_times
        - figure
        - axis
    """

    # ---------------------------------------------------------
    # 1. Locate subject data
    # ---------------------------------------------------------

    if root is None:
        root = Path.cwd().parent
    else:
        root = Path(root)

    data_dir = root / "data" / "raw" / subject / "arithmetic"

    # Behavior file
    behavior_file = data_dir / "photoBehavEvents.csv"

    if not behavior_file.exists():
        raise FileNotFoundError(
            f"Behavior file not found: {behavior_file}"
        )

    # Spike file
    # Most subjects use spikes.mat.
    # YFK, YFL, and YFM use spikesArithmetic.mat.
    spike_candidates = [
        data_dir / "spikes.mat",
        data_dir / "spikesArithmetic.mat",
    ]

    spike_file = next(
        (f for f in spike_candidates if f.exists()),
        None
    )

    if spike_file is None:
        raise FileNotFoundError(
            f"No arithmetic spike file found in {data_dir}. "
            f"Tried: {[f.name for f in spike_candidates]}"
        )

    # ---------------------------------------------------------
    # 2. Load behavioral data
    # ---------------------------------------------------------

    behav = pd.read_csv(behavior_file)

    required_columns = [
        "trial",
        "cue1",
        "cue2",
        "operationFirst",
        "tCue1",
        "tCue2",
        "tCue3",
    ]

    missing = [
        col for col in required_columns
        if col not in behav.columns
    ]

    if missing:
        raise ValueError(
            f"Missing behavioral columns: {missing}"
        )

    # ---------------------------------------------------------
    # 3. Determine actual operand onsets
    # ---------------------------------------------------------

# ---------------------------------------------------------
# 3. Determine actual operand onsets
# ---------------------------------------------------------

    operand_data = behav[
        [
            "trial",
            "cue1",
            "cue2",
            "operationFirst",
            "tCue1",
            "tCue2",
            "tCue3",
        ]
    ].copy()

    # Special case:
    # For YFR and YFS, the operation sign was always presented
    # as the second stimulus (Cue2 position).
    #
    # Therefore:
    #   operand 1 -> first presentation  -> tCue1
    #   operation -> second presentation -> tCue2
    #   operand 2 -> third presentation  -> tCue3
    #
    # For all other subjects, use operationFirst.

    if subject in {"YFR", "YFS"}:

        operand_data["operand1_onset"] = operand_data["tCue1"]
        operand_data["operand2_onset"] = operand_data["tCue3"]

    else:

    # operationFirst = 0:
    # operand1 -> first presentation
    # operand2 -> second presentation
    #
    # operationFirst = 1:
    # operation -> first presentation
    # operand1 -> second presentation
    # operand2 -> third presentation

        operand_data["operand1_onset"] = np.where(
            operand_data["operationFirst"] == 1,
            operand_data["tCue2"],
            operand_data["tCue1"],
        )

        operand_data["operand2_onset"] = np.where(
            operand_data["operationFirst"] == 1,
            operand_data["tCue3"],
            operand_data["tCue2"],
        )

    # ---------------------------------------------------------
    # 4. Pool operand 1 and operand 2 presentations
    # ---------------------------------------------------------

    op1 = operand_data[
        ["trial", "cue1", "operand1_onset"]
    ].copy()

    op1.columns = ["trial", "number", "onset"]
    op1["operand"] = 1

    op2 = operand_data[
        ["trial", "cue2", "operand2_onset"]
    ].copy()

    op2.columns = ["trial", "number", "onset"]
    op2["operand"] = 2

    presentations = pd.concat(
        [op1, op2],
        ignore_index=True,
    )

    # Keep numbers 1 through 9
    presentations = presentations[
        presentations["number"].between(1, 9)
    ].copy()

    # ---------------------------------------------------------
    # 5. Load spike matrix and region labels
    # ---------------------------------------------------------

    with h5py.File(spike_file, "r") as f:

    # Different subjects use different MATLAB variable names
        if "spikes" in f:
            g = f["spikes"]

        elif "spikesArithmetic" in f:
            g = f["spikesArithmetic"]

        else:
            raise KeyError(
                f"No recognized spike matrix found in {spike_file}. "
                f"Available keys: {list(f.keys())}"
            )

        data = g["data"][:]
        ir = g["ir"][:]
        jc = g["jc"][:]

        n_rows = int(g.attrs["MATLAB_sparse"])
        n_cols = len(jc) - 1

        spikes_sparse = sp.csc_matrix(
            (data, ir, jc),
            shape=(n_rows, n_cols),
        )

        regions = f["regionsVect"]

        region_names = []

        for ref in regions[0]:

            obj = f[ref]
            chars = obj[:].flatten()

            name = "".join(
                chr(int(c))
                for c in chars
            ).lower()

            region_names.append(name)

    # ---------------------------------------------------------
    # 6. Region information
    # ---------------------------------------------------------

    total_neurons = n_rows

    region_counts = Counter(region_names)

    mtl_region_counts = {
        region: count
        for region, count in region_counts.items()
        if region in MTL_REGIONS
    }

    total_mtl_neurons = sum(
        mtl_region_counts.values()
    )

    # ---------------------------------------------------------
    # 7. Select neuron
    # ---------------------------------------------------------

    matlab_row = neuron
    python_row = neuron - 1

    if python_row < 0 or python_row >= total_neurons:
        raise IndexError(
            f"Neuron {neuron} is outside the valid MATLAB "
            f"row range 1-{total_neurons}."
        )

    region = region_names[python_row]

    spike_times = (
        spikes_sparse
        .getrow(python_row)
        .nonzero()[1]
    )

    # ---------------------------------------------------------
    # 8. Compute firing rate for every presentation
    # ---------------------------------------------------------

    firing_rates = []
    spike_counts = []

    for _, row in presentations.iterrows():

        onset = row["onset"]

        # Paper arithmetic window:
        # 0.05 to 0.95 s after operand onset
        window_start = onset + 50
        window_end = onset + 950

        count = np.sum(
            (spike_times >= window_start)
            & (spike_times < window_end)
        )

        firing_rate = count / 0.9

        spike_counts.append(count)
        firing_rates.append(firing_rate)

    presentations["spike_count"] = spike_counts
    presentations["firing_rate"] = firing_rates

    # ---------------------------------------------------------
    # 9. Calculate tuning statistics
    # ---------------------------------------------------------

    tuning = (
        presentations
        .groupby("number")["firing_rate"]
        .agg(["count", "mean", "std"])
    )

    tuning["sem"] = (
        tuning["std"]
        / np.sqrt(tuning["count"])
    )

    # ---------------------------------------------------------
    # 10. Print summary
    # ---------------------------------------------------------

    if verbose:

        print("=" * 55)
        print(f"Subject: {subject}")
        print("=" * 55)

        print(f"\nTotal recorded neurons: {total_neurons}")

        print("\nAll recorded regions:")
        for r, count in sorted(region_counts.items()):
            print(f"  {r}: {count}")

        print("\nMTL regions used in paper:")
        for r, count in sorted(mtl_region_counts.items()):
            print(f"  {r}: {count}")

        print(f"\nTotal MTL neurons: {total_mtl_neurons}")

        print("\nSelected neuron:")
        print(f"  MATLAB row: {matlab_row}")
        print(f"  Python row: {python_row}")
        print(f"  Region: {region}")
        print(f"  Total spikes: {len(spike_times)}")

        if region not in MTL_REGIONS:
            print(
                "\nWARNING: This neuron is outside "
                "the paper's MTL regions."
            )

        print("\nNumber presentation counts:")
        print(
            presentations["number"]
            .value_counts()
            .sort_index()
        )

        print("\nFiring-rate tuning:")
        print(tuning)

    # ---------------------------------------------------------
    # 11. Plot tuning curve
    # ---------------------------------------------------------

    fig = None
    ax = None

    if plot:

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.errorbar(
            tuning.index,
            tuning["mean"],
            yerr=tuning["sem"],
            fmt="o-",
            capsize=4,
        )

        ax.set_xticks(range(1, 10))
        ax.set_xlabel("Number")
        ax.set_ylabel("Firing rate (Hz)")

        ax.set_title(
            f"{subject}.{region}.{matlab_row}"
        )

        fig.tight_layout()

        plt.show()

    # ---------------------------------------------------------
    # 12. Return everything useful
    # ---------------------------------------------------------

    return {
        "subject": subject,
        "matlab_row": matlab_row,
        "python_row": python_row,
        "region": region,
        "total_neurons": total_neurons,
        "total_mtl_neurons": total_mtl_neurons,
        "region_counts": dict(region_counts),
        "mtl_region_counts": mtl_region_counts,
        "presentations": presentations,
        "tuning": tuning,
        "spike_times": spike_times,
        "figure": fig,
        "axis": ax,
    }

# oUTPUT:
# from src.firing_rate_tuning import analyze_neuron

# result = analyze_neuron(
#     subject="YFS",
#     neuron=27
# )


