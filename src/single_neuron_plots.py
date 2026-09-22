"""Single-neuron visualization for the arithmetic task.

This module reuses the validated `analyze_neuron()` function from
`src/firing_rate_tuning.py`.

Given a subject and a MATLAB/paper neuron row number, it creates exactly
four plots:

1. Firing-rate tuning: mean +/- SEM for numbers 1-9.
2. Raster plot: spike trains grouped by number.
3. Gaussian-convolved temporal responses for numbers 1-9.
4. Gaussian-convolved temporal responses for selected numbers
   (default: 2, 5, 8).

Example
-------
from src.single_neuron_plots import analyze_single_neuron

out = analyze_single_neuron("YFU", 43)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .firing_rate_tuning import analyze_neuron
except ImportError:
    from firing_rate_tuning import analyze_neuron


def _gaussian_temporal_curves(
    presentations,
    spike_times,
    sigma_ms=150.0,
    display_start=0.0,
    display_end=1000.0,
    dt_ms=1.0,
    padding_sigma=4.0,
):
    """Smooth each presentation's spike train with a Gaussian kernel.

    Returns the pointwise mean and SEM across presentations for each
    number condition.
    """
    time = np.arange(display_start, display_end + dt_ms, dt_ms)
    padding = padding_sigma * sigma_ms

    # Gaussian normalization gives spikes/s; times below are in ms.
    norm = 1000.0 / (sigma_ms * np.sqrt(2.0 * np.pi))

    temporal = {}

    for number in range(1, 10):
        subset = presentations[presentations["number"] == number]
        trial_curves = []

        for _, row in subset.iterrows():
            onset = float(row["onset"])

            mask = (
                (spike_times >= onset + display_start - padding)
                & (spike_times <= onset + display_end + padding)
            )

            relative_spikes = spike_times[mask] - onset
            rate = np.zeros(time.size, dtype=float)

            for spike in relative_spikes:
                rate += norm * np.exp(
                    -0.5 * ((time - spike) / sigma_ms) ** 2
                )

            trial_curves.append(rate)

        trial_curves = np.asarray(trial_curves, dtype=float)

        if len(trial_curves) == 0:
            mean = np.full(time.size, np.nan)
            sem = np.full(time.size, np.nan)
        else:
            mean = trial_curves.mean(axis=0)

            if len(trial_curves) > 1:
                sem = (
                    trial_curves.std(axis=0, ddof=1)
                    / np.sqrt(len(trial_curves))
                )
            else:
                sem = np.zeros(time.size)

        temporal[number] = {
            "mean": mean,
            "sem": sem,
            "trials": trial_curves,
            "n": len(trial_curves),
        }

    return time, temporal


def _build_raster(
    presentations,
    spike_times,
    start_ms=0.0,
    end_ms=1000.0,
):
    """Return relative spike times for each number and presentation."""
    raster = {}

    for number in range(1, 10):
        subset = presentations[presentations["number"] == number]
        trials = []

        for _, row in subset.iterrows():
            onset = float(row["onset"])

            mask = (
                (spike_times >= onset + start_ms)
                & (spike_times <= onset + end_ms)
            )

            trials.append(spike_times[mask] - onset)

        raster[number] = trials

    return raster


def analyze_single_neuron(
    subject,
    neuron,
    root=None,
    sigma_ms=150.0,
    selected_numbers=(2, 5, 8),
    show=True,
    save=False,
    save_dir=None,
    verbose=True,
):
    """Generate four plots for one arithmetic-task neuron.

    Parameters
    ----------
    subject : str
        Subject code, for example "YFU".
    neuron : int
        MATLAB/paper neuron row number, 1-based. Example: 43.
    root : path-like or None
        Passed directly to `analyze_neuron()`.
    sigma_ms : float
        Gaussian standard deviation for temporal smoothing.
    selected_numbers : iterable of int
        Numbers shown in plot 4. Default is (2, 5, 8).
    show : bool
        Display figures when True.
    save : bool
        Save all four figures when True.
    save_dir : path-like or None
        Folder for saved figures. Defaults to `figures1`.
    verbose : bool
        Print neuron identity and basic information.

    Returns
    -------
    dict
        Data and matplotlib figure objects.
    """

    # ---------------------------------------------------------
    # Reuse the already validated firing-rate analysis/loading
    # ---------------------------------------------------------
    base = analyze_neuron(
        subject,
        neuron,
        root=root,
        plot=False,
        verbose=False,
    )

    region = str(base["region"])
    presentations = base["presentations"].copy()
    spike_times = np.asarray(base["spike_times"], dtype=float)
    tuning = base["tuning"].copy()

    # In firing_rate_tuning.py the number is the DataFrame index.
    numbers = tuning.index.to_numpy(dtype=int)
    means = tuning["mean"].to_numpy(dtype=float)
    sems = tuning["sem"].to_numpy(dtype=float)

    order = np.argsort(numbers)
    numbers = numbers[order]
    means = means[order]
    sems = sems[order]

    selected_numbers = tuple(int(n) for n in selected_numbers)

    if not selected_numbers:
        raise ValueError("selected_numbers cannot be empty.")

    if any(n < 1 or n > 9 for n in selected_numbers):
        raise ValueError("selected_numbers must contain only numbers 1-9.")

    neuron_name = f"{subject}.{region}.{neuron}"

    raster = _build_raster(
        presentations,
        spike_times,
        start_ms=0,
        end_ms=1000,
    )

    time, temporal = _gaussian_temporal_curves(
        presentations,
        spike_times,
        sigma_ms=sigma_ms,
        display_start=0,
        display_end=1000,
    )

    cmap = plt.get_cmap("tab10")

    # =========================================================
    # PLOT 1: FIRING-RATE TUNING
    # =========================================================
    fig_firing, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        numbers,
        means,
        yerr=sems,
        marker="o",
        capsize=4,
        linewidth=2,
    )

    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Number")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(
        f"{neuron_name} | Firing-rate tuning (mean +/- SEM)"
    )

    fig_firing.tight_layout()

    # =========================================================
    # PLOT 2: RASTER
    # =========================================================
    fig_raster, ax = plt.subplots(figsize=(11, 8))

    y = 0
    centers = []
    labels = []
    gap = 5

    for number in range(1, 10):
        start_y = y

        for trial in raster[number]:
            ax.eventplot(
                trial,
                lineoffsets=y,
                linelengths=1.5,
                linewidths=1.8,
                colors=[cmap(number - 1)],
            )
            y += 1

        end_y = max(start_y, y - 1)
        centers.append((start_y + end_y) / 2)
        labels.append(str(number))

        y += gap

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    ax.axvline(
        750,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlim(0, 1000)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Number condition")
    ax.set_title(f"{neuron_name} | Spike raster")

    fig_raster.tight_layout()

    # =========================================================
    # PLOT 3: GAUSSIAN CONVOLUTION, ALL NUMBERS
    # =========================================================
    fig_all, ax = plt.subplots(figsize=(10, 6))

    for number in range(1, 10):
        mean = temporal[number]["mean"]
        sem = temporal[number]["sem"]
        color = cmap(number - 1)

        ax.plot(
            time,
            mean,
            linewidth=2.2,
            color=color,
            label=str(number),
        )

        ax.fill_between(
            time,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.10,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    ax.axvline(
        750,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Gaussian-smoothed firing rate (Hz)")
    ax.set_title(
        f"{neuron_name} | Temporal responses "
        f"(Gaussian sigma = {sigma_ms:g} ms)"
    )
    ax.legend(title="Number", ncol=3)

    fig_all.tight_layout()

    # =========================================================
    # PLOT 4: SELECTED NUMBERS ONLY
    # Default: 2, 5, 8
    # =========================================================
    fig_selected, ax = plt.subplots(figsize=(8, 5))

    for number in selected_numbers:
        mean = temporal[number]["mean"]
        sem = temporal[number]["sem"]
        color = cmap(number - 1)

        ax.plot(
            time,
            mean,
            linewidth=2.8,
            color=color,
            label=f"Number {number}",
        )

        ax.fill_between(
            time,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.15,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    ax.axvline(
        750,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Gaussian-smoothed firing rate (Hz)")

    selected_text = ", ".join(str(n) for n in selected_numbers)

    ax.set_title(
        f"{neuron_name} | Temporal comparison: "
        f"numbers {selected_text}"
    )

    ax.legend()

    fig_selected.tight_layout()

    # =========================================================
    # OPTIONAL SAVE
    # =========================================================
    if save:
        if save_dir is None:
            save_dir = Path("figures1")
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{subject}_{region}_{neuron}"

        fig_firing.savefig(
            save_dir / f"{stem}_firing_rate_tuning.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig_raster.savefig(
            save_dir / f"{stem}_raster.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig_all.savefig(
            save_dir / f"{stem}_convolution_all_numbers.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig_selected.savefig(
            save_dir / f"{stem}_convolution_selected_numbers.png",
            dpi=300,
            bbox_inches="tight",
        )

    if verbose:
        print(f"Neuron: {neuron_name}")
        print(f"MATLAB/paper neuron row: {neuron}")
        print(f"Region: {region}")
        print(f"Selected temporal comparison: {selected_numbers}")
        print(f"Gaussian sigma: {sigma_ms:g} ms")

    if show:
        plt.show()
    else:
        plt.close(fig_firing)
        plt.close(fig_raster)
        plt.close(fig_all)
        plt.close(fig_selected)

    return {
        "subject": subject,
        "neuron": neuron,
        "region": region,
        "name": neuron_name,
        "tuning": tuning,
        "presentations": presentations,
        "spike_times": spike_times,
        "raster": raster,
        "time": time,
        "temporal": temporal,
        "selected_numbers": selected_numbers,
        "figure_firing_rate": fig_firing,
        "figure_raster": fig_raster,
        "figure_convolution_all": fig_all,
        "figure_convolution_selected": fig_selected,
    }



# from src.single_neuron_plots import analyze_single_neuron

# out = analyze_single_neuron("YFU", 43)