import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from preprocessing import (
    load_subject, temporal_counts, SUBJECTS, BIN_SIZES, GAMMAS
)
from lda_core import optimized_lda_accuracy


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "server_lda" / "results_selection_aware"
OUT.mkdir(parents=True, exist_ok=True)

N_PERM = 200


def analyze_neuron(task):
    subject, neuron, X_by_bin, y, folds = task

    output = OUT / f"{subject}_neuron_{neuron:03d}.npz"

    if output.exists():
        return subject, neuron, "SKIPPED"

    observed, best_bin, best_gamma = optimized_lda_accuracy(
        X_by_bin, y, folds, GAMMAS
    )

    seed = (
        20261119
        + list(SUBJECTS).index(subject) * 100000
        + neuron
    )
    rng = np.random.default_rng(seed)

    null = np.empty(N_PERM, dtype=float)

    for p in range(N_PERM):
        shuffled = rng.permutation(y)

        null[p], _, _ = optimized_lda_accuracy(
            X_by_bin, shuffled, folds, GAMMAS
        )

    p_value = (
        1 + np.count_nonzero(null >= observed)
    ) / (N_PERM + 1)

    temporary = output.with_suffix(".tmp.npz")

    np.savez_compressed(
        temporary,
        subject=subject,
        neuron=neuron,
        observed=observed,
        best_bin=best_bin,
        best_gamma=best_gamma,
        null=null,
        p_value=p_value,
        n_permutations=N_PERM,
        seed=seed,
    )
    os.replace(temporary, output)

    return subject, neuron, f"p={p_value:.5f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=list(SUBJECTS),
        choices=list(SUBJECTS),
    )
    args = parser.parse_args()

    for subject in args.subjects:
        spikes, starts, y, trials, indices = load_subject(
            subject
        )

        X_all = {
            bin_size: temporal_counts(
                spikes, starts, bin_size
            )
            for bin_size in BIN_SIZES
        }

        folds = 4 if subject == "YFM" else 10

        tasks = [
            (
                subject,
                neuron,
                {
                    bin_size: X_all[bin_size][:, neuron, :]
                    for bin_size in BIN_SIZES
                },
                y,
                folds,
            )
            for neuron in range(len(indices))
        ]

        print(
            f"{subject}: starting {len(tasks)} neurons",
            flush=True,
        )

        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as pool:
            for completed, result in enumerate(
                pool.map(analyze_neuron, tasks), start=1
            ):
                print(
                    f"{subject}: {completed}/{len(tasks)} "
                    f"{result}",
                    flush=True,
                )

    print(
        "Selection-aware permutation analysis complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
