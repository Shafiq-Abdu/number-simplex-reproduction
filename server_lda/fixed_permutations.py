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
import pandas as pd

from preprocessing import load_subject, temporal_counts, SUBJECTS
from lda_core import lda_cv_accuracy


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "server_lda" / "results_fixed"
OUT.mkdir(parents=True, exist_ok=True)

N_PERM = 200
BIN_SIZE = 100
GAMMA = 0.5


def analyze_neuron(task):
    subject, neuron, X, y, folds = task

    output = OUT / f"{subject}_neuron_{neuron:03d}.npz"

    if output.exists():
        return subject, neuron, "SKIPPED"

    observed = lda_cv_accuracy(X, y, GAMMA, folds)

    seed = (
        20261118
        + list(SUBJECTS).index(subject) * 100000
        + neuron
    )
    rng = np.random.default_rng(seed)

    null = np.empty(N_PERM, dtype=float)

    for p in range(N_PERM):
        shuffled = rng.permutation(y)
        null[p] = lda_cv_accuracy(
            X, shuffled, GAMMA, folds
        )

    p_value = (
        1 + np.count_nonzero(null >= observed)
    ) / (N_PERM + 1)

    # Atomic checkpoint: never treat a partial file as complete.
    temporary = output.with_suffix(".tmp.npz")

    np.savez_compressed(
        temporary,
        subject=subject,
        neuron=neuron,
        observed=observed,
        null=null,
        p_value=p_value,
        bin_size=BIN_SIZE,
        gamma=GAMMA,
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

        X_all = temporal_counts(
            spikes, starts, BIN_SIZE
        )

        folds = 4 if subject == "YFM" else 10

        tasks = [
            (subject, neuron, X_all[:, neuron, :], y, folds)
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

    print("Fixed-parameter analysis complete.", flush=True)


if __name__ == "__main__":
    main()
