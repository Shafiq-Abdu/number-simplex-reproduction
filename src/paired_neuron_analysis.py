"""
Paired-neuron number decoding.

This module extends the single-neuron decoding analysis to pairs of neurons
recorded from the same subject.

For each pair, it computes:

1. Firing-rate decoding
2. Temporal decoding
3. Cross-validated hyperparameter selection
4. Label-shuffle permutation tests
5. Number-coding classification

The two neurons must share the same behavioral presentations.
"""

import numpy as np
import pandas as pd

from src.firing_rate_tuning import analyze_neuron
from src.number_decoding_analysis import (
    make_firing_rate_features,
    make_temporal_features,
    cv_lda_accuracy,
)


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

BIN_SIZES = [
    60, 75, 90, 100, 150,
    180, 225, 300, 450, 900
]

GAMMAS = [0.2, 0.5, 0.8]


# ============================================================
# MAIN FUNCTION
# ============================================================

def analyze_neuron_pair(
    subject,
    neuron_1,
    neuron_2,
    root,
    n_permutations=200,
    random_state=42,
    permutation_seed=12345,
):
    """
    Analyze number decoding for a pair of neurons.

    Parameters
    ----------
    subject : str
        Subject ID, for example "YFF".

    neuron_1 : int
        First neuron number.

    neuron_2 : int
        Second neuron number.

    root : pathlib.Path
        Project root directory.

    n_permutations : int
        Number of label-shuffle permutations.

    random_state : int
        Random state used for cross-validation.

    permutation_seed : int
        Random seed used for label permutations.

    Returns
    -------
    dict
        Summary of firing-rate and temporal decoding results.
    """

    # --------------------------------------------------------
    # 1. LOAD BOTH NEURONS
    # --------------------------------------------------------

    result_1 = analyze_neuron(
        subject,
        neuron_1,
        root=root,
        plot=False,
        verbose=False
    )

    result_2 = analyze_neuron(
        subject,
        neuron_2,
        root=root,
        plot=False,
        verbose=False
    )

    presentations_1 = result_1["presentations"]
    presentations_2 = result_2["presentations"]

    spike_times_1 = result_1["spike_times"]
    spike_times_2 = result_2["spike_times"]


    # --------------------------------------------------------
    # 2. VERIFY PRESENTATION ALIGNMENT
    # --------------------------------------------------------

    if len(presentations_1) != len(presentations_2):
        raise ValueError(
            f"Presentation counts differ for neurons "
            f"{neuron_1} and {neuron_2}."
        )

    numbers_1 = presentations_1["number"].to_numpy()
    numbers_2 = presentations_2["number"].to_numpy()

    onsets_1 = presentations_1["onset"].to_numpy()
    onsets_2 = presentations_2["onset"].to_numpy()

    if not np.array_equal(numbers_1, numbers_2):
        raise ValueError(
            "Number labels are not aligned between neurons."
        )

    if not np.allclose(onsets_1, onsets_2, equal_nan=True):
        raise ValueError(
            "Presentation onsets are not aligned between neurons."
        )

    presentations = presentations_1


    # --------------------------------------------------------
    # 3. FIRING-RATE FEATURES
    # --------------------------------------------------------

    X1_fr, y1 = make_firing_rate_features(
        presentations,
        spike_times_1
    )

    X2_fr, y2 = make_firing_rate_features(
        presentations,
        spike_times_2
    )

    if not np.array_equal(y1, y2):
        raise ValueError(
            "Firing-rate labels are not aligned."
        )

    y = y1

    X_fr_pair = np.hstack([
        X1_fr,
        X2_fr
    ])


    # --------------------------------------------------------
    # 4. DETERMINE NUMBER OF CV FOLDS
    # --------------------------------------------------------

    _, class_counts = np.unique(
        y,
        return_counts=True
    )

    min_class_count = int(class_counts.min())

    n_splits = min(
        10,
        min_class_count
    )


    # --------------------------------------------------------
    # 5. FIRING-RATE GRID SEARCH
    # --------------------------------------------------------

    fr_results = []

    for gamma in GAMMAS:

        accuracy = cv_lda_accuracy(
            X_fr_pair,
            y,
            gamma=gamma,
            n_splits=n_splits,
            random_state=random_state
        )

        fr_results.append(
            (gamma, accuracy)
        )

    best_fr_gamma, best_fr_accuracy = max(
        fr_results,
        key=lambda x: x[1]
    )


    # --------------------------------------------------------
    # 6. TEMPORAL GRID SEARCH
    # --------------------------------------------------------

    tc_results = []

    for bin_size in BIN_SIZES:

        X1_tc, y1_tc, edges1 = make_temporal_features(
            presentations,
            spike_times_1,
            bin_size=bin_size
        )

        X2_tc, y2_tc, edges2 = make_temporal_features(
            presentations,
            spike_times_2,
            bin_size=bin_size
        )

        if not np.array_equal(y1_tc, y2_tc):
            raise ValueError(
                "Temporal labels are not aligned."
            )

        if not np.array_equal(y, y1_tc):
            raise ValueError(
                "Temporal labels differ from firing-rate labels."
            )

        if not np.array_equal(edges1, edges2):
            raise ValueError(
                "Temporal bin edges differ between neurons."
            )

        X_tc_pair = np.hstack([
            X1_tc,
            X2_tc
        ])

        for gamma in GAMMAS:

            accuracy = cv_lda_accuracy(
                X_tc_pair,
                y,
                gamma=gamma,
                n_splits=n_splits,
                random_state=random_state
            )

            tc_results.append(
                (
                    bin_size,
                    gamma,
                    accuracy
                )
            )

    best_tc_bin, best_tc_gamma, best_tc_accuracy = max(
        tc_results,
        key=lambda x: x[2]
    )


    # --------------------------------------------------------
    # 7. REBUILD BEST TEMPORAL REPRESENTATION
    # --------------------------------------------------------

    X1_tc_best, y1_best, _ = make_temporal_features(
        presentations,
        spike_times_1,
        bin_size=best_tc_bin
    )

    X2_tc_best, y2_best, _ = make_temporal_features(
        presentations,
        spike_times_2,
        bin_size=best_tc_bin
    )

    if not np.array_equal(y1_best, y2_best):
        raise ValueError(
            "Best temporal representation is not aligned."
        )

    X_tc_pair_best = np.hstack([
        X1_tc_best,
        X2_tc_best
    ])


    # --------------------------------------------------------
    # 8. PERMUTATION TEST
    # --------------------------------------------------------

    rng = np.random.default_rng(
        permutation_seed
    )

    fr_null = np.empty(
        n_permutations
    )

    tc_null = np.empty(
        n_permutations
    )

    for p in range(n_permutations):

        # One common label permutation.
        # The two neurons remain paired.
        y_shuffle = rng.permutation(y)

        fr_null[p] = cv_lda_accuracy(
            X_fr_pair,
            y_shuffle,
            gamma=best_fr_gamma,
            n_splits=n_splits,
            random_state=random_state
        )

        tc_null[p] = cv_lda_accuracy(
            X_tc_pair_best,
            y_shuffle,
            gamma=best_tc_gamma,
            n_splits=n_splits,
            random_state=random_state
        )


    # --------------------------------------------------------
    # 9. PERMUTATION P-VALUES
    # --------------------------------------------------------

    fr_p = (
        1
        + np.sum(
            fr_null >= best_fr_accuracy
        )
    ) / (
        n_permutations + 1
    )

    tc_p = (
        1
        + np.sum(
            tc_null >= best_tc_accuracy
        )
    ) / (
        n_permutations + 1
    )


    # --------------------------------------------------------
    # 10. RETURN SUMMARY
    # --------------------------------------------------------

    return {
        "subject": subject,

        "neuron_1": int(neuron_1),
        "neuron_2": int(neuron_2),

        "n_presentations": int(len(y)),
        "min_class_count": min_class_count,
        "n_splits": int(n_splits),

        # Firing-rate results
        "FR_accuracy": float(best_fr_accuracy),
        "FR_best_gamma": float(best_fr_gamma),
        "FR_null_mean": float(fr_null.mean()),
        "FR_null_sd": float(
            fr_null.std(ddof=1)
        ),
        "FR_p": float(fr_p),
        "FR_coding": bool(fr_p < 0.05),

        # Temporal results
        "TC_accuracy": float(best_tc_accuracy),
        "TC_best_bin_ms": int(best_tc_bin),
        "TC_best_gamma": float(best_tc_gamma),
        "TC_null_mean": float(tc_null.mean()),
        "TC_null_sd": float(
            tc_null.std(ddof=1)
        ),
        "TC_p": float(tc_p),
        "TC_coding": bool(tc_p < 0.05),

        # Direct representation comparison
        "TC_minus_FR": float(
            best_tc_accuracy
            - best_fr_accuracy
        ),
    }