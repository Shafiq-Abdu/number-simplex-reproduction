# src/paired_neuron_analysis_fast.py

from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from .firing_rate_tuning import analyze_neuron
from .number_decoding_analysis import (
    make_firing_rate_features,
    make_temporal_features,
)


# ============================================================
# CONSTANTS
# ============================================================

BIN_SIZES = [60, 75, 90, 100, 150, 180, 225, 300, 450, 900]
GAMMAS = [0.2, 0.5, 0.8]


# ============================================================
# LOW-LEVEL CV FUNCTION
# ============================================================

def _cv_accuracy_from_splits(X, y, splits, gamma):
    """
    Cross-validated LDA accuracy using precomputed CV splits.

    This avoids rebuilding StratifiedKFold for every pair,
    gamma, bin size, and permutation.
    """

    correct = 0
    total = 0

    for train_idx, test_idx in splits:

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Remove features with zero within-class variance
        keep = np.ones(X_train.shape[1], dtype=bool)

        for j in range(X_train.shape[1]):

            within_class_variances = []

            for cls in np.unique(y_train):
                values = X_train[y_train == cls, j]

                if len(values) > 1:
                    within_class_variances.append(
                        np.var(values, ddof=1)
                    )

            if (
                len(within_class_variances) == 0
                or np.all(
                    np.asarray(within_class_variances) == 0
                )
            ):
                keep[j] = False

        X_train = X_train[:, keep]
        X_test = X_test[:, keep]

        # Same fallback convention used in our reconstruction
        if X_train.shape[1] == 0:
            prediction = np.full(
                len(test_idx),
                np.min(y_train)
            )

        else:
            model = LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage=gamma,
                priors=np.ones(len(np.unique(y_train)))
                / len(np.unique(y_train)),
            )

            model.fit(X_train, y_train)
            prediction = model.predict(X_test)

        correct += np.sum(prediction == y_test)
        total += len(y_test)

    return correct / total


# ============================================================
# PREPARE ONE SUBJECT
# ============================================================

def prepare_subject_pair_cache(
    subject,
    neurons,
    root,
    random_state=42,
    permutation_seed=12345,
    n_permutations=200,
):
    """
    Load and preprocess every neuron ONCE.

    Cached:
        firing-rate features
        temporal features for every bin size
        labels
        CV splits
        shuffled labels

    Returns
    -------
    cache : dict
    """

    neurons = sorted(int(n) for n in neurons)

    if len(neurons) < 2:
        raise ValueError(
            "At least two neurons are required."
        )

    print(
        f"Preparing {subject}: "
        f"{len(neurons)} neurons..."
    )

    neuron_cache = {}

    reference_numbers = None
    reference_onsets = None
    y = None

    # --------------------------------------------------------
    # Load every neuron only once
    # --------------------------------------------------------

    for k, neuron in enumerate(neurons, start=1):

        result = analyze_neuron(
            subject,
            neuron,
            root=root,
            plot=False,
            verbose=False,
        )

        presentations = result["presentations"]
        spike_times = result["spike_times"]

        numbers = presentations["number"].to_numpy()
        onsets = presentations["onset"].to_numpy()

        # ----------------------------------------------------
        # Check behavioral alignment
        # ----------------------------------------------------

        if reference_numbers is None:
            reference_numbers = numbers.copy()
            reference_onsets = onsets.copy()

        else:
            if not np.array_equal(
                numbers,
                reference_numbers
            ):
                raise ValueError(
                    f"Number labels do not align "
                    f"for neuron {neuron}."
                )

            if not np.allclose(
                onsets,
                reference_onsets
            ):
                raise ValueError(
                    f"Presentation onsets do not align "
                    f"for neuron {neuron}."
                )

        # ----------------------------------------------------
        # Firing-rate representation
        # ----------------------------------------------------

        X_fr, y_neuron = make_firing_rate_features(
            presentations,
            spike_times,
        )

        if y is None:
            y = np.asarray(y_neuron)
        else:
            if not np.array_equal(
                y,
                np.asarray(y_neuron)
            ):
                raise ValueError(
                    f"Labels differ for neuron {neuron}."
                )

        # ----------------------------------------------------
        # Temporal representations
        # ----------------------------------------------------

        temporal = {}

        for bin_size in BIN_SIZES:

            X_temp, y_temp, _ = (
                make_temporal_features(
                    presentations,
                    spike_times,
                    bin_size=bin_size,
                )
            )

            if not np.array_equal(
                y,
                np.asarray(y_temp)
            ):
                raise ValueError(
                    f"Temporal labels differ "
                    f"for neuron {neuron}, "
                    f"bin {bin_size}."
                )

            temporal[bin_size] = np.asarray(
                X_temp,
                dtype=np.float64,
            )

        neuron_cache[neuron] = {
            "FR": np.asarray(
                X_fr,
                dtype=np.float64,
            ),
            "TC": temporal,
        }

        print(
            f"  cached neuron "
            f"{k:>2}/{len(neurons)}: {neuron}"
        )

    # --------------------------------------------------------
    # Number of CV folds
    # --------------------------------------------------------

    _, counts = np.unique(
        y,
        return_counts=True,
    )

    min_class_count = int(counts.min())

    n_splits = min(
        10,
        min_class_count,
    )

    # --------------------------------------------------------
    # Precompute observed CV splits ONCE
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    observed_splits = list(
        cv.split(
            np.zeros(len(y)),
            y,
        )
    )

    # --------------------------------------------------------
    # Generate permutation labels ONCE
    # --------------------------------------------------------

    rng = np.random.default_rng(
        permutation_seed
    )

    shuffled_labels = []

    for _ in range(n_permutations):
        shuffled_labels.append(
            rng.permutation(y)
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # For each shuffled label vector, construct valid
    # stratified CV splits using that shuffled labeling.
    # --------------------------------------------------------

    permutation_splits = []

    for y_perm in shuffled_labels:

        cv_perm = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        permutation_splits.append(
            list(
                cv_perm.split(
                    np.zeros(len(y_perm)),
                    y_perm,
                )
            )
        )

    print()
    print("Subject cache complete.")
    print(
        f"Presentations: {len(y)}"
    )
    print(
        f"Minimum class count: {min_class_count}"
    )
    print(
        f"CV folds: {n_splits}"
    )
    print(
        f"Permutations: {n_permutations}"
    )

    return {
        "subject": subject,
        "neurons": neurons,
        "y": y,
        "n_presentations": len(y),
        "min_class_count": min_class_count,
        "n_splits": n_splits,
        "observed_splits": observed_splits,
        "shuffled_labels": shuffled_labels,
        "permutation_splits": permutation_splits,
        "neurons_data": neuron_cache,
        "n_permutations": n_permutations,
    }


# ============================================================
# ANALYZE ONE PAIR USING CACHE
# ============================================================

def analyze_cached_pair(
    cache,
    neuron_1,
    neuron_2,
):
    """
    Analyze one neuron pair using precomputed subject cache.
    """

    neuron_1 = int(neuron_1)
    neuron_2 = int(neuron_2)

    if neuron_1 == neuron_2:
        raise ValueError(
            "A neuron cannot be paired with itself."
        )

    if neuron_1 > neuron_2:
        neuron_1, neuron_2 = (
            neuron_2,
            neuron_1,
        )

    y = cache["y"]

    splits = cache[
        "observed_splits"
    ]

    d1 = cache["neurons_data"][
        neuron_1
    ]

    d2 = cache["neurons_data"][
        neuron_2
    ]

    # ========================================================
    # FIRING RATE PAIR
    # ========================================================

    X_fr = np.hstack(
        [
            d1["FR"],
            d2["FR"],
        ]
    )

    best_fr_accuracy = -np.inf
    best_fr_gamma = None

    for gamma in GAMMAS:

        accuracy = (
            _cv_accuracy_from_splits(
                X_fr,
                y,
                splits,
                gamma,
            )
        )

        if accuracy > best_fr_accuracy:
            best_fr_accuracy = accuracy
            best_fr_gamma = gamma

    # ========================================================
    # TEMPORAL PAIR GRID SEARCH
    # ========================================================

    best_tc_accuracy = -np.inf
    best_tc_bin = None
    best_tc_gamma = None
    best_X_tc = None

    for bin_size in BIN_SIZES:

        X_tc = np.hstack(
            [
                d1["TC"][bin_size],
                d2["TC"][bin_size],
            ]
        )

        for gamma in GAMMAS:

            accuracy = (
                _cv_accuracy_from_splits(
                    X_tc,
                    y,
                    splits,
                    gamma,
                )
            )

            if accuracy > best_tc_accuracy:

                best_tc_accuracy = accuracy
                best_tc_bin = bin_size
                best_tc_gamma = gamma
                best_X_tc = X_tc

    # ========================================================
    # PERMUTATION TESTS
    #
    # Hyperparameters are frozen at observed optimum.
    # Same shuffled labels are used jointly for both neurons.
    # ========================================================

    n_perm = cache[
        "n_permutations"
    ]

    fr_null = np.empty(
        n_perm,
        dtype=float,
    )

    tc_null = np.empty(
        n_perm,
        dtype=float,
    )

    for p in range(n_perm):

        y_perm = cache[
            "shuffled_labels"
        ][p]

        perm_splits = cache[
            "permutation_splits"
        ][p]

        fr_null[p] = (
            _cv_accuracy_from_splits(
                X_fr,
                y_perm,
                perm_splits,
                best_fr_gamma,
            )
        )

        tc_null[p] = (
            _cv_accuracy_from_splits(
                best_X_tc,
                y_perm,
                perm_splits,
                best_tc_gamma,
            )
        )

    # ========================================================
    # P VALUES
    # ========================================================

    fr_p = (
        1
        + np.sum(
            fr_null >= best_fr_accuracy
        )
    ) / (
        n_perm + 1
    )

    tc_p = (
        1
        + np.sum(
            tc_null >= best_tc_accuracy
        )
    ) / (
        n_perm + 1
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "subject": cache["subject"],
        "neuron_1": neuron_1,
        "neuron_2": neuron_2,

        "n_presentations":
            cache["n_presentations"],

        "min_class_count":
            cache["min_class_count"],

        "n_splits":
            cache["n_splits"],

        # FR
        "FR_accuracy":
            best_fr_accuracy,

        "FR_best_gamma":
            best_fr_gamma,

        "FR_null_mean":
            fr_null.mean(),

        "FR_null_sd":
            fr_null.std(ddof=1),

        "FR_p":
            fr_p,

        "FR_coding":
            fr_p < 0.05,

        # Temporal
        "TC_accuracy":
            best_tc_accuracy,

        "TC_best_bin_ms":
            best_tc_bin,

        "TC_best_gamma":
            best_tc_gamma,

        "TC_null_mean":
            tc_null.mean(),

        "TC_null_sd":
            tc_null.std(ddof=1),

        "TC_p":
            tc_p,

        "TC_coding":
            tc_p < 0.05,

        "TC_minus_FR":
            best_tc_accuracy
            - best_fr_accuracy,
    }


# ============================================================
# RUN ALL PAIRS FOR SUBJECT
# ============================================================

def analyze_all_cached_pairs(
    cache,
    save_path=None,
    save_every=25,
):
    """
    Analyze all unique unordered neuron pairs.

    Optionally saves checkpoint CSV every `save_every` pairs.
    """

    pairs = list(
        combinations(
            cache["neurons"],
            2,
        )
    )

    total_pairs = len(pairs)

    print(
        f"Running {total_pairs} unique pairs..."
    )

    results = []

    for k, (n1, n2) in enumerate(
        pairs,
        start=1,
    ):

        result = analyze_cached_pair(
            cache,
            n1,
            n2,
        )

        results.append(result)

        if (
            k == 1
            or k % 10 == 0
            or k == total_pairs
        ):
            print(
                f"{k:>4}/{total_pairs} | "
                f"{n1:>3}, {n2:>3} | "
                f"FR={100*result['FR_accuracy']:.2f}% | "
                f"TC={100*result['TC_accuracy']:.2f}% | "
                f"p={result['TC_p']:.4f}"
            )

        if (
            save_path is not None
            and (
                k % save_every == 0
                or k == total_pairs
            )
        ):
            pd.DataFrame(
                results
            ).to_csv(
                save_path,
                index=False,
            )

    results = pd.DataFrame(
        results
    )

    print()
    print("Finished.")
    print(
        f"Successful pairs: {len(results)}"
    )

    return results