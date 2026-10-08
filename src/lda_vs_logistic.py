"""
lda_vs_logistic.py

Compare single-neuron temporal number decoding using:

    1. Linear Discriminant Analysis (LDA)
    2. Multinomial Logistic Regression

Current analysis strategy
-------------------------
For LDA:
    - Search bin size x LDA shrinkage gamma
    - Select best combination using real-label CV accuracy
    - Keep best bin size and gamma fixed
    - Run 200 label-shuffle permutations

For Logistic Regression:
    - Search temporal bin size
    - Select best bin size using real-label CV accuracy
    - Keep best bin size fixed
    - Run 200 label-shuffle permutations

IMPORTANT
---------
This intentionally reproduces the CURRENT permutation philosophy.

The bin size / hyperparameters are NOT re-selected separately
inside every permutation.

A stricter nested permutation procedure can be implemented later.
"""


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# ------------------------------------------------------------
# Existing project functions
# ------------------------------------------------------------

from src.firing_rate_tuning import analyze_neuron

from src.number_decoding_analysis import (
    BIN_SIZES_MS,
    GAMMAS,
    N_CLASSES,
    make_temporal_features,
    temporal_grid_search,
    cv_lda_accuracy,
    permutation_test_decoding,
)


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

N_SPLITS = 10
N_SHUFFLES = 200

CV_RANDOM_STATE = 42
SHUFFLE_SEED = 12345


# ============================================================
# REMOVE ZERO-WITHIN-CLASS-VARIANCE FEATURES
# ============================================================

def remove_zero_variance_features(
    X_train,
    X_test,
    y_train,
):
    """
    Remove features having zero within-class variance
    for every class.

    Feature selection is determined using TRAINING DATA ONLY.

    This follows the same convention used in the existing
    LDA decoding pipeline.
    """

    keep = np.ones(
        X_train.shape[1],
        dtype=bool,
    )

    for j in range(X_train.shape[1]):

        class_variances = []

        for c in np.unique(y_train):

            values = X_train[
                y_train == c,
                j,
            ]

            if len(values) > 1:

                class_variances.append(
                    np.var(
                        values,
                        ddof=1,
                    )
                )

        if (
            len(class_variances) == 0
            or
            np.all(
                np.asarray(class_variances) == 0
            )
        ):
            keep[j] = False

    return (
        X_train[:, keep],
        X_test[:, keep],
        keep,
    )


# ============================================================
# LOGISTIC REGRESSION CROSS-VALIDATED ACCURACY
# ============================================================

def cv_logistic_accuracy(
    X,
    y,
    n_splits=N_SPLITS,
    random_state=CV_RANDOM_STATE,
    return_folds=False,
):
    """
    Stratified K-fold logistic-regression decoding.

    Uses the same basic cross-validation structure as the
    existing LDA analysis.

    Final accuracy is pooled held-out accuracy:

        total correct held-out predictions
        ----------------------------------
        total held-out predictions

    Every observation is held out exactly once.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(y)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    all_true = []
    all_pred = []

    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        X_train = X[train_idx].copy()
        X_test = X[test_idx].copy()

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ----------------------------------------------------
        # Same feature-removal convention as LDA
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            keep,
        ) = remove_zero_variance_features(
            X_train,
            X_test,
            y_train,
        )

        # ----------------------------------------------------
        # Degenerate case:
        # no usable neural features remain
        # ----------------------------------------------------

        if X_train.shape[1] == 0:

            classes = np.sort(
                np.unique(y_train)
            )

            # Deterministic fallback.
            pred = np.full(
                len(y_test),
                classes[0],
                dtype=y_train.dtype,
            )

        else:

            # ------------------------------------------------
            # Logistic Regression
            # ------------------------------------------------

            model = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
            )

            model.fit(
                X_train,
                y_train,
            )

            pred = model.predict(
                X_test
            )

        # ----------------------------------------------------
        # Evaluate fold
        # ----------------------------------------------------

        n_correct = int(
            np.sum(
                pred == y_test
            )
        )

        n_test = len(y_test)

        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": n_test,
                "n_correct": n_correct,
                "accuracy":
                    n_correct / n_test,
                "accuracy_percent":
                    100 * n_correct / n_test,
                "n_features":
                    X_train.shape[1],
            }
        )

        all_true.extend(y_test)
        all_pred.extend(pred)

    # --------------------------------------------------------
    # Pool all held-out predictions
    # --------------------------------------------------------

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    total_correct = int(
        np.sum(
            all_true == all_pred
        )
    )

    total_test = len(all_true)

    pooled_accuracy = (
        total_correct / total_test
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    if return_folds:

        return (
            pooled_accuracy,
            fold_df,
            all_true,
            all_pred,
        )

    return pooled_accuracy


# ============================================================
# LOGISTIC REGRESSION BIN-SIZE SEARCH
# ============================================================

def logistic_bin_search(
    presentations,
    spike_times,
    bin_sizes=BIN_SIZES_MS,
    n_splits=N_SPLITS,
    random_state=CV_RANDOM_STATE,
):
    """
    Search temporal bin sizes for logistic regression.

    Unlike LDA, logistic regression does not use the LDA
    shrinkage parameter gamma.

    Best model = highest pooled CV accuracy.

    Tie-break:
        smaller bin size wins.
    """

    rows = []

    for bin_size in bin_sizes:

        X, y, edges = make_temporal_features(
            presentations,
            spike_times,
            bin_size,
        )

        accuracy = cv_logistic_accuracy(
            X,
            y,
            n_splits=n_splits,
            random_state=random_state,
        )

        rows.append(
            {
                "bin_size_ms": bin_size,
                "n_bins": X.shape[1],
                "pooled_cv_accuracy":
                    accuracy,
                "accuracy_percent":
                    100 * accuracy,
            }
        )

    table = pd.DataFrame(
        rows
    )

    table = table.sort_values(
        [
            "pooled_cv_accuracy",
            "bin_size_ms",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    best = table.iloc[0]

    return (
        table,
        best,
    )


# ============================================================
# LOGISTIC REGRESSION PERMUTATION TEST
# ============================================================

def logistic_permutation_test(
    X,
    y,
    observed_accuracy,
    n_shuffles=N_SHUFFLES,
    n_splits=N_SPLITS,
    cv_random_state=CV_RANDOM_STATE,
    shuffle_seed=SHUFFLE_SEED,
    alpha=0.05,
):
    """
    One-sided label-shuffle permutation test for logistic
    regression.

    IMPORTANT:
    X has already been constructed using the best temporal
    bin size selected from the REAL labels.

    That bin size remains FIXED during all permutations.

    Current p-value:

        1 + number(null accuracy >= observed accuracy)
        ----------------------------------------------
                    1 + N_shuffles
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(y)

    rng = np.random.default_rng(
        shuffle_seed
    )

    null_accuracies = np.empty(
        n_shuffles,
        dtype=float,
    )

    for i in range(n_shuffles):

        shuffled_y = rng.permutation(
            y
        )

        null_accuracies[i] = (
            cv_logistic_accuracy(
                X,
                shuffled_y,
                n_splits=n_splits,
                random_state=cv_random_state,
            )
        )

    n_equal_or_better = int(
        np.sum(
            null_accuracies
            >= observed_accuracy
        )
    )

    p_value = (
        1 + n_equal_or_better
    ) / (
        1 + n_shuffles
    )

    return {
        "observed_accuracy":
            observed_accuracy,

        "null_accuracies":
            null_accuracies,

        "null_mean":
            null_accuracies.mean(),

        "null_std":
            null_accuracies.std(ddof=1),

        "n_equal_or_better":
            n_equal_or_better,

        "n_shuffles":
            n_shuffles,

        "p_value":
            p_value,

        "is_coding":
            p_value < alpha,
    }


# ============================================================
# MASTER FUNCTION:
# ONE NEURON, LDA VS LOGISTIC
# ============================================================

def compare_lda_logistic_neuron(
    subject,
    neuron,
    root=None,
    n_splits=N_SPLITS,
    n_shuffles=N_SHUFFLES,
    random_state=CV_RANDOM_STATE,
    shuffle_seed=SHUFFLE_SEED,
    verbose=True,
):
    """
    Complete LDA-vs-logistic comparison for ONE neuron.

    Returns
    -------
    dict containing:

        subject
        neuron
        region

        LDA:
            best bin
            best gamma
            CV accuracy
            permutation p-value
            coding status

        Logistic:
            best bin
            CV accuracy
            permutation p-value
            coding status
    """

    # ========================================================
    # LOAD NEURON
    # ========================================================

    base = analyze_neuron(
        subject,
        neuron,
        root=root,
        plot=False,
        verbose=False,
    )

    presentations = base[
        "presentations"
    ]

    spike_times = base[
        "spike_times"
    ]

    region = base[
        "region"
    ]

    # ========================================================
    # LDA
    # ========================================================

    lda_grid, lda_best = (
        temporal_grid_search(
            presentations,
            spike_times,
            n_splits=n_splits,
            random_state=random_state,
        )
    )

    lda_best_bin = int(
        lda_best[
            "bin_size_ms"
        ]
    )

    lda_best_gamma = float(
        lda_best[
            "gamma"
        ]
    )

    # Rebuild features using best LDA bin size

    X_lda, y_lda, _ = (
        make_temporal_features(
            presentations,
            spike_times,
            lda_best_bin,
        )
    )

    lda_accuracy = (
        cv_lda_accuracy(
            X_lda,
            y_lda,
            gamma=lda_best_gamma,
            n_splits=n_splits,
            random_state=random_state,
        )
    )


    # --------------------------------------------------------
    # LDA permutation test
    #
    # Best bin and best gamma remain fixed.
    # --------------------------------------------------------

    lda_perm = (
        permutation_test_decoding(
            X_lda,
            y_lda,
            observed_accuracy=
                lda_accuracy,
            gamma=
                lda_best_gamma,
            n_permutations=
                n_shuffles,
            n_splits=
                n_splits,
            cv_random_state=
                random_state,
            permutation_seed=
                shuffle_seed,
            alpha=0.05,
            n_jobs=1,
        )
    )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    logistic_grid, logistic_best = (
        logistic_bin_search(
            presentations,
            spike_times,
            n_splits=n_splits,
            random_state=random_state,
        )
    )

    logistic_best_bin = int(
        logistic_best[
            "bin_size_ms"
        ]
    )

    # Rebuild features using best logistic bin size

    X_logistic, y_logistic, _ = (
        make_temporal_features(
            presentations,
            spike_times,
            logistic_best_bin,
        )
    )

    logistic_accuracy = (
        cv_logistic_accuracy(
            X_logistic,
            y_logistic,
            n_splits=n_splits,
            random_state=random_state,
        )
    )

    # --------------------------------------------------------
    # Logistic permutation test
    #
    # Best bin remains fixed.
    # --------------------------------------------------------

    logistic_perm = (
        logistic_permutation_test(
            X_logistic,
            y_logistic,
            observed_accuracy=
                logistic_accuracy,
            n_shuffles=
                n_shuffles,
            n_splits=
                n_splits,
            cv_random_state=
                random_state,
            shuffle_seed=
                shuffle_seed,
            alpha=0.05,
        )
    )

    # ========================================================
    # SUMMARY ROW
    # ========================================================

    summary = {
        "subject":
            subject,

        "neuron":
            neuron,

        "region":
            region,

        # ---------------- LDA ----------------

        "lda_best_bin_ms":
            lda_best_bin,

        "lda_best_gamma":
            lda_best_gamma,

        "lda_accuracy":
            lda_accuracy,

        "lda_accuracy_percent":
            100 * lda_accuracy,

        "lda_null_mean_percent":
            100 * lda_perm[
                "null_mean"
            ],

        "lda_p_value":
            lda_perm[
                "p_value"
            ],

        "lda_coding":
            lda_perm[
                "is_coding"
            ],

        # ------------- Logistic -------------

        "logistic_best_bin_ms":
            logistic_best_bin,

        "logistic_accuracy":
            logistic_accuracy,

        "logistic_accuracy_percent":
            100 * logistic_accuracy,

        "logistic_null_mean_percent":
            100 * logistic_perm[
                "null_mean"
            ],

        "logistic_p_value":
            logistic_perm[
                "p_value"
            ],

        "logistic_coding":
            logistic_perm[
                "is_coding"
            ],
    }

    # ========================================================
    # PRINT
    # ========================================================

    if verbose:

        print(
            "\n============================================"
        )

        print(
            f"{subject} neuron {neuron}"
        )

        print(
            f"Region: {region}"
        )

        print(
            "============================================"
        )

        print(
            "\nLDA"
        )

        print(
            "--------------------------------------------"
        )

        print(
            f"Best bin: "
            f"{lda_best_bin} ms"
        )

        print(
            f"Best gamma: "
            f"{lda_best_gamma}"
        )

        print(
            f"Accuracy: "
            f"{100 * lda_accuracy:.2f}%"
        )

        print(
            f"Permutation null mean: "
            f"{100 * lda_perm['null_mean']:.2f}%"
        )

        print(
            f"Permutation p: "
            f"{lda_perm['p_value']:.4f}"
        )

        print(
            f"Coding: "
            f"{lda_perm['is_coding']}"
        )

        print(
            "\nLOGISTIC REGRESSION"
        )

        print(
            "--------------------------------------------"
        )

        print(
            f"Best bin: "
            f"{logistic_best_bin} ms"
        )

        print(
            f"Accuracy: "
            f"{100 * logistic_accuracy:.2f}%"
        )

        print(
            f"Permutation null mean: "
            f"{100 * logistic_perm['null_mean']:.2f}%"
        )

        print(
            f"Permutation p: "
            f"{logistic_perm['p_value']:.4f}"
        )

        print(
            f"Coding: "
            f"{logistic_perm['is_coding']}"
        )

        print(
            "\nDIFFERENCE"
        )

        print(
            "--------------------------------------------"
        )

        difference = (
            logistic_accuracy
            - lda_accuracy
        )

        print(
            "Logistic - LDA: "
            f"{100 * difference:+.2f} percentage points"
        )

    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return {
        "summary":
            summary,

        "lda_grid":
            lda_grid,

        "logistic_grid":
            logistic_grid,

        "lda_permutation":
            lda_perm,

        "logistic_permutation":
            logistic_perm,

        "X_lda":
            X_lda,

        "X_logistic":
            X_logistic,

        "labels":
            y_lda,
    }
# ============================================================
# SUBJECT-LEVEL ANALYSIS
# ============================================================


def compare_lda_logistic_subject(
    subject,
    root=None,
    output_dir=None,
    n_splits=N_SPLITS,
    n_shuffles=N_SHUFFLES,
    random_state=CV_RANDOM_STATE,
    shuffle_seed=SHUFFLE_SEED,
    mtl_only=True,
):
    """
    Run LDA vs Logistic Regression decoding for every neuron
    in one subject.

    One row is saved per analyzed neuron.

    Parameters
    ----------
    subject : str
        Subject ID, e.g. "YFF".

    root : str or Path, optional
        Repository root.

    output_dir : str or Path, optional
        Directory in which the subject CSV will be saved.
        Default: <root>/tables1

    n_splits : int
        Number of stratified CV folds.

    n_shuffles : int
        Number of label-shuffle permutations.

    random_state : int
        Random seed for CV.

    shuffle_seed : int
        Random seed for permutation testing.

    mtl_only : bool
        If True, analyze only neurons in:
        HPC, EC, AMY, PHC.

    Returns
    -------
    pandas.DataFrame
        One row per analyzed neuron.
    """

    from pathlib import Path
    import time

    # --------------------------------------------------------
    # Locate repository root
    # --------------------------------------------------------

    if root is None:
        root = Path.cwd().parent
    else:
        root = Path(root)

    if output_dir is None:
        output_dir = root / "tables1"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir /
        f"{subject}_lda_vs_logistic.csv"
    )

    # --------------------------------------------------------
    # Use neuron 1 only to determine how many neurons
    # were recorded for this subject.
    # --------------------------------------------------------

    first = analyze_neuron(
        subject,
        1,
        root=root,
        plot=False,
        verbose=False,
    )

    total_neurons = int(
        first["total_neurons"]
    )

    print("=" * 60)
    print(f"Subject: {subject}")
    print(f"Total recorded neurons: {total_neurons}")
    print(f"MTL only: {mtl_only}")
    print(f"Permutations per decoder: {n_shuffles}")
    print("=" * 60)

    results = []

    start_time = time.time()

    # --------------------------------------------------------
    # Loop through every recorded neuron
    # --------------------------------------------------------

    for neuron in range(
        1,
        total_neurons + 1,
    ):

        print(
            f"\n[{subject}] "
            f"Neuron {neuron}/{total_neurons}"
        )

        try:

            # ------------------------------------------------
            # First inspect neuron to determine its region.
            # ------------------------------------------------

            base = analyze_neuron(
                subject,
                neuron,
                root=root,
                plot=False,
                verbose=False,
            )

            region = base["region"]

            # ------------------------------------------------
            # Paper MTL restriction
            # ------------------------------------------------

            if (
                mtl_only
                and region
                not in {"hpc", "ent", "amy", "para-hpc"}
            ):

                print(
                    f"  Skipping region: {region}"
                )

                continue

            print(
                f"  Region: {region}"
            )

            # ------------------------------------------------
            # Run complete decoder comparison
            # ------------------------------------------------

            out = compare_lda_logistic_neuron(
                subject=subject,
                neuron=neuron,
                root=root,
                n_splits=n_splits,
                n_shuffles=n_shuffles,
                random_state=random_state,
                shuffle_seed=shuffle_seed,
                verbose=False,
            )

            row = out["summary"]

            results.append(row)

            # ------------------------------------------------
            # Print concise progress
            # ------------------------------------------------

            print(
                f"  LDA: "
                f"{row['lda_accuracy_percent']:.2f}% "
                f"(p={row['lda_p_value']:.4f})"
            )

            print(
                f"  Logistic: "
                f"{row['logistic_accuracy_percent']:.2f}% "
                f"(p={row['logistic_p_value']:.4f})"
            )

            # ------------------------------------------------
            # CHECKPOINT SAVE
            #
            # Save after every successful neuron.
            # Very useful on the cluster if a job stops.
            # ------------------------------------------------

            pd.DataFrame(
                results
            ).to_csv(
                output_file,
                index=False,
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            # Continue to the next neuron rather than losing
            # the entire subject-level job.

            continue

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"FINISHED: {subject}")
    print("=" * 60)

    print(
        f"Successfully analyzed neurons: "
        f"{len(results_df)}"
    )

    print(
        f"Runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Saved to:\n{output_file}"
    )

    if len(results_df) > 0:

        print("\nMean decoding accuracy:")

        print(
            f"  LDA: "
            f"{results_df['lda_accuracy_percent'].mean():.2f}%"
        )

        print(
            f"  Logistic: "
            f"{results_df['logistic_accuracy_percent'].mean():.2f}%"
        )

        print("\nCoding neurons:")

        print(
            f"  LDA: "
            f"{results_df['lda_coding'].sum()}"
            f"/{len(results_df)}"
        )

        print(
            f"  Logistic: "
            f"{results_df['logistic_coding'].sum()}"
            f"/{len(results_df)}"
        )

    return results_df