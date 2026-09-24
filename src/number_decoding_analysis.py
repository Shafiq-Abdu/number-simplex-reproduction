"""
number_decoding_analysis.py

Single-neuron number-decoding analysis for the arithmetic task.

Pipeline
--------
1. Load neuron using firing_rate_tuning.analyze_neuron()
2. Construct temporal spike-count features
3. Grid search:
       bin size x shrinkage Gamma
4. Select best model by pooled held-out CV accuracy
5. Fit optimized LDA
6. Figure G:
       first two temporal-component weights
7. Figure H:
       projection onto first two temporal components
       + class centroids
       + SEM-normalized covariance ellipses
8. Firing-rate decoding
9. 200-label-shuffle permutation test:
       temporal representation
       firing-rate representation
10. Plot both null distributions
11. Return all numerical results and figures

IMPORTANT
---------
The Figure G/H implementation uses scikit-learn's canonical LDA
projection. This reproduces the analysis conceptually, but the paper
does not provide enough implementation detail to guarantee that these
coordinates are numerically identical to the authors' MATLAB plotting
implementation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from src.firing_rate_tuning import analyze_neuron


# ============================================================
# PAPER PARAMETERS
# ============================================================

ARITH_START_MS = 50
ARITH_END_MS = 950
ARITH_DURATION_MS = 900

BIN_SIZES_MS = [
    60,
    75,
    90,
    100,
    150,
    180,
    225,
    300,
    450,
    900,
]

GAMMAS = [
    0.2,
    0.5,
    0.8,
]

N_CLASSES = 9
CHANCE_LEVEL = 1 / N_CLASSES


# ============================================================
# TEMPORAL FEATURE MATRIX
# ============================================================

def make_temporal_features(
    presentations,
    spike_times,
    bin_size,
    window_start=ARITH_START_MS,
    window_end=ARITH_END_MS,
):
    """
    Convert each operand presentation into binned spike counts.

    Example
    -------
    150-ms bins over 50-950 ms:

        [50,200)
        [200,350)
        ...
        [800,950)

    giving six temporal features per presentation.
    """

    bin_edges = np.arange(
        window_start,
        window_end + bin_size,
        bin_size,
        dtype=float,
    )

    # protect against floating/integer edge issues
    bin_edges[-1] = window_end

    X = []
    y = []

    for _, row in presentations.iterrows():

        onset = float(row["onset"])
        number = int(row["number"])

        relative_spikes = spike_times - onset

        counts, _ = np.histogram(
            relative_spikes,
            bins=bin_edges,
        )

        X.append(counts)
        y.append(number)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        bin_edges,
    )


# ============================================================
# FIRING-RATE FEATURE MATRIX
# ============================================================

def make_firing_rate_features(
    presentations,
    spike_times,
    window_start=ARITH_START_MS,
    window_end=ARITH_END_MS,
):
    """
    One whole-window spike-count feature per presentation.

    X shape:
        n_presentations x 1

    Dividing by 0.9 s would convert this to Hz, but that constant
    scaling does not change LDA classification.
    """

    X = []
    y = []

    for _, row in presentations.iterrows():

        onset = float(row["onset"])
        number = int(row["number"])

        relative_spikes = spike_times - onset

        count = np.sum(
            (relative_spikes >= window_start)
            &
            (relative_spikes < window_end)
        )

        X.append([count])
        y.append(number)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
    )


# ============================================================
# REMOVE ZERO-WITHIN-CLASS-VARIANCE FEATURES
# ============================================================

def _remove_zero_variance_features(
    X_train,
    X_test,
    y_train,
):
    """
    Determine feature removal using TRAINING DATA ONLY.
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
                j
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
# STRATIFIED CROSS-VALIDATED LDA
# ============================================================
def cv_lda_accuracy(
    X,
    y,
    gamma,
    n_splits=10,
    random_state=42,
    return_folds=False,
):
    """
    Stratified K-fold LDA.

    Final accuracy is pooled held-out accuracy:

        total correct held-out predictions
        ----------------------------------
        total held-out predictions

    Every observation is held out exactly once.

    If zero-within-class-variance removal leaves no usable
    features in a training fold, use a deterministic
    prior-only prediction. Because the classifier uses
    uniform class priors, all classes are tied, so the
    smallest class label is chosen deterministically.
    """

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
        # Remove zero-within-class-variance features
        # using TRAINING DATA ONLY
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            keep,
        ) = _remove_zero_variance_features(
            X_train,
            X_test,
            y_train,
        )

        # ----------------------------------------------------
        # Degenerate case:
        # no usable features remain
        # ----------------------------------------------------

        if X_train.shape[1] == 0:

            # No neural feature remains from which LDA
            # can learn.
            #
            # With uniform class priors, all classes are tied.
            # Use the smallest class label deterministically.

            classes = np.sort(
                np.unique(y_train)
            )

            pred = np.full(
                len(y_test),
                classes[0],
                dtype=y_train.dtype,
            )

        # ----------------------------------------------------
        # Normal LDA case
        # ----------------------------------------------------

        else:

            lda = LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage=gamma,
                priors=np.ones(N_CLASSES) / N_CLASSES,
            )

            lda.fit(
                X_train,
                y_train,
            )

            pred = lda.predict(
                X_test
            )

        # ----------------------------------------------------
        # Evaluate held-out predictions
        # ----------------------------------------------------

        n_correct = int(
            np.sum(pred == y_test)
        )

        n_test = len(y_test)

        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": n_test,
                "n_correct": n_correct,
                "accuracy": n_correct / n_test,
                "accuracy_percent":
                    100 * n_correct / n_test,
                "n_features":
                    X_train.shape[1],
            }
        )

        all_true.extend(y_test)
        all_pred.extend(pred)

    # --------------------------------------------------------
    # Pool predictions across all held-out folds
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

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    if return_folds:

        return (
            pooled_accuracy,
            fold_df,
            all_true,
            all_pred,
        )

    return pooled_accuracy
# ============================================================
# TEMPORAL GRID SEARCH
# ============================================================

def temporal_grid_search(
    presentations,
    spike_times,
    bin_sizes=BIN_SIZES_MS,
    gammas=GAMMAS,
    n_splits=10,
    random_state=42,
):
    """
    Evaluate every bin-size x Gamma combination.

    Best model = highest pooled CV accuracy.
    """

    rows = []

    for bin_size in bin_sizes:

        X, y, edges = make_temporal_features(
            presentations,
            spike_times,
            bin_size,
        )

        for gamma in gammas:

            accuracy = cv_lda_accuracy(
                X,
                y,
                gamma=gamma,
                n_splits=n_splits,
                random_state=random_state,
            )

            rows.append(
                {
                    "bin_size_ms": bin_size,
                    "n_bins": X.shape[1],
                    "gamma": gamma,
                    "pooled_cv_accuracy":
                        accuracy,
                    "accuracy_percent":
                        100 * accuracy,
                }
            )

    table = pd.DataFrame(rows)

    table = table.sort_values(
        [
            "pooled_cv_accuracy",
            "bin_size_ms",
            "gamma",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    best = table.iloc[0]

    return table, best


# ============================================================
# FIT TEMPORAL COMPONENTS
# ============================================================

def fit_temporal_components(
    X,
    y,
    gamma,
):
    """
    Fit canonical LDA projection to the complete dataset.

    Used for Figure G/H visualization.
    """

    lda = LinearDiscriminantAnalysis(
        solver="eigen",
        shrinkage=gamma,
        priors=np.ones(N_CLASSES) / N_CLASSES,
        n_components=2,
    )

    lda.fit(
        X,
        y,
    )

    Z = lda.transform(
        X
    )

    LD1 = lda.scalings_[:, 0]
    LD2 = lda.scalings_[:, 1]

    return (
        lda,
        Z,
        LD1,
        LD2,
    )


# ============================================================
# FIGURE G
# ============================================================

def plot_figure_g(
    LD1,
    LD2,
    bin_edges,
    subject,
    neuron,
    gamma,
):
    """
    Plot first two temporal-component weights.
    """

    centers_ms = (
        bin_edges[:-1]
        +
        bin_edges[1:]
    ) / 2

    centers_s = (
        centers_ms / 1000
    )

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.plot(
        centers_s,
        LD1,
        marker="o",
        linewidth=2,
        label="LDA1",
    )

    ax.plot(
        centers_s,
        LD2,
        marker="o",
        linewidth=2,
        label="LDA2",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.set_xlabel(
        "Time from number onset (s)"
    )

    ax.set_ylabel(
        "Temporal-component weight"
    )

    ax.set_title(
        f"{subject} neuron {neuron} — Temporal components\n"
        f"Gamma = {gamma}"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.legend()

    fig.tight_layout()

    return fig, ax


# ============================================================
# SEM COVARIANCE ELLIPSE
# ============================================================

def _add_sem_covariance_ellipse(
    points,
    ax,
    color,
    n_std=1.0,
):
    """
    Ellipse based on covariance of the class mean.

        Sigma_mean = Sigma_trials / N

    Equivalent to scaling SD by sqrt(N).
    """

    n = len(points)

    mean = points.mean(
        axis=0
    )

    cov = np.cov(
        points,
        rowvar=False,
    )

    cov_mean = cov / n

    eigvals, eigvecs = np.linalg.eigh(
        cov_mean
    )

    order = np.argsort(
        eigvals
    )[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.maximum(
        eigvals,
        0,
    )

    angle = np.degrees(
        np.arctan2(
            eigvecs[1, 0],
            eigvecs[0, 0],
        )
    )

    width = (
        2
        * n_std
        * np.sqrt(eigvals[0])
    )

    height = (
        2
        * n_std
        * np.sqrt(eigvals[1])
    )

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor=color,
        alpha=0.18,
        linewidth=2,
    )

    ax.add_patch(
        ellipse
    )

    return mean


# ============================================================
# FIGURE H
# ============================================================

def plot_figure_h(
    Z,
    y,
    subject,
    neuron,
    bin_size,
    gamma,
):
    """
    Plot class means and SEM-normalized covariance ellipses.
    """

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    colors = plt.cm.hsv(
        np.linspace(
            0,
            0.9,
            N_CLASSES,
        )
    )

    centroids = {}

    for i, number in enumerate(
        range(1, 10)
    ):

        points = Z[
            y == number
        ]

        mean = (
            _add_sem_covariance_ellipse(
                points,
                ax,
                colors[i],
            )
        )

        centroids[number] = mean

        ax.scatter(
            mean[0],
            mean[1],
            marker="s",
            s=90,
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

        ax.text(
            mean[0],
            mean[1] + 0.04,
            str(number),
            color=colors[i],
            fontsize=12,
            fontweight="bold",
            ha="center",
        )

    ax.set_xlabel(
        "Temporal component 1"
    )

    ax.set_ylabel(
        "Temporal component 2"
    )

    ax.set_title(
        f"{subject} neuron {neuron} — Reduced temporal representation\n"
        f"{bin_size}-ms bins, Gamma = {gamma}"
    )

    fig.tight_layout()

    return (
        fig,
        ax,
        centroids,
    )


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(
    X,
    y,
    observed_accuracy,
    gamma,
    n_shuffles=200,
    n_splits=10,
    cv_random_state=42,
    shuffle_seed=12345,
):
    """
    One-sided label-shuffle permutation test.

    p =
        (1 + number(shuffle_accuracy >= observed_accuracy))
        ----------------------------------------------------
                         (1 + N_shuffles)
    """

    rng = np.random.default_rng(
        shuffle_seed
    )

    null = np.empty(
        n_shuffles,
        dtype=float,
    )

    for i in range(
        n_shuffles
    ):

        shuffled_y = rng.permutation(
            y
        )

        null[i] = cv_lda_accuracy(
            X,
            shuffled_y,
            gamma=gamma,
            n_splits=n_splits,
            random_state=cv_random_state,
        )

    n_ge = int(
        np.sum(
            null >= observed_accuracy
        )
    )

    p_value = (
        1 + n_ge
    ) / (
        1 + n_shuffles
    )

    return {
        "observed_accuracy":
            observed_accuracy,

        "null_accuracies":
            null,

        "mean_null_accuracy":
            null.mean(),

        "n_ge_observed":
            n_ge,

        "n_shuffles":
            n_shuffles,

        "p_value":
            p_value,

        "significant":
            p_value < 0.05,
    }


# ============================================================
# NULL DISTRIBUTION PLOT
# ============================================================

def plot_null_distribution(
    result,
    title,
):
    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    null = (
        100
        * result[
            "null_accuracies"
        ]
    )

    observed = (
        100
        * result[
            "observed_accuracy"
        ]
    )

    ax.hist(
        null,
        bins=20,
        edgecolor="black",
        alpha=0.75,
    )

    ax.axvline(
        observed,
        linewidth=2.5,
        label=(
            f"Observed = "
            f"{observed:.2f}%"
        ),
    )

    ax.axvline(
        100 * CHANCE_LEVEL,
        linestyle="--",
        linewidth=1.5,
        label="Chance = 11.11%",
    )

    ax.set_xlabel(
        "Cross-validated decoding accuracy (%)"
    )

    ax.set_ylabel(
        "Number of label shuffles"
    )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    return fig, ax


# ============================================================
# MASTER FUNCTION
# ============================================================

def analyze_number_decoding(
    subject,
    neuron,
    root=None,
    n_splits=10,
    n_shuffles=200,
    random_state=42,
    shuffle_seed=12345,
    save=False,
    save_dir="figures",
    show=True,
    verbose=True,
):
    """
    Complete single-neuron number-decoding analysis.

    Example
    -------
    out = analyze_number_decoding(
        "YFU",
        43
    )
    """

    # --------------------------------------------------------
    # LOAD EXISTING NEURON DATA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GRID SEARCH
    # --------------------------------------------------------

    grid_table, best = (
        temporal_grid_search(
            presentations,
            spike_times,
            n_splits=n_splits,
            random_state=random_state,
        )
    )

    best_bin = int(
        best["bin_size_ms"]
    )

    best_gamma = float(
        best["gamma"]
    )

    best_temporal_accuracy = float(
        best["pooled_cv_accuracy"]
    )

    # --------------------------------------------------------
    # REBUILD OPTIMAL TEMPORAL FEATURES
    # --------------------------------------------------------

    X_temporal, y, bin_edges = (
        make_temporal_features(
            presentations,
            spike_times,
            best_bin,
        )
    )

    (
        temporal_accuracy,
        temporal_fold_table,
        temporal_true,
        temporal_pred,
    ) = cv_lda_accuracy(
        X_temporal,
        y,
        gamma=best_gamma,
        n_splits=n_splits,
        random_state=random_state,
        return_folds=True,
    )

    # --------------------------------------------------------
    # TEMPORAL COMPONENTS
    # --------------------------------------------------------

    (
        lda_components,
        Z,
        LD1,
        LD2,
    ) = fit_temporal_components(
        X_temporal,
        y,
        best_gamma,
    )

    # --------------------------------------------------------
    # FIGURE G
    # --------------------------------------------------------

    fig_g, ax_g = (
        plot_figure_g(
            LD1,
            LD2,
            bin_edges,
            subject,
            neuron,
            best_gamma,
        )
    )

    # --------------------------------------------------------
    # FIGURE H
    # --------------------------------------------------------

    (
        fig_h,
        ax_h,
        centroids,
    ) = plot_figure_h(
        Z,
        y,
        subject,
        neuron,
        best_bin,
        best_gamma,
    )

    # --------------------------------------------------------
    # FIRING-RATE DECODING
    # --------------------------------------------------------

    X_fr, y_fr = (
        make_firing_rate_features(
            presentations,
            spike_times,
        )
    )

    (
        fr_accuracy,
        fr_fold_table,
        fr_true,
        fr_pred,
    ) = cv_lda_accuracy(
        X_fr,
        y_fr,
        gamma=0.2,
        n_splits=n_splits,
        random_state=random_state,
        return_folds=True,
    )

    # --------------------------------------------------------
    # TEMPORAL PERMUTATION
    # --------------------------------------------------------

    temporal_perm = permutation_test(
        X_temporal,
        y,
        temporal_accuracy,
        gamma=best_gamma,
        n_shuffles=n_shuffles,
        n_splits=n_splits,
        cv_random_state=random_state,
        shuffle_seed=shuffle_seed,
    )

    # --------------------------------------------------------
    # FIRING-RATE PERMUTATION
    # --------------------------------------------------------

    fr_perm = permutation_test(
        X_fr,
        y_fr,
        fr_accuracy,
        gamma=0.2,
        n_shuffles=n_shuffles,
        n_splits=n_splits,
        cv_random_state=random_state,
        shuffle_seed=shuffle_seed + 1,
    )

    # --------------------------------------------------------
    # NULL DISTRIBUTION FIGURES
    # --------------------------------------------------------

    fig_temporal_null, _ = (
        plot_null_distribution(
            temporal_perm,
            (
                f"{subject} neuron {neuron} — "
                "Temporal decoding permutation test"
            ),
        )
    )

    fig_fr_null, _ = (
        plot_null_distribution(
            fr_perm,
            (
                f"{subject} neuron {neuron} — "
                "Firing-rate decoding permutation test"
            ),
        )
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "subject": subject,
                "neuron": neuron,
                "region": region,
                "representation":
                    "Firing rate",
                "bin_size_ms":
                    900,
                "n_features":
                    1,
                "gamma":
                    np.nan,
                "cv_accuracy":
                    fr_accuracy,
                "cv_accuracy_percent":
                    100 * fr_accuracy,
                "mean_shuffle_percent":
                    100
                    * fr_perm[
                        "mean_null_accuracy"
                    ],
                "p_value":
                    fr_perm["p_value"],
                "significant":
                    fr_perm["significant"],
            },

            {
                "subject": subject,
                "neuron": neuron,
                "region": region,
                "representation":
                    "Temporal",
                "bin_size_ms":
                    best_bin,
                "n_features":
                    X_temporal.shape[1],
                "gamma":
                    best_gamma,
                "cv_accuracy":
                    temporal_accuracy,
                "cv_accuracy_percent":
                    100 * temporal_accuracy,
                "mean_shuffle_percent":
                    100
                    * temporal_perm[
                        "mean_null_accuracy"
                    ],
                "p_value":
                    temporal_perm[
                        "p_value"
                    ],
                "significant":
                    temporal_perm[
                        "significant"
                    ],
            },
        ]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    if save:

        save_dir = Path(
            save_dir
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prefix = (
            f"{subject}_neuron_{neuron}"
        )

        fig_g.savefig(
            save_dir
            / f"{prefix}_figure_G.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig_h.savefig(
            save_dir
            / f"{prefix}_figure_H.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig_temporal_null.savefig(
            save_dir
            / f"{prefix}_temporal_null.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig_fr_null.savefig(
            save_dir
            / f"{prefix}_firing_rate_null.png",
            dpi=300,
            bbox_inches="tight",
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    if verbose:

        print(
            "\n=========================================="
        )

        print(
            f"{subject} neuron {neuron}"
        )

        print(
            f"Region: {region}"
        )

        print(
            "=========================================="
        )

        print(
            "\nBEST TEMPORAL MODEL"
        )

        print(
            "-------------------"
        )

        print(
            f"Bin size: "
            f"{best_bin} ms"
        )

        print(
            f"Number of bins: "
            f"{X_temporal.shape[1]}"
        )

        print(
            f"Gamma: "
            f"{best_gamma}"
        )

        print(
            f"Pooled CV accuracy: "
            f"{100 * temporal_accuracy:.2f}%"
        )

        print(
            "\nFIRING-RATE MODEL"
        )

        print(
            "-------------------"
        )

        print(
            f"Pooled CV accuracy: "
            f"{100 * fr_accuracy:.2f}%"
        )

        print(
            "\nPERMUTATION TESTS"
        )

        print(
            "-------------------"
        )

        print(
            "Temporal:"
        )

        print(
            f"  mean shuffle = "
            f"{100 * temporal_perm['mean_null_accuracy']:.2f}%"
        )

        print(
            f"  >= observed = "
            f"{temporal_perm['n_ge_observed']}"
            f"/{n_shuffles}"
        )

        print(
            f"  p = "
            f"{temporal_perm['p_value']:.4f}"
        )

        print(
            f"  significant = "
            f"{temporal_perm['significant']}"
        )

        print(
            "\nFiring rate:"
        )

        print(
            f"  mean shuffle = "
            f"{100 * fr_perm['mean_null_accuracy']:.2f}%"
        )

        print(
            f"  >= observed = "
            f"{fr_perm['n_ge_observed']}"
            f"/{n_shuffles}"
        )

        print(
            f"  p = "
            f"{fr_perm['p_value']:.4f}"
        )

        print(
            f"  significant = "
            f"{fr_perm['significant']}"
        )

        print(
            "\nFINAL CLASSIFICATION"
        )

        print(
            "-------------------"
        )

        if temporal_perm[
            "significant"
        ]:

            print(
                "Significant temporal "
                "number-decoding neuron."
            )

        else:

            print(
                "Not significant under "
                "temporal decoding."
            )

    # --------------------------------------------------------
    # SHOW / CLOSE
    # --------------------------------------------------------

    if show:

        plt.show()

    else:

        plt.close(
            fig_g
        )

        plt.close(
            fig_h
        )

        plt.close(
            fig_temporal_null
        )

        plt.close(
            fig_fr_null
        )

    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {
        "subject":
            subject,

        "neuron":
            neuron,

        "region":
            region,

        "grid_search":
            grid_table,

        "best_bin_ms":
            best_bin,

        "best_gamma":
            best_gamma,

        "temporal_accuracy":
            temporal_accuracy,

        "temporal_fold_table":
            temporal_fold_table,

        "firing_rate_accuracy":
            fr_accuracy,

        "firing_rate_fold_table":
            fr_fold_table,

        "temporal_permutation":
            temporal_perm,

        "firing_rate_permutation":
            fr_perm,

        "summary":
            summary,

        "X_temporal":
            X_temporal,

        "X_firing_rate":
            X_fr,

        "labels":
            y,

        "bin_edges":
            bin_edges,

        "LD1":
            LD1,

        "LD2":
            LD2,

        "projection":
            Z,

        "centroids":
            centroids,

        "lda":
            lda_components,

        "figure_G":
            fig_g,

        "figure_H":
            fig_h,

        "figure_temporal_null":
            fig_temporal_null,

        "figure_firing_rate_null":
            fig_fr_null,
    }


def permutation_test_decoding(
    X,
    y,
    observed_accuracy,
    gamma,
    n_permutations=200,
    n_splits=10,
    cv_random_state=42,
    permutation_seed=12345,
    alpha=0.05,
):
    """
    Permutation test for single-neuron numeral decoding.

    Parameters
    ----------
    X : ndarray
        Neural features.

    y : ndarray
        Numeral labels.

    observed_accuracy : float
        Cross-validated decoding accuracy obtained using
        the real numeral labels.

    gamma : float
        LDA shrinkage parameter.

    n_permutations : int
        Number of label permutations.

    n_splits : int
        Number of cross-validation folds.

    cv_random_state : int
        Random seed used by cross-validation.

    permutation_seed : int
        Seed controlling label permutations.

    alpha : float
        Significance level used to classify the neuron.

    Returns
    -------
    dict
        Dictionary containing observed accuracy,
        permutation distribution, p-value, and coding status.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(
        permutation_seed
    )

    null_accuracies = np.empty(
        n_permutations,
        dtype=float,
    )

    # --------------------------------------------------------
    # Shuffle numeral labels
    # --------------------------------------------------------

    for p in range(n_permutations):

        y_perm = rng.permutation(
            y
        )

        null_accuracies[p] = cv_lda_accuracy(
            X,
            y_perm,
            gamma=gamma,
            n_splits=n_splits,
            random_state=cv_random_state,
        )

    # --------------------------------------------------------
    # Permutation p-value
    # --------------------------------------------------------

    n_equal_or_better = int(
        np.sum(
            null_accuracies
            >= observed_accuracy
        )
    )

    p_value = (
        1 + n_equal_or_better
    ) / (
        n_permutations + 1
    )

    is_coding = (
        p_value < alpha
    )

    return {
        "observed_accuracy":
            observed_accuracy,

        "null_mean":
            null_accuracies.mean(),

        "null_std":
            null_accuracies.std(ddof=1),

        "n_equal_or_better":
            n_equal_or_better,

        "p_value":
            p_value,

        "is_coding":
            is_coding,

        "null_accuracies":
            null_accuracies,
    }

# from src.number_decoding_analysis import analyze_number_decoding

# out = analyze_number_decoding(
#     "YFU",
#     43,
#     n_shuffles=200,
#     save=True,
#     show=True
# )

# display(
#     out["grid_search"]
# )

# display(
#     out["summary"]
# )