from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from .firing_rate_tuning import analyze_neuron
from .number_decoding_analysis import make_temporal_features
from .poisson_naive_bayes import (
    fit_poisson_nb,
    predict_poisson_nb,
    predict_proba_poisson_nb,
)


CLASSES = np.arange(1, 10)

MTL_REGIONS = {
    "hpc",
    "ent",
    "amy",
    "para-hpc",
}


# ============================================================
# BUILD POPULATION MATRIX
# ============================================================

def build_temporal_population(
    subject,
    root,
    bin_size=100,
):
    root = Path(root)

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

    blocks = []
    feature_rows = []

    reference = None
    y_reference = None
    bin_edges_reference = None

    included_neurons = []

    for neuron in range(
        1,
        total_neurons + 1,
    ):

        base = analyze_neuron(
            subject,
            neuron,
            root=root,
            plot=False,
            verbose=False,
        )

        region = base["region"]

        if region not in MTL_REGIONS:
            continue

        presentations = (
            base["presentations"]
            .reset_index(drop=True)
        )

        X_n, y_n, edges = (
            make_temporal_features(
                presentations,
                base["spike_times"],
                bin_size=bin_size,
                window_start=50,
                window_end=950,
            )
        )

        if reference is None:

            reference = presentations[
                [
                    "trial",
                    "number",
                    "onset",
                    "operand",
                ]
            ].copy()

            y_reference = y_n.copy()
            bin_edges_reference = edges.copy()

        else:

            assert np.array_equal(
                presentations["trial"].to_numpy(),
                reference["trial"].to_numpy(),
            )

            assert np.array_equal(
                presentations["number"].to_numpy(),
                reference["number"].to_numpy(),
            )

            assert np.array_equal(
                presentations["operand"].to_numpy(),
                reference["operand"].to_numpy(),
            )

            assert np.allclose(
                presentations["onset"].to_numpy(),
                reference["onset"].to_numpy(),
            )

            assert np.array_equal(
                y_n,
                y_reference,
            )

            assert np.allclose(
                edges,
                bin_edges_reference,
            )

        blocks.append(X_n)

        included_neurons.append(
            {
                "neuron": neuron,
                "region": region,
            }
        )

        for b in range(
            len(edges) - 1
        ):

            feature_rows.append(
                {
                    "feature_index":
                        len(feature_rows),

                    "neuron":
                        neuron,

                    "region":
                        region,

                    "bin_start_ms":
                        int(edges[b]),

                    "bin_end_ms":
                        int(edges[b + 1]),

                    "feature_name":
                        (
                            f"{region}.{neuron}_"
                            f"{int(edges[b])}-"
                            f"{int(edges[b + 1])}ms"
                        ),
                }
            )

    if len(blocks) == 0:
        raise RuntimeError(
            f"No MTL neurons found for {subject}"
        )

    X = np.concatenate(
        blocks,
        axis=1,
    )

    y = y_reference.astype(int)

    groups = (
        reference["trial"]
        .to_numpy()
    )

    operands = (
        reference["operand"]
        .to_numpy()
    )

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "operands": operands,
        "presentations": reference,
        "feature_map":
            pd.DataFrame(feature_rows),
        "neurons":
            pd.DataFrame(included_neurons),
        "bin_edges":
            bin_edges_reference,
    }


# ============================================================
# DESCRIPTIVE LAMBDA + FANO
# ============================================================

def compute_lambda_fano(X, y):

    n_features = X.shape[1]

    lambda_matrix = np.full(
        (9, n_features),
        np.nan,
    )

    fano_matrix = np.full(
        (9, n_features),
        np.nan,
    )

    for i, k in enumerate(CLASSES):

        X_k = X[y == k]

        means = X_k.mean(axis=0)

        variances = X_k.var(
            axis=0,
            ddof=1,
        )

        lambda_matrix[i] = means

        fano_matrix[i] = np.divide(
            variances,
            means,
            out=np.full(
                n_features,
                np.nan,
                dtype=float,
            ),
            where=means > 0,
        )

    return (
        lambda_matrix,
        fano_matrix,
    )


# ============================================================
# FIXED GROUPED CV
# ============================================================

def make_fixed_folds(
    X,
    y,
    groups,
    n_splits=5,
    random_state=42,
):

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    return list(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )


def evaluate_fixed_folds(
    X,
    y,
    folds,
):

    predictions = np.empty(
        len(y),
        dtype=int,
    )

    posterior = np.zeros(
        (len(y), 9),
        dtype=float,
    )

    for train_idx, test_idx in folds:

        model = fit_poisson_nb(
            X[train_idx],
            y[train_idx],
            classes=CLASSES,
            uniform_prior=True,
        )

        predictions[test_idx] = (
            predict_poisson_nb(
                model,
                X[test_idx],
            )
        )

        posterior[test_idx] = (
            predict_proba_poisson_nb(
                model,
                X[test_idx],
            )
        )

    accuracy = np.mean(
        predictions == y
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions,
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_accuracy,
        "predictions": predictions,
        "posterior": posterior,
    }


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test_population(
    X,
    y,
    groups,
    operands,
    observed_accuracy,
    n_permutations=200,
    n_splits=5,
    cv_random_state=42,
    random_state=12345,
):

    rng = np.random.default_rng(
        random_state
    )

    null = np.zeros(
        n_permutations,
        dtype=float,
    )

    for p in range(n_permutations):

        # ----------------------------------------------------
        # Shuffle labels within operand position
        # ----------------------------------------------------

        y_perm = y.copy()

        for operand in np.unique(operands):

            idx = np.where(
                operands == operand
            )[0]

            y_perm[idx] = rng.permutation(
                y_perm[idx]
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Create new stratified grouped folds
        # using the SHUFFLED labels
        # ----------------------------------------------------

        perm_splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=cv_random_state,
        )

        perm_folds = list(
            perm_splitter.split(
                X,
                y_perm,
                groups=groups,
            )
        )

        # ----------------------------------------------------
        # Evaluate shuffled labels
        # ----------------------------------------------------

        out = evaluate_fixed_folds(
            X,
            y_perm,
            perm_folds,
        )

        null[p] = out["accuracy"]

    # --------------------------------------------------------
    # One-sided permutation p-value
    # --------------------------------------------------------

    p_value = (
        1
        + np.sum(
            null >= observed_accuracy
        )
    ) / (
        n_permutations + 1
    )

    return {
        "null_accuracies":
            null,

        "null_mean":
            float(
                null.mean()
            ),

        "null_std":
            float(
                null.std(ddof=1)
            ),

        "p_value":
            float(
                p_value
            ),
    }


# ============================================================
# COMPLETE SUBJECT ANALYSIS
# ============================================================

def analyze_population_subject(
    subject,
    root,
    bin_size=100,
    n_splits=5,
    n_permutations=200,
    cv_random_state=42,
    permutation_seed=12345,
):

    root = Path(root)

    output_dir = (
        root
        / "tables1"
        / "poisson_population_100ms"
        / subject
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(f"SUBJECT: {subject}")
    print("=" * 70)

    data = build_temporal_population(
        subject,
        root=root,
        bin_size=bin_size,
    )

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    operands = data["operands"]

    feature_map = data["feature_map"]
    presentations = data["presentations"]

    n_neurons = len(
        data["neurons"]
    )

    print(
        f"Presentations: {len(y)}"
    )

    print(
        f"MTL neurons: {n_neurons}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Population matrix: {X.shape}"
    )

    # --------------------------------------------------------
    # Full-data descriptive lambda/Fano matrices
    # --------------------------------------------------------

    (
        lambda_matrix,
        fano_matrix,
    ) = compute_lambda_fano(
        X,
        y,
    )

    valid_fano = fano_matrix[
        np.isfinite(fano_matrix)
    ]

    # --------------------------------------------------------
    # Grouped CV
    # --------------------------------------------------------

    folds = make_fixed_folds(
        X,
        y,
        groups,
        n_splits=n_splits,
        random_state=cv_random_state,
    )

    cv = evaluate_fixed_folds(
        X,
        y,
        folds,
    )

    predictions = cv[
        "predictions"
    ]

    posterior = cv[
        "posterior"
    ]

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y,
        predictions,
        labels=CLASSES,
    )

    cm_normalized = (
        cm
        / cm.sum(
            axis=1,
            keepdims=True,
        )
    )

    # --------------------------------------------------------
    # Fixed-100-ms permutation test
    # --------------------------------------------------------

    permutation = (
    permutation_test_population(
        X=X,
        y=y,
        groups=groups,
        operands=operands,
        observed_accuracy=
            cv["accuracy"],
        n_permutations=
            n_permutations,
        n_splits=
            n_splits,
        cv_random_state=
            cv_random_state,
        random_state=
            permutation_seed,
        )
    )

    # --------------------------------------------------------
    # SAVE FEATURE MAP
    # --------------------------------------------------------

    feature_map.to_csv(
        output_dir
        / f"{subject}_feature_map.csv",
        index=False,
    )

    data["neurons"].to_csv(
        output_dir
        / f"{subject}_neurons.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE LAMBDA
    # --------------------------------------------------------

    feature_names = (
        feature_map[
            "feature_name"
        ].tolist()
    )

    lambda_df = pd.DataFrame(
        lambda_matrix,
        index=CLASSES,
        columns=feature_names,
    )

    lambda_df.index.name = (
        "number"
    )

    lambda_df.to_csv(
        output_dir
        / f"{subject}_lambda.csv"
    )

    # --------------------------------------------------------
    # SAVE FANO
    # --------------------------------------------------------

    fano_df = pd.DataFrame(
        fano_matrix,
        index=CLASSES,
        columns=feature_names,
    )

    fano_df.index.name = (
        "number"
    )

    fano_df.to_csv(
        output_dir
        / f"{subject}_fano.csv"
    )

    # --------------------------------------------------------
    # SAVE CONFUSION MATRICES
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    )

    cm_df.index.name = (
        "true_number"
    )

    cm_df.columns.name = (
        "predicted_number"
    )

    cm_df.to_csv(
        output_dir
        / f"{subject}_confusion.csv"
    )

    cm_norm_df = pd.DataFrame(
        cm_normalized,
        index=CLASSES,
        columns=CLASSES,
    )

    cm_norm_df.index.name = (
        "true_number"
    )

    cm_norm_df.columns.name = (
        "predicted_number"
    )

    cm_norm_df.to_csv(
        output_dir
        / f"{subject}_confusion_normalized.csv"
    )

    # --------------------------------------------------------
    # SAVE HELD-OUT PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "trial":
                presentations[
                    "trial"
                ].to_numpy(),

            "operand":
                presentations[
                    "operand"
                ].to_numpy(),

            "true_number":
                y,

            "predicted_number":
                predictions,
        }
    )

    for j, k in enumerate(CLASSES):

        prediction_df[
            f"posterior_{k}"
        ] = posterior[:, j]

    prediction_df.to_csv(
        output_dir
        / f"{subject}_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE PERMUTATION NULL
    # --------------------------------------------------------

    np.save(
        output_dir
        / f"{subject}_null_accuracies.npy",
        permutation[
            "null_accuracies"
        ],
    )

    # --------------------------------------------------------
    # SUBJECT SUMMARY
    # --------------------------------------------------------

    summary = {
        "subject":
            subject,

        "bin_size_ms":
            bin_size,

        "n_bins_per_neuron":
            int(900 / bin_size),

        "n_presentations":
            len(y),

        "n_mtl_neurons":
            n_neurons,

        "n_features":
            X.shape[1],

        "accuracy":
            cv["accuracy"],

        "balanced_accuracy":
            cv[
                "balanced_accuracy"
            ],

        "chance":
            1 / 9,

        "mean_fano":
            float(
                np.mean(valid_fano)
            ),

        "median_fano":
            float(
                np.median(valid_fano)
            ),

        "min_fano":
            float(
                np.min(valid_fano)
            ),

        "max_fano":
            float(
                np.max(valid_fano)
            ),

        "frac_fano_0.8_1.2":
            float(
                np.mean(
                    (
                        valid_fano >= 0.8
                    )
                    &
                    (
                        valid_fano <= 1.2
                    )
                )
            ),

        "frac_fano_0.5_1.5":
            float(
                np.mean(
                    (
                        valid_fano >= 0.5
                    )
                    &
                    (
                        valid_fano <= 1.5
                    )
                )
            ),

        "permutation_null_mean":
            permutation[
                "null_mean"
            ],

        "permutation_null_std":
            permutation[
                "null_std"
            ],

        "permutation_p":
            permutation[
                "p_value"
            ],

        "n_permutations":
            n_permutations,
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        output_dir
        / f"{subject}_summary.csv",
        index=False,
    )

    print("\nRESULT")
    print("-" * 70)

    print(
        f"Accuracy: "
        f"{100 * summary['accuracy']:.2f}%"
    )

    print(
        f"Balanced accuracy: "
        f"{100 * summary['balanced_accuracy']:.2f}%"
    )

    print(
        f"Chance: "
        f"{100 / 9:.2f}%"
    )

    print(
        f"Mean Fano: "
        f"{summary['mean_fano']:.4f}"
    )

    print(
        f"Permutation null mean: "
        f"{100 * summary['permutation_null_mean']:.2f}%"
    )

    print(
        f"Permutation p: "
        f"{summary['permutation_p']:.4f}"
    )

    print(
        f"\nSaved to:\n{output_dir}"
    )

    return summary_df