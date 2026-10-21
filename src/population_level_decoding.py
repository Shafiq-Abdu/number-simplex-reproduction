"""
Population-level number decoding.

For one subject:
    1. Load all selected neurons.
    2. Verify presentation alignment across neurons.
    3. Construct fixed-width temporal spike-count features.
    4. Concatenate neuron-wise temporal features into a population vector.
    5. Decode numbers 1-9 using trial-grouped cross-validation.
    6. Compare shrinkage LDA and multinomial logistic regression.

Primary analysis:
    Window: 50-950 ms after number onset
    Bin size: 100 ms
    CV: 5-fold StratifiedGroupKFold
    Group: behavioral trial

One presentation therefore has:

    n_neurons * n_time_bins

features.

For the default 900-ms window and 100-ms bins:
    n_time_bins = 9.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import src.number_decoding_analysis as nda


def build_population_matrix(
    subject,
    neuron_ids,
    root,
    bin_ms=100,
    window_start_ms=50,
    window_end_ms=950,
):
    """
    Construct the temporal population feature matrix for one subject.

    Returns
    -------
    X : ndarray, shape (n_presentations, n_features)
        Population spike-count features.

    y : ndarray, shape (n_presentations,)
        Number labels 1-9.

    groups : ndarray, shape (n_presentations,)
        Behavioral trial IDs.

    presentations : DataFrame
        Reference presentation table.

    metadata : dict
        Information about matrix construction.
    """

    root = Path(root)
    neuron_ids = [int(n) for n in neuron_ids]

    if len(neuron_ids) == 0:
        raise ValueError("neuron_ids is empty.")

    # ---------------------------------------------------------
    # Fixed temporal bins
    # ---------------------------------------------------------

    window_length = window_end_ms - window_start_ms

    if window_length % bin_ms != 0:
        raise ValueError(
            "Analysis window must be exactly divisible by bin_ms."
        )

    bin_edges = np.arange(
        window_start_ms,
        window_end_ms + bin_ms,
        bin_ms,
    )

    n_bins = len(bin_edges) - 1

    # ---------------------------------------------------------
    # Reference neuron / presentation sequence
    # ---------------------------------------------------------

    ref_id = neuron_ids[0]

    ref_out = nda.analyze_neuron(
        subject,
        ref_id,
        root=root,
        plot=False,
        verbose=False,
    )

    presentations = (
        ref_out["presentations"]
        .reset_index(drop=True)
        .copy()
    )

    alignment_columns = [
        "trial",
        "number",
        "operand",
        "onset",
    ]

    population_blocks = []

    # ---------------------------------------------------------
    # Process every neuron
    # ---------------------------------------------------------

    for neuron_id in neuron_ids:

        out_n = nda.analyze_neuron(
            subject,
            neuron_id,
            root=root,
            plot=False,
            verbose=False,
        )

        p = (
            out_n["presentations"]
            .reset_index(drop=True)
        )

        # -----------------------------------------------------
        # Verify presentation alignment
        # -----------------------------------------------------

        if len(p) != len(presentations):
            raise ValueError(
                f"{subject}, neuron {neuron_id}: "
                f"presentation count does not match reference neuron."
            )

        same_trial = np.array_equal(
            p["trial"].to_numpy(),
            presentations["trial"].to_numpy(),
        )

        same_number = np.array_equal(
            p["number"].to_numpy(),
            presentations["number"].to_numpy(),
        )

        same_operand = np.array_equal(
            p["operand"].to_numpy(),
            presentations["operand"].to_numpy(),
        )

        same_onset = np.allclose(
            p["onset"].to_numpy(),
            presentations["onset"].to_numpy(),
        )

        if not (
            same_trial
            and same_number
            and same_operand
            and same_onset
        ):
            raise ValueError(
                f"{subject}, neuron {neuron_id}: "
                "presentation alignment failed."
            )

        # -----------------------------------------------------
        # Temporal spike-count matrix for this neuron
        # -----------------------------------------------------

        spike_times = out_n["spike_times"]

        neuron_matrix = []

        for onset in presentations["onset"]:

            relative_spikes = spike_times - onset

            # Match the original analysis window exactly:
            # include 50 ms, exclude 950 ms.
            relative_spikes = relative_spikes[
                (relative_spikes >= window_start_ms)
                & (relative_spikes < window_end_ms)
            ]
            counts, _ = np.histogram(
                relative_spikes,
                bins=bin_edges,
            )

            neuron_matrix.append(counts)

        neuron_matrix = np.asarray(neuron_matrix)

        # -----------------------------------------------------
        # Sanity check against original 50-950 ms spike count
        # -----------------------------------------------------

        original_counts = (
            p["spike_count"]
            .to_numpy()
        )

        binned_counts = neuron_matrix.sum(axis=1)

        if not np.array_equal(
            binned_counts,
            original_counts,
        ):
            raise ValueError(
                f"{subject}, neuron {neuron_id}: "
                "temporal bins do not reproduce original spike counts."
            )

        population_blocks.append(neuron_matrix)

    # ---------------------------------------------------------
    # Concatenate all neurons horizontally
    # ---------------------------------------------------------

    X = np.hstack(population_blocks)

    y = (
        presentations["number"]
        .astype(int)
        .to_numpy()
    )

    groups = (
        presentations["trial"]
        .astype(int)
        .to_numpy()
    )

    expected_features = len(neuron_ids) * n_bins

    if X.shape[1] != expected_features:
        raise RuntimeError(
            f"Expected {expected_features} features, "
            f"but obtained {X.shape[1]}."
        )

    metadata = {
        "subject": subject,
        "n_neurons": len(neuron_ids),
        "n_presentations": len(presentations),
        "n_trials": len(np.unique(groups)),
        "n_bins": n_bins,
        "n_features": X.shape[1],
        "bin_ms": bin_ms,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "reference_neuron": ref_id,
    }

    return X, y, groups, presentations, metadata


def decode_population(
    X,
    y,
    groups,
    n_splits=5,
    random_state=42,
):
    """
    Cross-validated population decoding using identical folds
    for shrinkage LDA and multinomial logistic regression.

    Returns
    -------
    fold_results : DataFrame
        Accuracy and balanced accuracy for every CV fold.

    summary : dict
        Mean CV performance.
    """

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = list(
        cv.split(
            X,
            y,
            groups=groups,
        )
    )

    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
    )

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
        ),
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Ensure trial groups never cross train/test
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        overlap = train_groups.intersection(
            test_groups
        )

        if overlap:
            raise RuntimeError(
                f"Fold {fold}: train/test trial leakage detected."
            )

        # -----------------------------------------------------
        # LDA
        # -----------------------------------------------------

        lda.fit(
            X_train,
            y_train,
        )

        pred_lda = lda.predict(
            X_test
        )

        # -----------------------------------------------------
        # Logistic regression
        # -----------------------------------------------------

        logistic.fit(
            X_train,
            y_train,
        )

        pred_logistic = logistic.predict(
            X_test
        )

        fold_results.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_trials": len(train_groups),
                "n_test_trials": len(test_groups),

                "lda_accuracy": accuracy_score(
                    y_test,
                    pred_lda,
                ),

                "lda_balanced_accuracy":
                    balanced_accuracy_score(
                        y_test,
                        pred_lda,
                    ),

                "logistic_accuracy":
                    accuracy_score(
                        y_test,
                        pred_logistic,
                    ),

                "logistic_balanced_accuracy":
                    balanced_accuracy_score(
                        y_test,
                        pred_logistic,
                    ),
            }
        )

    fold_results = pd.DataFrame(
        fold_results
    )

    summary = {
        "lda_accuracy":
            fold_results[
                "lda_accuracy"
            ].mean(),

        "lda_balanced_accuracy":
            fold_results[
                "lda_balanced_accuracy"
            ].mean(),

        "logistic_accuracy":
            fold_results[
                "logistic_accuracy"
            ].mean(),

        "logistic_balanced_accuracy":
            fold_results[
                "logistic_balanced_accuracy"
            ].mean(),

        "n_splits": n_splits,
        "random_state": random_state,
    }

    return fold_results, summary


def analyze_subject_population(
    subject,
    neuron_ids,
    root,
    bin_ms=100,
    window_start_ms=50,
    window_end_ms=950,
    n_splits=5,
    random_state=42,
    save=True,
):
    """
    Complete population-decoding analysis for one subject.

    This is the main function intended for parallel server runs.
    """

    root = Path(root)

    print("=" * 60)
    print(f"POPULATION DECODING: {subject}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Build population representation
    # ---------------------------------------------------------

    X, y, groups, presentations, metadata = (
        build_population_matrix(
            subject=subject,
            neuron_ids=neuron_ids,
            root=root,
            bin_ms=bin_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
    )

    print(
        f"Neurons       : {metadata['n_neurons']}"
    )
    print(
        f"Presentations : {metadata['n_presentations']}"
    )
    print(
        f"Trials        : {metadata['n_trials']}"
    )
    print(
        f"Bins/neuron   : {metadata['n_bins']}"
    )
    print(
        f"Features      : {metadata['n_features']}"
    )
    print(
        f"X shape       : {X.shape}"
    )

    # ---------------------------------------------------------
    # Decode
    # ---------------------------------------------------------

    fold_results, decoding = decode_population(
        X=X,
        y=y,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
    )

    result = {
        **metadata,
        **decoding,
        "nominal_chance": 1 / len(np.unique(y)),
    }

    summary_df = pd.DataFrame(
        [result]
    )

    print("\nDecoding results")
    print("-" * 60)

    print(
        f"LDA accuracy               : "
        f"{result['lda_accuracy']:.3f}"
    )

    print(
        f"LDA balanced accuracy      : "
        f"{result['lda_balanced_accuracy']:.3f}"
    )

    print(
        f"Logistic accuracy          : "
        f"{result['logistic_accuracy']:.3f}"
    )

    print(
        f"Logistic balanced accuracy : "
        f"{result['logistic_balanced_accuracy']:.3f}"
    )

    print(
        f"Nominal chance             : "
        f"{result['nominal_chance']:.3f}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    if save:

        output_dir = (
            root
            / "tables1"
            / "population_decoding"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = (
            output_dir
            / f"{subject}_population_summary.csv"
        )

        folds_path = (
            output_dir
            / f"{subject}_population_folds.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        fold_results.to_csv(
            folds_path,
            index=False,
        )

        print(
            f"\nSaved summary : {summary_path}"
        )

        print(
            f"Saved folds   : {folds_path}"
        )

    return {
        "summary": summary_df,
        "fold_results": fold_results,
        "X": X,
        "y": y,
        "groups": groups,
        "presentations": presentations,
    }