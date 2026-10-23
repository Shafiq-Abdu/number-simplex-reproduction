"""
Gamma-Poisson Naive Bayes population decoding.

For each subject:

1. Build the population temporal feature matrix.
2. Run 5-fold StratifiedGroupKFold cross-validation.
3. Evaluate:
      - Uniform class prior
      - Empirical class prior
4. Report:
      - OOF accuracy
      - OOF balanced accuracy
5. Fit the Gamma-Poisson model on the full dataset.
6. Return the 9 x p posterior-mean lambda matrix.

Model
-----
For class k and feature j:

    X_ij | Y=k, lambda_kj ~ Poisson(lambda_kj * dt)

Gamma prior:

    lambda_kj ~ Gamma(alpha_j, beta)

The pooled training-fold MLE rate is

    m_j = sum_i X_ij / (N_train * dt)

and we use

    m_prior_j = max(m_j, epsilon)

with epsilon = 0.01 spikes/second.

Then

    alpha_j = m_prior_j * beta.

Posterior:

    alpha*_kj = alpha_j + S_kj
    beta*_kj  = beta + n_k * dt

Posterior mean:

    lambda*_kj = alpha*_kj / beta*_kj
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.special import gammaln

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.population_level_decoding import build_population_matrix


# ============================================================
# GAMMA-POISSON POSTERIOR
# ============================================================

def fit_gamma_poisson(
    X_train,
    y_train,
    dt=0.1,
    beta=0.1,
    epsilon=0.01,
):
    """
    Fit class-specific Gamma posteriors.

    Returns
    -------
    alpha_post : ndarray, shape (9, p)
    beta_post : ndarray, shape (9, p)
    lambda_post : ndarray, shape (9, p)
        Posterior mean firing rates.
    """

    classes = np.arange(1, 10)

    n_train, n_features = X_train.shape

    # --------------------------------------------------------
    # Global pooled MLE firing rate
    # --------------------------------------------------------

    m_mle = (
        X_train.sum(axis=0)
        / (n_train * dt)
    )

    # --------------------------------------------------------
    # Positive floor
    # --------------------------------------------------------

    m_prior = np.maximum(
        m_mle,
        epsilon
    )

    # --------------------------------------------------------
    # Gamma prior
    # --------------------------------------------------------

    alpha = m_prior * beta

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    alpha_post = np.zeros(
        (9, n_features)
    )

    beta_post = np.zeros(
        (9, n_features)
    )

    lambda_post = np.zeros(
        (9, n_features)
    )

    # --------------------------------------------------------
    # Class-specific posteriors
    # --------------------------------------------------------

    for k in classes:

        X_k = X_train[
            y_train == k
        ]

        n_k = len(X_k)

        if n_k == 0:
            raise ValueError(
                f"Class {k} has no training observations."
            )

        S_k = X_k.sum(axis=0)

        alpha_star = (
            alpha + S_k
        )

        beta_star = (
            beta + n_k * dt
        )

        lambda_star = (
            alpha_star
            / beta_star
        )

        alpha_post[k - 1] = (
            alpha_star
        )

        beta_post[k - 1] = (
            beta_star
        )

        lambda_post[k - 1] = (
            lambda_star
        )

    diagnostics = {
        "zero_mle_features":
            int(np.sum(m_mle == 0)),

        "floored_features":
            int(np.sum(m_mle < epsilon)),

        "min_mle":
            float(m_mle.min()),

        "min_prior_rate":
            float(m_prior.min()),
    }

    return (
        alpha_post,
        beta_post,
        lambda_post,
        diagnostics,
    )


# ============================================================
# POSTERIOR-PREDICTIVE CLASSIFICATION
# ============================================================

def predict_gamma_poisson(
    X_test,
    alpha_post,
    beta_post,
    class_priors,
    dt=0.1,
):
    """
    Predict number labels using the Gamma-Poisson
    posterior predictive distribution.
    """

    predictions = []

    for x in X_test:

        class_scores = np.zeros(9)

        for k in range(9):

            a = alpha_post[k]
            b = beta_post[k]

            # Gamma-Poisson posterior predictive
            log_predictive = (
                gammaln(a + x)
                - gammaln(a)
                - gammaln(x + 1)
                + a * np.log(
                    b / (b + dt)
                )
                + x * np.log(
                    dt / (b + dt)
                )
            )

            log_class_prior = np.log(
                class_priors[k]
            )

            class_scores[k] = (
                log_predictive.sum()
                + log_class_prior
            )

        prediction = (
            np.argmax(class_scores)
            + 1
        )

        predictions.append(
            prediction
        )

    return np.asarray(predictions)


# ============================================================
# CROSS-VALIDATED DECODING
# ============================================================

def cross_validate_gamma_poisson(
    X,
    y,
    groups,
    class_prior="uniform",
    dt=0.1,
    beta=0.1,
    epsilon=0.01,
    n_splits=5,
    random_state=42,
):
    """
    Run grouped out-of-fold Gamma-Poisson decoding.

    class_prior
    -----------
    "uniform"
        P(Y=k) = 1/9

    "empirical"
        P(Y=k) = n_k / N_train
        estimated separately inside each training fold.
    """

    if class_prior not in {
        "uniform",
        "empirical",
    }:
        raise ValueError(
            "class_prior must be "
            "'uniform' or 'empirical'."
        )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    y_pred_oof = np.zeros_like(y)

    fold_results = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):

        # ----------------------------------------------------
        # Train / test split
        # ----------------------------------------------------

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ----------------------------------------------------
        # Fit Gamma-Poisson model
        # ----------------------------------------------------

        (
            alpha_post,
            beta_post,
            _,
            diagnostics,
        ) = fit_gamma_poisson(
            X_train=X_train,
            y_train=y_train,
            dt=dt,
            beta=beta,
            epsilon=epsilon,
        )

        # ----------------------------------------------------
        # Class prior
        # ----------------------------------------------------

        if class_prior == "uniform":

            class_priors = np.full(
                9,
                1 / 9,
            )

        else:

            class_priors = np.array([
                np.mean(
                    y_train == k
                )
                for k in range(1, 10)
            ])

        # ----------------------------------------------------
        # Predict held-out presentations
        # ----------------------------------------------------

        fold_predictions = (
            predict_gamma_poisson(
                X_test=X_test,
                alpha_post=alpha_post,
                beta_post=beta_post,
                class_priors=class_priors,
                dt=dt,
            )
        )

        # Store OOF predictions
        y_pred_oof[test_idx] = (
            fold_predictions
        )

        # ----------------------------------------------------
        # Fold metrics
        # ----------------------------------------------------

        fold_accuracy = (
            accuracy_score(
                y_test,
                fold_predictions,
            )
        )

        fold_balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                fold_predictions,
            )
        )

        fold_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "zero_mle_features":
                diagnostics[
                    "zero_mle_features"
                ],
            "floored_features":
                diagnostics[
                    "floored_features"
                ],
            "accuracy":
                fold_accuracy,
            "balanced_accuracy":
                fold_balanced_accuracy,
        })

    # --------------------------------------------------------
    # Pooled OOF metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        y_pred_oof,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            y_pred_oof,
        )
    )

    return {
        "class_prior": class_prior,
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_accuracy,
        "n_correct":
            int(
                np.sum(
                    y == y_pred_oof
                )
            ),
        "n_presentations":
            len(y),
        "y_pred_oof":
            y_pred_oof,
        "fold_results":
            pd.DataFrame(
                fold_results
            ),
    }


# ============================================================
# SINGLE SUBJECT
# ============================================================

def analyze_subject_gamma_poisson(
    subject,
    root,
    dt=0.1,
    beta=0.1,
    epsilon=0.01,
    bin_ms=100,
    window_start_ms=50,
    window_end_ms=950,
    n_splits=5,
    random_state=42,
    save_lambda=True,
):
    """
    Complete Gamma-Poisson analysis for one subject.

    Runs:
        1. Uniform class-prior decoding
        2. Empirical class-prior decoding
        3. Full-data posterior lambda matrix
    """

    root = Path(root)

    # --------------------------------------------------------
    # Load selected neurons
    # --------------------------------------------------------

    per_neuron_path = (
        root
        / "tables1"
        / f"figure1M_{subject}_permutation.csv"
    )

    per_neuron = pd.read_csv(
        per_neuron_path
    )

    neuron_ids = (
        per_neuron["neuron"]
        .astype(int)
        .tolist()
    )

    # --------------------------------------------------------
    # Build population matrix
    # --------------------------------------------------------

    (
        X,
        y,
        groups,
        presentations,
        metadata,
    ) = build_population_matrix(
        subject=subject,
        neuron_ids=neuron_ids,
        root=root,
        bin_ms=bin_ms,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
    )

    # --------------------------------------------------------
    # Uniform class prior
    # --------------------------------------------------------

    uniform_result = (
        cross_validate_gamma_poisson(
            X=X,
            y=y,
            groups=groups,
            class_prior="uniform",
            dt=dt,
            beta=beta,
            epsilon=epsilon,
            n_splits=n_splits,
            random_state=random_state,
        )
    )

    # --------------------------------------------------------
    # Empirical class prior
    # --------------------------------------------------------

    empirical_result = (
        cross_validate_gamma_poisson(
            X=X,
            y=y,
            groups=groups,
            class_prior="empirical",
            dt=dt,
            beta=beta,
            epsilon=epsilon,
            n_splits=n_splits,
            random_state=random_state,
        )
    )

    # --------------------------------------------------------
    # Final full-data Lambda matrix
    #
    # This is for representation / geometry.
    # It is NOT used for OOF accuracy.
    # --------------------------------------------------------

    (
        alpha_post_full,
        beta_post_full,
        lambda_matrix,
        full_diagnostics,
    ) = fit_gamma_poisson(
        X_train=X,
        y_train=y,
        dt=dt,
        beta=beta,
        epsilon=epsilon,
    )

    # --------------------------------------------------------
    # Save Lambda
    # --------------------------------------------------------

    lambda_path = None

    if save_lambda:

        lambda_df = pd.DataFrame(
            lambda_matrix,
            index=[
                f"number_{k}"
                for k in range(1, 10)
            ],
        )

        lambda_df.index.name = "number"

        lambda_path = (
            root
            / "tables1"
            / (
                f"{subject}_"
                "gamma_poisson_"
                "lambda_matrix.csv"
            )
        )

        lambda_df.to_csv(
            lambda_path
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "subject": subject,
        "n_neurons":
            len(neuron_ids),
        "n_presentations":
            len(y),
        "n_features":
            X.shape[1],

        "uniform_correct":
            uniform_result[
                "n_correct"
            ],

        "uniform_accuracy":
            uniform_result[
                "accuracy"
            ],

        "uniform_balanced_accuracy":
            uniform_result[
                "balanced_accuracy"
            ],

        "empirical_correct":
            empirical_result[
                "n_correct"
            ],

        "empirical_accuracy":
            empirical_result[
                "accuracy"
            ],

        "empirical_balanced_accuracy":
            empirical_result[
                "balanced_accuracy"
            ],

        "full_zero_mle_features":
            full_diagnostics[
                "zero_mle_features"
            ],

        "full_floored_features":
            full_diagnostics[
                "floored_features"
            ],

        "lambda_rows":
            lambda_matrix.shape[0],

        "lambda_columns":
            lambda_matrix.shape[1],
    }

    return {
        "summary": summary,
        "uniform": uniform_result,
        "empirical": empirical_result,
        "lambda_matrix":
            lambda_matrix,
        "alpha_post_full":
            alpha_post_full,
        "beta_post_full":
            beta_post_full,
        "lambda_path":
            lambda_path,
        "metadata":
            metadata,
        "presentations":
            presentations,
    }


# ============================================================
# ALL SUBJECTS
# ============================================================

def analyze_all_subjects_gamma_poisson(
    subjects,
    root,
    dt=0.1,
    beta=0.1,
    epsilon=0.01,
    bin_ms=100,
    window_start_ms=50,
    window_end_ms=950,
    n_splits=5,
    random_state=42,
    save_results=True,
):
    """
    Run the complete Gamma-Poisson analysis
    across all subjects.
    """

    root = Path(root)

    all_results = {}
    summary_rows = []

    for subject in subjects:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Running subject: {subject}"
        )

        print(
            "=" * 60
        )

        result = (
            analyze_subject_gamma_poisson(
                subject=subject,
                root=root,
                dt=dt,
                beta=beta,
                epsilon=epsilon,
                bin_ms=bin_ms,
                window_start_ms=
                    window_start_ms,
                window_end_ms=
                    window_end_ms,
                n_splits=n_splits,
                random_state=
                    random_state,
                save_lambda=True,
            )
        )

        all_results[subject] = result

        summary = result["summary"]

        summary_rows.append(
            summary
        )

        print(
            "Uniform accuracy: "
            f"{100 * summary['uniform_accuracy']:.2f}%"
        )

        print(
            "Uniform balanced: "
            f"{100 * summary['uniform_balanced_accuracy']:.2f}%"
        )

        print(
            "Empirical accuracy: "
            f"{100 * summary['empirical_accuracy']:.2f}%"
        )

        print(
            "Empirical balanced: "
            f"{100 * summary['empirical_balanced_accuracy']:.2f}%"
        )

        print(
            "Lambda shape: "
            f"({summary['lambda_rows']}, "
            f"{summary['lambda_columns']})"
        )

    # --------------------------------------------------------
    # Final summary table
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    if save_results:

        output_path = (
            root
            / "tables1"
            / "gamma_poisson_all_subjects_summary.csv"
        )

        summary_df.to_csv(
            output_path,
            index=False,
        )

        print(
            "\nSaved summary to:"
        )

        print(output_path)

    return (
        summary_df,
        all_results,
    )