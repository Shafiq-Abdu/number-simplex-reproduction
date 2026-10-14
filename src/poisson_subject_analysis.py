import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from src.firing_rate_tuning import analyze_neuron
from src.number_decoding_analysis import make_temporal_features
from src.poisson_naive_bayes import cv_poisson_nb


MTL_REGIONS = {"hpc", "ent", "amy", "para-hpc"}


def compute_fano_lambda(X, y):
    """
    Compute 9 x n_features lambda and Fano matrices.
    """

    classes = np.arange(1, 10)

    n_features = X.shape[1]

    lambda_matrix = np.full(
        (9, n_features),
        np.nan,
    )

    fano_matrix = np.full(
        (9, n_features),
        np.nan,
    )

    for i, k in enumerate(classes):

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

    return lambda_matrix, fano_matrix
def analyze_poisson_subject(
    subject,
    root,
    bin_size=100,
    n_splits=5,
    mtl_only=True,
):
    """
    Run single-neuron Poisson analysis for every neuron
    in one subject.
    """

    # First neuron only to learn total neuron count
    first = analyze_neuron(
        subject,
        1,
        root=root,
        plot=False,
        verbose=False,
    )

    total_neurons = first["total_neurons"]

    summary_rows = []

    lambda_results = {}
    fano_results = {}
    confusion_results = {}

    for neuron in range(1, total_neurons + 1):

        print(
            f"{subject}: neuron "
            f"{neuron}/{total_neurons}"
        )

        result = analyze_neuron(
            subject,
            neuron,
            root=root,
            plot=False,
            verbose=False,
        )

        region = result["region"]

        if (
            mtl_only
            and region not in MTL_REGIONS
        ):
            continue

        presentations = result["presentations"]
        spike_times = result["spike_times"]

        X, y, edges = make_temporal_features(
            presentations,
            spike_times,
            bin_size=bin_size,
        )

        # ------------------------------------
        # Lambda + Fano
        # ------------------------------------

        lambda_matrix, fano_matrix = (
            compute_fano_lambda(
                X,
                y,
            )
        )

        # ------------------------------------
        # Grouped Poisson CV
        # ------------------------------------

        groups = (
            presentations["trial"]
            .to_numpy()
        )

        cv_result = cv_poisson_nb(
            X,
            y,
            groups=groups,
            n_splits=n_splits,
            grouped=True,
        )

        # ------------------------------------
        # Confusion matrix
        # ------------------------------------

        cm = confusion_matrix(
            cv_result["y_true"],
            cv_result["y_pred"],
            labels=np.arange(1, 10),
        )

        # ------------------------------------
        # Fano summaries
        # ------------------------------------

        valid_fano = (
            fano_matrix[
                np.isfinite(fano_matrix)
            ]
        )

        mean_fano = np.mean(valid_fano)
        median_fano = np.median(valid_fano)

        frac_08_12 = np.mean(
            (valid_fano >= 0.8)
            &
            (valid_fano <= 1.2)
        )

        frac_05_15 = np.mean(
            (valid_fano >= 0.5)
            &
            (valid_fano <= 1.5)
        )

        summary_rows.append(
            {
                "subject": subject,
                "neuron": neuron,
                "region": region,
                "n_presentations": len(y),
                "bin_size_ms": bin_size,
                "accuracy": cv_result[
                    "accuracy"
                ],
                "chance": 1 / 9,
                "mean_fano": mean_fano,
                "median_fano": median_fano,
                "frac_fano_0.8_1.2":
                    frac_08_12,
                "frac_fano_0.5_1.5":
                    frac_05_15,
            }
        )

        lambda_results[neuron] = (
            lambda_matrix
        )

        fano_results[neuron] = (
            fano_matrix
        )

        confusion_results[neuron] = cm

    summary = pd.DataFrame(
        summary_rows
    )

    return {
        "summary": summary,
        "lambda": lambda_results,
        "fano": fano_results,
        "confusion": confusion_results,
    }