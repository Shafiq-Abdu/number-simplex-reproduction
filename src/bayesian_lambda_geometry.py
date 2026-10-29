import numpy as np

from math import factorial

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Helper: magnitude-axis regression
# ============================================================

def _fit_magnitude_axis(C, x):
    """
    Fit the best one-dimensional magnitude axis

        C_k ≈ x_k * b

    using least squares.

    Parameters
    ----------
    C : ndarray, shape (9, p)
        Centered condition-mean matrix.

    x : ndarray, shape (9,)
        Numerical predictor.

    Returns
    -------
    dict
        Contains fitted direction, prediction, residual, and R².
    """

    x = np.asarray(x, dtype=float)

    # Center predictor
    x = x - x.mean()

    # Least-squares neural direction
    b = (x @ C) / (x @ x)

    # Predicted condition geometry
    C_hat = np.outer(x, b)

    # Residual geometry
    residual = C - C_hat

    total_ss = np.sum(C ** 2)
    explained_ss = np.sum(C_hat ** 2)
    residual_ss = np.sum(residual ** 2)

    r2 = explained_ss / total_ss

    return {
        "x": x,
        "b": b,
        "C_hat": C_hat,
        "residual": residual,
        "total_ss": total_ss,
        "explained_ss": explained_ss,
        "residual_ss": residual_ss,
        "r2": r2,
    }


# ============================================================
# Helper: RDM permutation tests
# ============================================================

def _rdm_permutation_test(
    pairwise_distances,
    n_perm=20000,
    random_state=42,
):
    """
    Permutation test for the relationship between neural
    distance and numerical distance.

    Neural geometry is kept fixed while labels 1,...,9
    are randomly reassigned.
    """

    numbers = np.arange(1, 10, dtype=float)
    upper = np.triu_indices(9, k=1)

    # --------------------------------------------------------
    # Observed linear numerical distance
    # --------------------------------------------------------

    numerical_D = np.abs(
        numbers[:, None] - numbers[None, :]
    )

    numerical_distances = numerical_D[upper]

    obs_linear_pearson = pearsonr(
        numerical_distances,
        pairwise_distances,
    ).statistic

    obs_linear_spearman = spearmanr(
        numerical_distances,
        pairwise_distances,
    ).statistic

    # --------------------------------------------------------
    # Observed logarithmic numerical distance
    # --------------------------------------------------------

    log_numbers = np.log(numbers)

    log_D = np.abs(
        log_numbers[:, None] - log_numbers[None, :]
    )

    log_distances = log_D[upper]

    obs_log_pearson = pearsonr(
        log_distances,
        pairwise_distances,
    ).statistic

    obs_log_spearman = spearmanr(
        log_distances,
        pairwise_distances,
    ).statistic

    # --------------------------------------------------------
    # Null distributions
    # --------------------------------------------------------

    rng = np.random.default_rng(random_state)

    null_linear_pearson = np.empty(n_perm)
    null_linear_spearman = np.empty(n_perm)

    null_log_pearson = np.empty(n_perm)
    null_log_spearman = np.empty(n_perm)

    for b in range(n_perm):

        permuted_numbers = rng.permutation(numbers)

        # Linear numerical distance
        perm_D = np.abs(
            permuted_numbers[:, None]
            - permuted_numbers[None, :]
        )

        perm_distances = perm_D[upper]

        null_linear_pearson[b] = pearsonr(
            perm_distances,
            pairwise_distances,
        ).statistic

        null_linear_spearman[b] = spearmanr(
            perm_distances,
            pairwise_distances,
        ).statistic

        # Logarithmic numerical distance
        perm_log = np.log(permuted_numbers)

        perm_log_D = np.abs(
            perm_log[:, None]
            - perm_log[None, :]
        )

        perm_log_distances = perm_log_D[upper]

        null_log_pearson[b] = pearsonr(
            perm_log_distances,
            pairwise_distances,
        ).statistic

        null_log_spearman[b] = spearmanr(
            perm_log_distances,
            pairwise_distances,
        ).statistic

    # --------------------------------------------------------
    # Two-sided permutation p-values
    # --------------------------------------------------------

    linear_pearson_two_p = (
        np.sum(
            np.abs(null_linear_pearson)
            >= abs(obs_linear_pearson)
        ) + 1
    ) / (n_perm + 1)

    linear_spearman_two_p = (
        np.sum(
            np.abs(null_linear_spearman)
            >= abs(obs_linear_spearman)
        ) + 1
    ) / (n_perm + 1)

    log_pearson_two_p = (
        np.sum(
            np.abs(null_log_pearson)
            >= abs(obs_log_pearson)
        ) + 1
    ) / (n_perm + 1)

    log_spearman_two_p = (
        np.sum(
            np.abs(null_log_spearman)
            >= abs(obs_log_spearman)
        ) + 1
    ) / (n_perm + 1)

    # --------------------------------------------------------
    # Positive-tail tests
    #
    # Number-line hypothesis predicts positive correlation.
    # --------------------------------------------------------

    linear_pearson_positive_p = (
        np.sum(
            null_linear_pearson
            >= obs_linear_pearson
        ) + 1
    ) / (n_perm + 1)

    linear_spearman_positive_p = (
        np.sum(
            null_linear_spearman
            >= obs_linear_spearman
        ) + 1
    ) / (n_perm + 1)

    log_pearson_positive_p = (
        np.sum(
            null_log_pearson
            >= obs_log_pearson
        ) + 1
    ) / (n_perm + 1)

    log_spearman_positive_p = (
        np.sum(
            null_log_spearman
            >= obs_log_spearman
        ) + 1
    ) / (n_perm + 1)

    return {
        "linear_pearson_two_p": linear_pearson_two_p,
        "linear_spearman_two_p": linear_spearman_two_p,
        "log_pearson_two_p": log_pearson_two_p,
        "log_spearman_two_p": log_spearman_two_p,

        "linear_pearson_positive_p": linear_pearson_positive_p,
        "linear_spearman_positive_p": linear_spearman_positive_p,
        "log_pearson_positive_p": log_pearson_positive_p,
        "log_spearman_positive_p": log_spearman_positive_p,

        "null_linear_pearson": null_linear_pearson,
        "null_linear_spearman": null_linear_spearman,
        "null_log_pearson": null_log_pearson,
        "null_log_spearman": null_log_spearman,
    }


# ============================================================
# Helper: magnitude-axis permutation test
# ============================================================

def _magnitude_permutation_test(
    C,
    n_perm=20000,
    random_state=42,
):
    """
    Test whether the canonical numerical ordering explains
    more one-dimensional variance than random label orderings.
    """

    numbers = np.arange(1, 10, dtype=float)

    observed_linear = _fit_magnitude_axis(
        C,
        numbers,
    )

    observed_log = _fit_magnitude_axis(
        C,
        np.log(numbers),
    )

    observed_linear_r2 = observed_linear["r2"]
    observed_log_r2 = observed_log["r2"]

    rng = np.random.default_rng(random_state)

    null_linear_r2 = np.empty(n_perm)
    null_log_r2 = np.empty(n_perm)

    for b in range(n_perm):

        permuted_numbers = rng.permutation(numbers)

        null_linear_r2[b] = _fit_magnitude_axis(
            C,
            permuted_numbers,
        )["r2"]

        null_log_r2[b] = _fit_magnitude_axis(
            C,
            np.log(permuted_numbers),
        )["r2"]

    # One-sided test:
    # Does the true ordering explain MORE than random orderings?

    linear_p = (
        np.sum(
            null_linear_r2 >= observed_linear_r2
        ) + 1
    ) / (n_perm + 1)

    log_p = (
        np.sum(
            null_log_r2 >= observed_log_r2
        ) + 1
    ) / (n_perm + 1)

    return {
        "observed_linear": observed_linear,
        "observed_log": observed_log,

        "linear_p": linear_p,
        "log_p": log_p,

        "null_linear_r2": null_linear_r2,
        "null_log_r2": null_log_r2,

        "linear_null_mean": null_linear_r2.mean(),
        "log_null_mean": null_log_r2.mean(),

        "linear_null_95": np.percentile(
            null_linear_r2,
            95,
        ),

        "log_null_95": np.percentile(
            null_log_r2,
            95,
        ),
    }


# ============================================================
# Main analysis function
# ============================================================

def analyze_lambda_geometry(
    Lambda,
    subject=None,
    n_perm=20000,
    random_state=42,
):
    """
    Run the complete Steps 1-10 geometry analysis on one
    subject's Bayesian posterior-mean Lambda matrix.

    Parameters
    ----------
    Lambda : ndarray, shape (9, p)
        Bayesian posterior-mean number representation.

        Rows:
            numbers 1,...,9

        Columns:
            neural temporal features

    subject : str, optional
        Subject identifier such as "YFF".

    n_perm : int
        Number of permutations for number-line tests.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        All scalar results and detailed intermediate objects.
    """

    # ========================================================
    # STEP 1
    # Verify Lambda
    # ========================================================

    Lambda = np.asarray(
        Lambda,
        dtype=float,
    )

    if Lambda.ndim != 2:
        raise ValueError(
            "Lambda must be a two-dimensional matrix."
        )

    if Lambda.shape[0] != 9:
        raise ValueError(
            "Lambda must contain exactly 9 rows "
            "for number conditions 1-9."
        )

    if not np.all(np.isfinite(Lambda)):
        raise ValueError(
            "Lambda contains NaN or infinite values."
        )

    n_conditions = Lambda.shape[0]
    n_features = Lambda.shape[1]

    # ========================================================
    # STEP 2
    # Centering
    # ========================================================

    centroid = Lambda.mean(axis=0)

    C = Lambda - centroid

    centroid_distances = np.linalg.norm(
        C,
        axis=1,
    )

    radius_mean = centroid_distances.mean()

    radius_sd = centroid_distances.std(
        ddof=0
    )

    radius_cv = radius_sd / radius_mean

    # ========================================================
    # STEP 3
    # SVD and dimensionality
    # ========================================================

    U, singular_values, Vt = np.linalg.svd(
        C,
        full_matrices=False,
    )

    affine_rank = np.linalg.matrix_rank(C)

    variance = singular_values ** 2

    variance_fraction = (
        variance / variance.sum()
    )

    cumulative_variance = np.cumsum(
        variance_fraction
    )

    # Only numerically nonzero dimensions
    p_nonzero = variance_fraction[
        :affine_rank
    ]

    effective_rank = np.exp(
        -np.sum(
            p_nonzero
            * np.log(p_nonzero)
        )
    )

    participation_ratio = (
        1.0
        / np.sum(p_nonzero ** 2)
    )

    pc1_variance = variance_fraction[0]

    symmetric_condition_number = (
        singular_values[0]
        / singular_values[affine_rank - 1]
    )

    isotropy_ratio = (
        singular_values[affine_rank - 1]
        / singular_values[0]
    )

    pca_scores = (
        U * singular_values
    )

    # ========================================================
    # STEP 4
    # Pairwise Euclidean distances / RDM
    # ========================================================

    distance_matrix = squareform(
        pdist(
            Lambda,
            metric="euclidean",
        )
    )

    upper = np.triu_indices(
        n_conditions,
        k=1,
    )

    pairwise_distances = (
        distance_matrix[upper]
    )

    distance_mean = (
        pairwise_distances.mean()
    )

    distance_sd = (
        pairwise_distances.std(ddof=0)
    )

    distance_cv = (
        distance_sd / distance_mean
    )

    distance_min = (
        pairwise_distances.min()
    )

    distance_max = (
        pairwise_distances.max()
    )

    distance_max_min_ratio = (
        distance_max / distance_min
    )

    # ========================================================
    # STEP 5
    # PCA visualization quantities
    # ========================================================

    pc1_pc2_variance = (
        variance_fraction[:2].sum()
    )

    pc1_pc2_pc3_variance = (
        variance_fraction[:3].sum()
    )

    # ========================================================
    # STEP 6
    # Gram matrix and ideal regular simplex
    # ========================================================

    gram_matrix = C @ C.T

    squared_radii = np.diag(
        gram_matrix
    )

    mean_radius_sq = (
        squared_radii.mean()
    )

    matched_radius = np.sqrt(
        mean_radius_sq
    )

    ideal_gram_matrix = np.full(
        (n_conditions, n_conditions),
        -mean_radius_sq
        / (n_conditions - 1),
    )

    np.fill_diagonal(
        ideal_gram_matrix,
        mean_radius_sq,
    )

    gram_difference = (
        gram_matrix
        - ideal_gram_matrix
    )

    relative_gram_error = (
        np.linalg.norm(
            gram_difference,
            ord="fro",
        )
        /
        np.linalg.norm(
            ideal_gram_matrix,
            ord="fro",
        )
    )

    diagonal_difference = np.diag(
        gram_difference
    )

    diagonal_rmse = np.sqrt(
        np.mean(
            diagonal_difference ** 2
        )
    )

    offdiag_mask = ~np.eye(
        n_conditions,
        dtype=bool,
    )

    offdiag_difference = (
        gram_difference[offdiag_mask]
    )

    offdiag_rmse = np.sqrt(
        np.mean(
            offdiag_difference ** 2
        )
    )

    total_squared_gram_error = np.sum(
        gram_difference ** 2
    )

    diagonal_squared_error = np.sum(
        diagonal_difference ** 2
    )

    offdiag_squared_error = (
        total_squared_gram_error
        - diagonal_squared_error
    )

    diagonal_error_fraction = (
        diagonal_squared_error
        / total_squared_gram_error
    )

    offdiag_error_fraction = (
        offdiag_squared_error
        / total_squared_gram_error
    )

    # ========================================================
    # STEP 7
    # Centroid / Gram angles
    # ========================================================

    centroid_angle_matrix = np.full(
        (n_conditions, n_conditions),
        np.nan,
    )

    for i in range(n_conditions):

        for j in range(n_conditions):

            denominator = (
                np.linalg.norm(C[i])
                * np.linalg.norm(C[j])
            )

            cosine = (
                np.dot(C[i], C[j])
                / denominator
            )

            cosine = np.clip(
                cosine,
                -1.0,
                1.0,
            )

            centroid_angle_matrix[i, j] = (
                np.degrees(
                    np.arccos(cosine)
                )
            )

    centroid_angles = (
        centroid_angle_matrix[upper]
    )

    centroid_angle_mean = (
        centroid_angles.mean()
    )

    centroid_angle_sd = (
        centroid_angles.std(ddof=0)
    )

    centroid_angle_min = (
        centroid_angles.min()
    )

    centroid_angle_max = (
        centroid_angles.max()
    )

    regular_centroid_angle = np.degrees(
        np.arccos(
            -1.0 / (n_conditions - 1)
        )
    )

    # ========================================================
    # STEP 8
    # Simplex volume and conditioning
    # ========================================================

    # Use number 1 as reference vertex
    edge_matrix = (
        Lambda[1:]
        - Lambda[0]
    )

    edge_singular_values = np.linalg.svd(
        edge_matrix,
        compute_uv=False,
    )

    edge_rank = np.linalg.matrix_rank(
        edge_matrix
    )

    edge_condition_number = (
        edge_singular_values[0]
        / edge_singular_values[-1]
    )

    simplex_volume = (
        np.prod(edge_singular_values)
        / factorial(8)
    )

    # --------------------------------------------------------
    # Matched regular 8-simplex volume
    # --------------------------------------------------------

    d = 8

    regular_edge_length = np.sqrt(
        2.0
        * mean_radius_sq
        * n_conditions
        / (n_conditions - 1)
    )

    regular_simplex_volume = (
        np.sqrt(d + 1)
        /
        (
            factorial(d)
            * np.sqrt(2.0 ** d)
        )
        *
        regular_edge_length ** d
    )

    normalized_volume = (
        simplex_volume
        / regular_simplex_volume
    )

    effective_rank_fraction = (
        effective_rank / d
    )

    participation_ratio_fraction = (
        participation_ratio / d
    )

    # ========================================================
    # STEP 9
    # Sequential transition angles
    # ========================================================

    transition_vectors = np.diff(
        Lambda,
        axis=0,
    )

    transition_lengths = np.linalg.norm(
        transition_vectors,
        axis=1,
    )

    transition_angles = []

    for k in range(
        len(transition_vectors) - 1
    ):

        a = transition_vectors[k]
        b = transition_vectors[k + 1]

        cosine = (
            np.dot(a, b)
            /
            (
                np.linalg.norm(a)
                * np.linalg.norm(b)
            )
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0,
        )

        angle = np.degrees(
            np.arccos(cosine)
        )

        transition_angles.append(angle)

    transition_angles = np.asarray(
        transition_angles
    )

    transition_angle_mean = (
        transition_angles.mean()
    )

    transition_angle_sd = (
        transition_angles.std(ddof=0)
    )

    transition_angle_min = (
        transition_angles.min()
    )

    transition_angle_max = (
        transition_angles.max()
    )

    # ========================================================
    # STEP 10A
    # Linear numerical distance
    # ========================================================

    numbers = np.arange(
        1,
        10,
        dtype=float,
    )

    numerical_distance_matrix = np.abs(
        numbers[:, None]
        - numbers[None, :]
    )

    numerical_distances = (
        numerical_distance_matrix[upper]
    )

    linear_pearson = pearsonr(
        numerical_distances,
        pairwise_distances,
    )

    linear_spearman = spearmanr(
        numerical_distances,
        pairwise_distances,
    )

    # Mean neural distance for each |i-j|
    mean_distance_by_separation = {}

    for separation in range(1, 9):

        mask = (
            numerical_distances
            == separation
        )

        values = (
            pairwise_distances[mask]
        )

        mean_distance_by_separation[
            separation
        ] = {
            "mean": values.mean(),
            "sd": (
                values.std(ddof=0)
                if len(values) > 1
                else np.nan
            ),
            "n": len(values),
        }

    # ========================================================
    # STEP 10B
    # Logarithmic numerical distance
    # ========================================================

    log_numbers = np.log(numbers)

    log_distance_matrix = np.abs(
        log_numbers[:, None]
        - log_numbers[None, :]
    )

    log_distances = (
        log_distance_matrix[upper]
    )

    log_pearson = pearsonr(
        log_distances,
        pairwise_distances,
    )

    log_spearman = spearmanr(
        log_distances,
        pairwise_distances,
    )

    # ========================================================
    # STEP 10C
    # Permutation test for RDM correlations
    # ========================================================

    rdm_perm = _rdm_permutation_test(
        pairwise_distances,
        n_perm=n_perm,
        random_state=random_state,
    )

    # ========================================================
    # STEP 10D
    # Magnitude-axis regression
    # ========================================================

    linear_magnitude = (
        _fit_magnitude_axis(
            C,
            numbers,
        )
    )

    log_magnitude = (
        _fit_magnitude_axis(
            C,
            np.log(numbers),
        )
    )

    linear_magnitude_r2 = (
        linear_magnitude["r2"]
    )

    log_magnitude_r2 = (
        log_magnitude["r2"]
    )

    linear_r2_vs_pc1 = (
        linear_magnitude_r2
        / pc1_variance
    )

    log_r2_vs_pc1 = (
        log_magnitude_r2
        / pc1_variance
    )

    # ========================================================
    # STEP 10E
    # Permutation test for magnitude-axis R²
    # ========================================================

    magnitude_perm = (
        _magnitude_permutation_test(
            C,
            n_perm=n_perm,
            random_state=random_state,
        )
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results = {

        # ----------------------------------------------------
        # Identification
        # ----------------------------------------------------

        "subject": subject,
        "n_conditions": n_conditions,
        "n_features": n_features,

        # ----------------------------------------------------
        # Raw / centered matrices
        # ----------------------------------------------------

        "Lambda": Lambda,
        "centroid": centroid,
        "C": C,

        # ----------------------------------------------------
        # Step 2: radii
        # ----------------------------------------------------

        "centroid_distances": centroid_distances,
        "radius_mean": radius_mean,
        "radius_sd": radius_sd,
        "radius_cv": radius_cv,

        # ----------------------------------------------------
        # Step 3: dimensionality
        # ----------------------------------------------------

        "affine_rank": affine_rank,
        "singular_values": singular_values,
        "variance_fraction": variance_fraction,
        "cumulative_variance": cumulative_variance,

        "pc1_variance": pc1_variance,

        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,

        "effective_rank_fraction": effective_rank_fraction,
        "participation_ratio_fraction": participation_ratio_fraction,

        "condition_number": symmetric_condition_number,
        "isotropy_ratio": isotropy_ratio,

        # ----------------------------------------------------
        # Step 4: pairwise distances
        # ----------------------------------------------------

        "distance_matrix": distance_matrix,
        "pairwise_distances": pairwise_distances,

        "distance_mean": distance_mean,
        "distance_sd": distance_sd,
        "distance_cv": distance_cv,

        "distance_min": distance_min,
        "distance_max": distance_max,
        "distance_max_min_ratio": distance_max_min_ratio,

        # ----------------------------------------------------
        # Step 5: PCA
        # ----------------------------------------------------

        "pca_scores": pca_scores,

        "pc1_pc2_variance": pc1_pc2_variance,
        "pc1_pc2_pc3_variance": pc1_pc2_pc3_variance,

        # ----------------------------------------------------
        # Step 6: Gram
        # ----------------------------------------------------

        "gram_matrix": gram_matrix,
        "ideal_gram_matrix": ideal_gram_matrix,

        "mean_radius_sq": mean_radius_sq,
        "matched_radius": matched_radius,

        "relative_gram_error": relative_gram_error,

        "diagonal_gram_rmse": diagonal_rmse,
        "offdiag_gram_rmse": offdiag_rmse,

        "diagonal_error_fraction": diagonal_error_fraction,
        "offdiag_error_fraction": offdiag_error_fraction,

        # ----------------------------------------------------
        # Step 7: centroid angles
        # ----------------------------------------------------

        "centroid_angle_matrix": centroid_angle_matrix,
        "centroid_angles": centroid_angles,

        "centroid_angle_mean": centroid_angle_mean,
        "centroid_angle_sd": centroid_angle_sd,
        "centroid_angle_min": centroid_angle_min,
        "centroid_angle_max": centroid_angle_max,

        "regular_centroid_angle": regular_centroid_angle,

        # ----------------------------------------------------
        # Step 8: volume
        # ----------------------------------------------------

        "edge_matrix": edge_matrix,
        "edge_rank": edge_rank,
        "edge_singular_values": edge_singular_values,

        "edge_condition_number": edge_condition_number,

        "simplex_volume": simplex_volume,
        "regular_simplex_volume": regular_simplex_volume,
        "normalized_volume": normalized_volume,

        # ----------------------------------------------------
        # Step 9: transition angles
        # ----------------------------------------------------

        "transition_vectors": transition_vectors,
        "transition_lengths": transition_lengths,
        "transition_angles": transition_angles,

        "transition_angle_mean": transition_angle_mean,
        "transition_angle_sd": transition_angle_sd,
        "transition_angle_min": transition_angle_min,
        "transition_angle_max": transition_angle_max,

        # ----------------------------------------------------
        # Step 10A-B: number-line RDM
        # ----------------------------------------------------

        "numerical_distances": numerical_distances,
        "log_distances": log_distances,

        "mean_distance_by_separation": mean_distance_by_separation,

        "linear_pearson_r": linear_pearson.statistic,
        "linear_pearson_naive_p": linear_pearson.pvalue,

        "linear_spearman_rho": linear_spearman.statistic,
        "linear_spearman_naive_p": linear_spearman.pvalue,

        "log_pearson_r": log_pearson.statistic,
        "log_pearson_naive_p": log_pearson.pvalue,

        "log_spearman_rho": log_spearman.statistic,
        "log_spearman_naive_p": log_spearman.pvalue,

        # ----------------------------------------------------
        # Step 10C: RDM permutation tests
        # ----------------------------------------------------

        "linear_pearson_perm_two_p":
            rdm_perm["linear_pearson_two_p"],

        "linear_spearman_perm_two_p":
            rdm_perm["linear_spearman_two_p"],

        "log_pearson_perm_two_p":
            rdm_perm["log_pearson_two_p"],

        "log_spearman_perm_two_p":
            rdm_perm["log_spearman_two_p"],

        "linear_pearson_perm_positive_p":
            rdm_perm["linear_pearson_positive_p"],

        "linear_spearman_perm_positive_p":
            rdm_perm["linear_spearman_positive_p"],

        "log_pearson_perm_positive_p":
            rdm_perm["log_pearson_positive_p"],

        "log_spearman_perm_positive_p":
            rdm_perm["log_spearman_positive_p"],

        # ----------------------------------------------------
        # Step 10D-E: magnitude axes
        # ----------------------------------------------------

        "linear_magnitude_r2": linear_magnitude_r2,
        "log_magnitude_r2": log_magnitude_r2,

        "linear_magnitude_r2_vs_pc1": linear_r2_vs_pc1,
        "log_magnitude_r2_vs_pc1": log_r2_vs_pc1,

        "linear_magnitude_perm_p":
            magnitude_perm["linear_p"],

        "log_magnitude_perm_p":
            magnitude_perm["log_p"],

        "linear_magnitude_null_mean":
            magnitude_perm["linear_null_mean"],

        "log_magnitude_null_mean":
            magnitude_perm["log_null_mean"],

        "linear_magnitude_null_95":
            magnitude_perm["linear_null_95"],

        "log_magnitude_null_95":
            magnitude_perm["log_null_95"],

        # ----------------------------------------------------
        # Optional detailed permutation distributions
        # ----------------------------------------------------

        "rdm_permutation_results": rdm_perm,
        "magnitude_permutation_results": magnitude_perm,
    }

    return results