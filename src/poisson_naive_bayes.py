import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


CLASSES = np.arange(1, 10)


def fit_poisson_nb(
    X_train,
    y_train,
    classes=CLASSES,
    uniform_prior=True,
    min_rate=1e-9,
):
    """
    Fit a Poisson Naive Bayes classifier.

    Parameters
    ----------
    X_train : array, shape (n_samples, n_features)
        Raw non-negative spike counts.

    y_train : array, shape (n_samples,)
        Number labels 1,...,9.

    classes : array
        Class labels.

    uniform_prior : bool
        If True, use P(Y=k)=1/K.

    min_rate : float
        Tiny numerical floor to avoid log(0).

    Returns
    -------
    model : dict
        Contains classes, Poisson rates, and class priors.
    """

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    if np.any(X_train < 0):
        raise ValueError(
            "Poisson Naive Bayes requires non-negative spike counts."
        )

    n_classes = len(classes)
    n_features = X_train.shape[1]

    rates = np.zeros(
        (n_classes, n_features),
        dtype=float,
    )

    priors = np.zeros(
        n_classes,
        dtype=float,
    )

    for i, k in enumerate(classes):

        X_k = X_train[y_train == k]

        if len(X_k) == 0:
            raise ValueError(
                f"Class {k} is missing from training data."
            )

        # MLE Poisson rate for every temporal bin
        rates[i] = X_k.mean(axis=0)

        # Numerical floor only
        rates[i] = np.maximum(
            rates[i],
            min_rate,
        )

        if uniform_prior:
            priors[i] = 1.0 / n_classes
        else:
            priors[i] = len(X_k) / len(X_train)

    return {
        "classes": np.asarray(classes),
        "rates": rates,
        "priors": priors,
    }


def predict_log_proba_poisson_nb(model, X):
    """
    Compute posterior log-probabilities for every class.
    """

    X = np.asarray(X, dtype=float)

    classes = model["classes"]
    rates = model["rates"]
    priors = model["priors"]

    log_scores = []

    for i, k in enumerate(classes):

        lam = rates[i]

        # Poisson log likelihood:
        #
        # sum_j [
        #     x_j log(lambda_j)
        #     - lambda_j
        #     - log(x_j!)
        # ]
        #
        # log(x_j!) is identical across classes,
        # so it cancels when normalizing posterior probabilities.

        log_likelihood = np.sum(
            X * np.log(lam) - lam,
            axis=1,
        )

        log_prior = np.log(priors[i])

        log_scores.append(
            log_prior + log_likelihood
        )

    log_scores = np.column_stack(log_scores)

    # Convert unnormalized log posterior
    # to normalized log posterior
    log_posterior = (
        log_scores
        -
        logsumexp(
            log_scores,
            axis=1,
            keepdims=True,
        )
    )

    return log_posterior


def predict_proba_poisson_nb(model, X):
    """
    Posterior probabilities:
        P(Y=k | x)
    """

    return np.exp(
        predict_log_proba_poisson_nb(
            model,
            X,
        )
    )


def predict_poisson_nb(model, X):
    """
    Maximum-posterior class prediction.
    """

    probabilities = predict_proba_poisson_nb(
        model,
        X,
    )

    best = np.argmax(
        probabilities,
        axis=1,
    )

    return model["classes"][best]



def cv_poisson_nb(
    X,
    y,
    groups=None,
    n_splits=5,
    grouped=False,
    random_state=42,
):
    """
    Cross-validated Poisson Naive Bayes decoding.

    Returns pooled held-out predictions and posterior probabilities.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if grouped:

        if groups is None:
            raise ValueError(
                "groups must be supplied for grouped CV."
            )

        groups = np.asarray(groups)

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        splits = cv.split(
            X,
            y,
            groups=groups,
        )

    else:

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        splits = cv.split(
            X,
            y,
        )

    y_true_all = []
    y_pred_all = []
    posterior_all = []
    index_all = []

    for fold, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = fit_poisson_nb(
            X_train,
            y_train,
            uniform_prior=True,
        )

        posterior = predict_proba_poisson_nb(
            model,
            X_test,
        )

        y_pred = model["classes"][
            np.argmax(
                posterior,
                axis=1,
            )
        ]

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        posterior_all.append(posterior)
        index_all.append(test_idx)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    posterior_all = np.vstack(
        posterior_all
    )

    index_all = np.concatenate(
        index_all
    )

    accuracy = np.mean(
        y_true_all == y_pred_all
    )

    return {
        "accuracy": accuracy,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "posterior": posterior_all,
        "indices": index_all,
    }