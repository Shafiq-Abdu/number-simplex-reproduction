import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold


def lda_cv_accuracy(X, y, gamma, n_splits):
    """
    Reproduce the original notebook's single-neuron LDA CV accuracy.

    X: presentations x temporal bins
    y: number labels (1 through 9)
    gamma: LDA shrinkage
    n_splits: number of stratified CV folds
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    predictions = np.full(len(y), -1, dtype=int)

    for train_idx, test_idx in cv.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]

        within_variance = np.zeros(X.shape[1])

        for number in np.unique(y_train):
            class_X = X_train[y_train == number]
            within_variance += (
                class_X.var(axis=0) * len(class_X)
            )

        keep = within_variance > 0

        if not np.any(keep):
            predictions[test_idx] = 1
            continue

        lda = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage=gamma,
            priors=np.ones(9) / 9,
        )

        lda.fit(X_train[:, keep], y_train)
        predictions[test_idx] = lda.predict(
            X_test[:, keep]
        )

    return float(np.mean(predictions == y))


def optimized_lda_accuracy(X_by_bin, y, n_splits, gammas):
    """
    Search all bin-size/shrinkage combinations.

    X_by_bin: dictionary mapping bin size to feature matrix
              for one neuron.
    """
    best_accuracy = -1.0
    best_bin = None
    best_gamma = None

    for bin_size, X in X_by_bin.items():
        for gamma in gammas:
            accuracy = lda_cv_accuracy(X, y, gamma, n_splits)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_bin = bin_size
                best_gamma = gamma

    return best_accuracy, best_bin, best_gamma
