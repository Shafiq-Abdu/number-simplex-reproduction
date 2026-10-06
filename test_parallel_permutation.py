import numpy as np

from src.number_decoding_analysis import permutation_test_decoding


def main():

    rng = np.random.default_rng(2026)

    # 9 classes, 20 observations per class
    y = np.repeat(np.arange(1, 10), 20)

    # Simple synthetic neural features
    X = rng.normal(size=(len(y), 6))

    observed_accuracy = 0.20

    common_args = dict(
        X=X,
        y=y,
        observed_accuracy=observed_accuracy,
        gamma=0.5,
        n_permutations=20,
        n_splits=5,
        cv_random_state=42,
        permutation_seed=12345,
    )

    print("Running sequential...")
    sequential = permutation_test_decoding(
        **common_args,
        n_jobs=1,
    )

    print("Running parallel...")
    parallel = permutation_test_decoding(
        **common_args,
        n_jobs=4,
    )

    same_null = np.array_equal(
        sequential["null_accuracies"],
        parallel["null_accuracies"],
    )

    print()
    print("Same null distribution:", same_null)
    print("Sequential p-value:", sequential["p_value"])
    print("Parallel p-value:  ", parallel["p_value"])


if __name__ == "__main__":
    main()