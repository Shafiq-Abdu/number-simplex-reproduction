import argparse
from pathlib import Path

from src.poisson_population_analysis import (
    analyze_population_subject,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "subject",
        type=str,
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    if args.root is None:
        root = Path(__file__).resolve().parent
    else:
        root = Path(args.root)

    analyze_population_subject(
        subject=args.subject.upper(),
        root=root,
        bin_size=100,
        n_splits=5,
        n_permutations=args.n_permutations,
        cv_random_state=42,
        permutation_seed=12345,
    )


if __name__ == "__main__":
    main()