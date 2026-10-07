import sys
from pathlib import Path

from src.lda_vs_logistic import compare_lda_logistic_subject


# Repository root
ROOT = Path(__file__).resolve().parent


if len(sys.argv) != 2:
    print(
        "Usage: python run_lda_vs_logistic_subject.py SUBJECT"
    )
    print(
        "Example: python run_lda_vs_logistic_subject.py YFF"
    )
    sys.exit(1)


subject = sys.argv[1].upper()

valid_subjects = {
    "YFF", "YFI", "YFJ", "YFK", "YFL",
    "YFM", "YFP", "YFR", "YFS", "YFT", "YFU",
}

if subject not in valid_subjects:
    raise ValueError(
        f"Unknown subject: {subject}"
    )


print("=" * 60)
print("LDA vs Logistic Regression")
print(f"Starting subject: {subject}")
print("=" * 60)


results = compare_lda_logistic_subject(
    subject=subject,
    root=ROOT,
    output_dir=ROOT / "tables1",
    n_splits=10,
    n_shuffles=200,
    random_state=42,
    shuffle_seed=12345,
    mtl_only=True,
)


print("\nDone.")
print(f"Subject: {subject}")
print(f"Analyzed neurons: {len(results)}")