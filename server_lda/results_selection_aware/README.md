# Selection-aware LDA permutation results

Date: 2026-11-18

Optimized single-neuron LDA with 200 selection-aware permutations per neuron.
Benjamini-Hochberg FDR correction applied across all 364 neurons.

| Subject | Neurons | Nominal p < 0.05 | p <= 0.125 | FDR q < 0.05 |
|---|---:|---:|---:|---:|
| YFM | 129 | 3 | 11 | 0 |
| YFS | 144 | 9 | 18 | 0 |
| YFT | 91 | 5 | 11 | 0 |

- Minimum p-value: 0.00497512
- Minimum FDR q-value: 0.51741294
- Earlier selection-unaware nominal significant neurons: 90/364.
- Selection-aware nominal significant neurons: 17/364.
- FDR significant neurons: 0/364.

Interpretation: No single neuron survives FDR correction in this analysis.
This does not rule out distributed population-level number representations.

See the accompanying CSV and per-neuron NPZ files for detailed results.
