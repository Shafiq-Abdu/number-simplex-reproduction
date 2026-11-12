
# Synthetic Geometry Recovery Analysis

**Date:** November 12, 2026  
**Project:** Number Simplex Reproduction  
**Dataset reference:** YFU arithmetic, human medial temporal lobe  
**Objective:** Evaluate whether supervised temporal LDA recovers the ground-truth geometry of numerical representations.

---

## 1. Research Question

Does supervised temporal Linear Discriminant Analysis (LDA) faithfully recover the underlying geometry of numerical representations, or can the pipeline produce apparent high-dimensional structure when the true representation is lower-dimensional?

We evaluate seven synthetic ground-truth geometries:

1. Linear number line (1D)
2. Logarithmic number line (1D)
3. Curved representation (2D)
4. Regular simplex (8D)
5. Irregular simplex (8D)
6. Random high-dimensional representation (8D)
7. Null representation (0D)

---

## 2. Experimental Setup

| Parameter | Value |
|---|---|
| Numerical classes | 9 (numbers 1–9) |
| Synthetic neurons | 62 |
| Presentations | 494 |
| Observation window | 900 ms |
| Spike generation | Poisson model |
| Firing-rate reference | Real YFU data |
| Classifier | Temporal LDA |
| Cross-validation | Stratified 10-fold |
| Permutations | 99 |
| Neuron-selection threshold | p ≤ 0.125 |
| Maximum LDA components | 3 per selected neuron |
| Simulation replicates | 1 per model |
| Random seed | 2026 |
| Signal-strength parameter | 0.6 |

The synthetic datasets were generated using different known geometries while retaining firing-rate characteristics calibrated to the YFU data.

---

## 3. Ground Truth vs Recovered Geometry

| Model | True Affine Rank | Raw Affine Rank | LDA Affine Rank | LDA Effective Rank | PC1 Variance |
|---|---:|---:|---:|---:|---:|
| Linear | 1 | 8 | 8 | 2.678 | 76.72% |
| Logarithmic | 1 | 8 | 8 | 3.307 | 69.44% |
| Curved | 2 | 8 | 8 | 3.821 | 56.39% |
| Regular simplex | 8 | 8 | 8 | 7.803 | 16.60% |
| Irregular simplex | 8 | 8 | 8 | 7.063 | 25.20% |
| Random high-dimensional | 8 | 8 | 8 | 6.480 | 26.93% |
| Null | 0 | 8 | 8 | 6.895 | 23.44% |

**Important:** The raw and LDA affine ranks are numerical ranks of estimated class centroids. All seven configurations have numerical rank 8, including those generated from lower-dimensional ground truths.

---

## 4. Neuron Selection

| Model | Selected Neurons | Total Neurons | Selection Percentage |
|---|---:|---:|---:|
| Linear | 52 | 62 | 83.87% |
| Logarithmic | 49 | 62 | 79.03% |
| Curved | 49 | 62 | 79.03% |
| Regular simplex | 54 | 62 | 87.10% |
| Irregular simplex | 55 | 62 | 88.71% |
| Random high-dimensional | 54 | 62 | 87.10% |
| Null | 6 | 62 | 9.68% |

The null model selected six neurons despite having no number-specific tuning by construction.

---

## 5. Effective Rank Before and After LDA

The saved JSON results contain three geometric representations:

- `true_mean_centroid_metrics`: Geometry of the generating population means.
- `raw_temporal_centroids`: Geometry estimated from synthetic spike data before LDA.
- `lda_centroids_in_sample`: Geometry after supervised temporal LDA.

For the linear model:

| Measurement | Effective Rank |
|---|---:|
| Ground-truth centroids | 1.000 |
| Raw temporal centroids | 7.351 |
| LDA centroids | 2.678 |

For the null model:

| Measurement | Effective Rank |
|---|---:|
| Ground-truth centroids | Undefined (zero signal) |
| Raw temporal centroids | 7.945 |
| LDA centroids | 6.895 |

The null model's saved ground-truth effective rank of 1 is a numerical artifact because its singular values are effectively zero.

---

## 6. Interpretation

### Finding 1 — Affine Independence Does Not Establish Simplex-Specific Geometry

All recovered configurations have affine rank 8.

This occurs even when the ground truth is a 1D line, 2D curve, or null representation.

However, the raw population centroids already have affine rank 8 before applying LDA.

Therefore, the results do not establish that LDA itself causes affine independence.

### Finding 2 — LDA Retains Some Ground-Truth Dimensional Structure

For the linear model, the recovered effective rank is 2.678, and the first principal component explains 76.72% of the variance.

For the regular simplex, the recovered effective rank is 7.803, and PC1 explains only 16.60%.

This suggests that the pipeline retains information about the dominant dimensional structure.

### Finding 3 — High Effective Rank Is Not Unique to Simplex Representations

The random high-dimensional model has recovered effective rank 6.480.

The null model has recovered effective rank 6.895.

Therefore, high effective rank alone cannot establish the presence of meaningful simplex-like numerical geometry.

### Finding 4 — Supervised LDA May Alter the Apparent Geometry

For the linear model, effective rank decreases from 7.351 before LDA to 2.678 after LDA.

This indicates that LDA concentrates much of the recovered variance into a dominant direction.

However, these LDA centroids were estimated in-sample, so supervised fitting and selection bias remain potential concerns.

---

## 7. Preliminary Conclusion

The geometry-recovery experiment provides evidence that:

1. Numerical affine rank is insufficient to distinguish the seven synthetic ground-truth models.
2. Finite-sample noise can produce affinely independent estimated class centroids.
3. Effective rank retains partial information about the underlying geometry.
4. Random high-dimensional and null representations can also exhibit high effective rank.
5. Additional validation is required to determine whether the pipeline recovers simplex-specific geometry.

**Central conclusion:**

> A recovered affine rank of eight does not, by itself, establish that numerical representations form a simplex with a particular geometric structure.

These findings are preliminary because each model was simulated only once.

---

## 8. Recommended Next Analyses

- Compare ground-truth and recovered pairwise distance matrices.
- Calculate simplex edge-length variability and geometric regularity.
- Perform split-half or held-out geometry recovery.
- Estimate noise-corrected signal dimensionality.
- Repeat simulations across multiple random seeds.
- Compare supervised LDA with unsupervised PCA and raw population representations.
- Calculate the effective rank of the original YFF neural population LDA centroids.

---

## 9. Output Files

Original results are stored on the UH server at:

`~/geometry_recovery/full_results/`

Each model contains:

- `rep_000/summary.json`
- `rep_000/metadata.json`
- `rep_000/neuron_selection.csv`

Simulation logs:

`~/geometry_recovery/logs/`

**Note:** This Markdown report preserves the principal numerical findings and interpretations, not every individual neuron-level result or the full precision of all singular values.
