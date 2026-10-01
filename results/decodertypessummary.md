# Exploratory Decoder Comparison: Results So Far

## Goal

After reproducing the paper's LDA-based number-decoding analysis, we asked whether the identification of number information depends on the choice of linear decoder.

We therefore compared three linear classifiers:

1. **Linear Discriminant Analysis (LDA)**
2. **Multinomial Logistic Regression**
3. **Linear Support Vector Machine (Linear SVM)**

We later performed a preliminary nonlinear comparison using:

4. **RBF-kernel Support Vector Machine (RBF-SVM)**

The purpose of these analyses is exploratory. They test the robustness of number decoding to classifier choice and do not themselves test the simplex hypothesis.

---

# 1. Linear Decoder Comparison

## Independent optimization

For the exploratory comparison, each linear decoder was optimized independently.

### LDA

For each neuron, LDA independently searched over:

- temporal bin size:
  - 60, 75, 90, 100, 150, 180, 225, 300, 450, 900 ms
- shrinkage parameter:
  - gamma = 0.2, 0.5, 0.8

### Logistic Regression

For each neuron, Logistic Regression independently searched over:

- the same temporal bin sizes
- regularization parameter:
  - C = 0.01, 0.1, 1, 10, 100

StandardScaler was fitted within the training folds.

### Linear SVM

For each neuron, Linear SVM independently searched over:

- the same temporal bin sizes
- regularization parameter:
  - C = 0.01, 0.1, 1, 10, 100

StandardScaler was fitted within the training folds.

Thus, the temporal representation selected by LDA was **not imposed on Logistic Regression or Linear SVM**.

All three classifiers were allowed to select their own preferred temporal representation and hyperparameter.

---

# 2. YFI: Linear Decoder Results

Number of analyzed neurons:

**N = 29**

| Decoder | Mean decoding accuracy | Median accuracy | Mean null accuracy | Coding neurons | Coding proportion |
|---|---:|---:|---:|---:|---:|
| LDA | 16.31% | 16.35% | 10.88% | 10/29 | 34.48% |
| Linear SVM | 16.98% | 17.31% | 10.75% | 14/29 | 48.28% |
| Logistic Regression | 16.81% | 17.31% | 10.66% | 15/29 | 51.72% |

### Observation

Mean decoding accuracy was relatively similar across the three linear classifiers:

- LDA: 16.31%
- Linear SVM: 16.98%
- Logistic Regression: 16.81%

However, the proportion of neurons classified as statistically significant number-coding neurons differed substantially:

- LDA: 34.48%
- Linear SVM: 48.28%
- Logistic Regression: 51.72%

Thus, similar average decoding accuracy did not imply identical classification of neurons as number-coding.

---

# 3. YFI: Decoder Overlap

The coding-neuron sets showed substantial overlap.

| Comparison | Number of neurons |
|---|---:|
| LDA ∩ Logistic | 9 |
| LDA ∩ SVM | 9 |
| Logistic ∩ SVM | 10 |
| All three | 8 |
| Any decoder | 19 |
| No decoder | 10 |

Therefore:

**8/29 neurons were classified as number-coding by all three linear decoders.**

Among the 10 neurons classified as coding by LDA:

**8/10 = 80%**

were also classified as coding by both Logistic Regression and Linear SVM.

At the same time:

**19/29 = 65.5%**

of YFI neurons were classified as coding by at least one decoder.

### Preliminary interpretation

The classifiers appear to recover a common core of strongly decodable neurons, while additional neurons near the classification boundary depend on the decoder used.

---

# 4. YFK: Linear Decoder Results

Number of analyzed neurons:

**N = 44**

| Decoder | Mean decoding accuracy | SEM | Median accuracy | Mean null accuracy | Coding neurons | Coding proportion |
|---|---:|---:|---:|---:|---:|---:|
| LDA | 15.91% | 0.60% | 15.79% | 10.99% | 17/44 | 38.64% |
| Linear SVM | 16.99% | 0.49% | 16.67% | 11.12% | 22/44 | 50.00% |
| Logistic Regression | 16.29% | 0.45% | 16.67% | 10.45% | 22/44 | 50.00% |

### Observation

Again, the mean decoding accuracies were relatively similar:

- LDA: 15.91%
- Linear SVM: 16.99%
- Logistic Regression: 16.29%

Linear SVM had the highest mean accuracy for this subject, approximately 1.08 percentage points above LDA.

However, the coding proportions again differed:

- LDA: 38.64%
- Linear SVM: 50.00%
- Logistic Regression: 50.00%

Thus, both alternative linear classifiers identified five more coding neurons than LDA in YFK.

---

# 5. YFK: Decoder Overlap

| Comparison | Number of neurons |
|---|---:|
| LDA ∩ Logistic | 16 |
| LDA ∩ SVM | 14 |
| Logistic ∩ SVM | 16 |
| All three | 14 |
| Any decoder | 29 |
| No decoder | 15 |

The exact decomposition is:

| Coding pattern | Number of neurons |
|---|---:|
| All three | 14 |
| LDA + Logistic only | 2 |
| LDA + SVM only | 0 |
| Logistic + SVM only | 2 |
| LDA only | 1 |
| Logistic only | 4 |
| SVM only | 6 |
| None | 15 |

Therefore:

**14/17 = 82.4%**

of LDA's coding neurons were also classified as coding by both alternative linear classifiers.

At the same time:

**29/44 = 65.9%**

of YFK neurons were classified as coding by at least one decoder.

### Preliminary interpretation

As in YFI, YFK contains a substantial core population that is consistently identified by all three decoders.

However, outside this common core, coding classification depends on decoder choice.

---

# 6. Cross-Subject Pattern So Far

The first two completed subjects show a similar qualitative pattern.

| Subject | N | LDA coding | Linear SVM coding | Logistic coding |
|---|---:|---:|---:|---:|
| YFI | 29 | 34.48% | 48.28% | 51.72% |
| YFK | 44 | 38.64% | 50.00% | 50.00% |

For both subjects:

**Logistic Regression and Linear SVM classified a larger proportion of neurons as number-coding than LDA.**

However, this should not yet be interpreted as evidence that Logistic Regression or Linear SVM is a superior decoder.

The three classifiers also showed substantial overlap in the neurons they identified.

A more conservative current interpretation is:

> The three linear decoders recover a common core of number-decodable neurons, while the binary classification of some additional neurons as "number-coding" is sensitive to decoder choice.

---

# 7. Important Statistical Caveat

The current subject-level comparison is exploratory.

The hyperparameter search spaces are not identical in size:

- LDA:
  - 10 temporal bins × 3 gamma values
  - 30 candidate configurations

- Logistic Regression:
  - 10 temporal bins × 5 C values
  - 50 candidate configurations

- Linear SVM:
  - 10 temporal bins × 5 C values
  - 50 candidate configurations

Furthermore, the current permutation procedure freezes the observed selected temporal bin and hyperparameter during permutation rather than repeating the complete model-selection procedure for every shuffled dataset.

Therefore, the larger number of candidate configurations available to Logistic Regression and Linear SVM could contribute to differences in the number of neurons crossing the p < 0.05 threshold.

For this reason, the current results demonstrate **decoder sensitivity**, but they do not yet establish that one decoder is statistically superior to another.

---

# 8. Nonlinear Decoder Experiment

To investigate whether a nonlinear decision boundary reveals additional number information, we performed a preliminary comparison on the example neuron:

**YFU.hpc.43**

Four classifiers were compared:

1. LDA
2. Logistic Regression
3. Linear SVM
4. nonlinear RBF-SVM

Unlike the earlier paper-style analysis, this comparison used **nested cross-validation**.

---

# 9. Why Nested Cross-Validation Was Used

The nonlinear RBF-SVM has a substantially larger hyperparameter search space.

For example:

- temporal bin size
- C
- RBF gamma

were all selected independently.

Therefore, simply reporting the best cross-validation accuracy after searching many combinations could favor the more flexible model.

Nested cross-validation separates:

**Inner CV**

Model and representation selection:

- temporal bin
- regularization
- kernel parameters

from:

**Outer CV**

Evaluation on held-out observations that were not involved in model selection.

Thus:

**inner CV = model selection**

**outer CV = generalization evaluation**

Each of the four classifiers independently performed its own model-selection procedure inside the outer training data.

The outer test observations were shared across classifiers only to provide a fair comparison.

---

# 10. YFU.hpc.43: Linear vs Nonlinear Decoding

Number of presentations:

**N = 494**

Nominal 9-class chance accuracy:

**11.11%**

| Decoder | Type | Correct | Total | Nested-CV accuracy |
|---|---|---:|---:|---:|
| LDA | Linear | 75 | 494 | 15.18% |
| Logistic Regression | Linear | 76 | 494 | 15.38% |
| Linear SVM | Linear | 69 | 494 | 13.97% |
| RBF-SVM | Nonlinear | 56 | 494 | 11.34% |

### Result

Logistic Regression and LDA produced very similar nested-CV performance:

- Logistic Regression: 15.38%
- LDA: 15.18%

Linear SVM produced:

- 13.97%

The nonlinear RBF-SVM produced:

- 11.34%

which was close to the nominal 9-class chance level of 11.11%.

Therefore, for this particular neuron, introducing a nonlinear RBF decision boundary did **not** improve held-out decoding performance.

---

# 11. Relation to Previous YFU.hpc.43 LDA Result

Our earlier paper-style LDA analysis of YFU.hpc.43 gave approximately:

**16.40% decoding accuracy**

with:

- temporal bin = 150 ms
- gamma = 0.2

The nested-CV LDA analysis gave:

**15.18%**

These values should not be expected to be identical because they answer slightly different questions.

The earlier procedure selected the best representation/model using the cross-validation analysis and reported its performance.

The nested procedure separates model selection from final held-out evaluation.

Therefore, the nested-CV estimate is a stricter estimate of generalization performance.

---

# 12. Current Interpretation

The results so far suggest three main observations.

### Observation 1: Linear decoder accuracies are broadly similar

Across YFI and YFK, LDA, Logistic Regression, and Linear SVM generally produced comparable average decoding accuracies.

There is not yet evidence from these analyses that one linear decoder universally performs best.

### Observation 2: Coding classification is decoder-sensitive

Although mean decoding accuracies are relatively similar, the number of neurons crossing the permutation-test significance threshold differs across classifiers.

In both YFI and YFK, Logistic Regression and Linear SVM classified more neurons as number-coding than LDA.

However, substantial overlap exists between the coding sets.

This suggests the existence of:

- a decoder-robust core of coding neurons
- additional decoder-sensitive neurons near the classification boundary

### Observation 3: Nonlinear decoding did not help YFU.hpc.43

For the single example neuron YFU.hpc.43, nonlinear RBF-SVM decoding did not outperform the linear classifiers.

RBF-SVM performance was approximately at nominal chance.

This result is specific to this neuron and should not be generalized to the population.

---

# 13. Relation to the Simplex Question

These decoder analyses should be kept conceptually separate from the simplex analysis.

Decoder performance asks:

> How accessible is number information to a particular readout?

The simplex analysis asks:

> What is the geometry of the population-level representations of the nine numbers?

Therefore:

**better linear decoding does not prove a simplex**

and

**poor nonlinear decoding does not prove a simplex**

The simplex claim must instead be tested directly using population geometry, including analyses such as:

- affine dimensionality / affine independence
- centroid geometry
- convex-hull membership
- whether all nine centroids behave as vertices
- distance structure
- comparison with alternative polytopes
- Gaussian-centroid controls
- shattering dimensionality and related analyses from Figure 2

A later population-level comparison between linear and nonlinear decoding may nevertheless be informative about whether number information is primarily accessible through linear readouts.

---

# 14. Current Status

Completed exploratory analyses:

- [x] LDA single-neuron decoding
- [x] Logistic Regression single-neuron decoding
- [x] Linear SVM single-neuron decoding
- [x] Independent temporal-bin optimization for each linear decoder
- [x] YFI subject-level comparison
- [x] YFI coding-set overlap
- [x] YFK subject-level comparison
- [x] YFK coding-set overlap
- [x] Preliminary nonlinear RBF-SVM test
- [x] Nested-CV four-decoder comparison for YFU.hpc.43

Still to investigate:

- [ ] Additional subjects
- [ ] Cross-subject summary
- [ ] Paired neuron-by-neuron decoder comparisons
- [ ] Selection-aware permutation testing if required
- [ ] Population-level linear vs nonlinear decoding
- [ ] Figure 2 population geometry
- [ ] Shattering dimensionality
- [ ] Direct simplex-vs-polytope tests
- [ ] Gaussian-centroid controls

---

## Main Takeaway So Far

**LDA, Logistic Regression, and Linear SVM identify a substantial common core of number-decodable neurons, but the exact proportion of neurons classified as number-coding is sensitive to decoder choice.**

**For YFU.hpc.43, independently optimized nonlinear RBF-SVM decoding did not improve generalization over the linear decoders.**

These observations motivate further robustness analyses but do not, by themselves, establish or reject the proposed number-simplex geometry.