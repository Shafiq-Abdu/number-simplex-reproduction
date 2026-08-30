# Figure 1 Arithmetic Analysis — Reproduction Summary

## 1. Goal
To reproduce the **arithmetic-task analyses in Figure 1** of the number-simplex paper using the provided raw neural and behavioral data.

The main questions were:

1. Can arithmetic numbers 1–9 be decoded from single-neuron activity?
2. Does the **temporal spike pattern** contain more number information than the conventional **firing-rate (FR) code**?
3. What percentage of neurons are significantly number-coding?
4. Does number coding differ across MTL regions?
5. Do coding neurons systematically prefer particular numbers?
6. How much does temporal decoding improve single-neuron decoding accuracy relative to firing-rate decoding?


---

# 2. Dataset

The raw dataset contains **849 neurons** across 11 subjects:

- YFF
- YFI
- YFJ
- YFK
- YFL
- YFM
- YFP
- YFR
- YFS
- YFT
- YFU

The paper's Figure 1 analysis uses only neurons from the **medial temporal lobe (MTL)**.

The relevant regions were:

| Raw dataset label | Region |
|---|---|
| `hpc` | Hippocampus (HPC) |
| `ent` | Entorhinal cortex (ENT) |
| `amy` | Amygdala (AMY) |
| `para-hpc` | Parahippocampal cortex |

After filtering to these regions:

| Region | Neurons |
|---|---:|
| HPC | 389 |
| ENT | 70 |
| AMY | 77 |
| PARA-HPC | 18 |
| **Total** | **554** |

This exactly matches the paper's arithmetic MTL population:

554 neurons

---

# 3. Arithmetic Trial Construction

For Figures 1–3, the analysis is based on the **presented operands**, not the participant's final answer.

Only operands with values

$$
1,2,...,9
$$

were retained because the classifier is a **9-way number classifier**.

Values 0 and 10–16 were excluded.

Both operands were pooled.

For example:

- `2 + 1` contributes one sample labeled `2` and another labeled `1`.
- `5 - 16` contributes only the sample labeled `5`.

Across all 11 subjects this produced:

2328 operand presentations

Incorrect trials were retained because the paper states that Figures 1–3 analyze stimulus-defined representations rather than response-corrected representations.


---

# 4. Important Cue-Timing Issue

A major preprocessing issue was determining the actual onset of each operand.

For most subjects:

### `operationFirst = 1`

- Cue1 = operation
- Cue2 = operand 1
- Cue3 = operand 2

### `operationFirst = 0`

- Cue1 = operand 1
- Cue2 = operand 2
- Cue3 = operation

For subjects **YFR and YFS**, the operation was always Cue2:

- Cue1 = operand 1
- Cue2 = operation
- Cue3 = operand 2

This information came from clarification from the first author.

Therefore, the behavioral columns `cue1` and `cue2` represent operand values, but their actual presentation times must be reconstructed using `tCue1`, `tCue2`, `tCue3`, and `operationFirst`.


---

# 5. Neural Analysis Window

For arithmetic decoding, the paper analyzes:


0.05-0.95seconds(50 ms - 950 ms)


after operand onset.

Therefore each neural response contains:

900 ms(0.9 s)

of activity.

The raw behavioral timestamps and spike matrices were already aligned in millisecond/sample units, so the analysis window was implemented as:

```python
start_sample = onset_sample + 50
end_sample   = onset_sample + 950
```


---

# 6. Temporal Code

The temporal code divides the 900-ms response into smaller time bins.

Candidate bin sizes:

```text
60, 75, 90, 100, 150, 180, 225, 300, 450, 900 ms
```

For example:

| Bin size | Number of temporal features |
|---:|---:|
| 60 ms | 15 |
| 90 ms | 10 |
| 300 ms | 3 |
| 900 ms | 1 |

The features are **raw spike counts within each temporal bin**.

Thus, unlike the firing-rate representation, the temporal representation retains information about **when spikes occurred within the 900-ms response window**.


---

# 7. Classifier

A **9-way Linear Discriminant Analysis (LDA)** classifier was reconstructed.

The classes were:

$$
1,2,...,9.
$$

A uniform class prior was used:

$$
P(k)=\frac{1}{9}.
$$

The paper tested three MATLAB `fitcdiscr` Gamma values:

```text
Gamma = 0.2, 0.5, 0.8
```

The regularized covariance was implemented as:

$$
\Sigma_{\gamma}
=
(1-\gamma)\Sigma
+
\gamma\,\mathrm{diag}(\Sigma).
$$

For each neuron, every combination of:

- temporal bin size
- Gamma

was evaluated.

The combination giving the highest cross-validated decoding accuracy was selected for that neuron.


---

# 8. Cross-Validation

The paper requests stratified 10-fold cross-validation.

If there were too few examples of a number to place that number into every test fold, the number of folds was reduced.

Therefore:

```python
n_splits = min(10, minimum_class_count)
```

For example, YFF had only 7 presentations for its least frequent number, so YFF used:

$$
\boxed{7\text{-fold stratified CV}}
$$

Decoding accuracy was calculated from all pooled held-out predictions:

$$
\text{Accuracy}
=
\frac{\text{total correct held-out predictions}}
{\text{total held-out predictions}}.
$$

This is preferable to simply averaging fold accuracies because folds can contain slightly different numbers of observations.


---

# 9. Permutation Test — Number-Coding Neurons

After selecting the best bin size and Gamma for a neuron, statistical significance was assessed using:

$$
\boxed{200\text{ label permutations}}
$$

Importantly, the bin size and Gamma were **not re-optimized during every permutation**.

The paper's permutation p-value convention was reproduced:

$$
p=
\frac{
1+\#(\text{permuted accuracy}\geq\text{observed accuracy})
}{
1+200
}.
$$

Therefore:

$$
p=
\frac{
1+\#(\text{permuted accuracy}\geq\text{observed accuracy})
}{
201
}.
$$

The smallest possible p-value is:

$$
\frac{1}{201}
=
0.004975.
$$

A neuron was classified as number-coding when:

$$
\boxed{p<0.05}.
$$


---

# 10. Temporal Coding Result

Our temporal analysis found:

- Total MTL neurons = 554
- Temporal coding neurons = 160

Using the complete 554-neuron MTL population as the denominator, the mean subject-level temporal coding percentage was:

$$
\boxed{28.61\pm1.38\%}
$$

compared with the paper:

$$
\boxed{24.80\pm1.73\%}.
$$

Therefore our coding percentage is somewhat higher than the reported value.

However, the qualitative conclusion is reproduced:

> A substantial population of MTL neurons carries arithmetic-number information in temporal spike patterns.


---

# 11. Firing-Rate Code

The firing-rate representation reduces the entire 900-ms response to one feature:

$$
\text{FR feature}
=
\text{total spikes during }0.05-0.95\text{ s}.
$$

This is equivalent to using a single:

$$
900\text{-ms bin}.
$$

Our result was:

$$
\boxed{4.62\pm0.84\%}
$$

number-coding neurons across subjects.

The paper reports:

$$
\boxed{3.36\pm0.71\%}.
$$

Again, our percentage is somewhat higher, but the same major result is obtained:

> Far fewer neurons are identified as number-coding when neural activity is reduced to a single fixed-window firing-rate value.


---

# 12. Temporal vs Firing-Rate Coding Proportions

The temporal and FR coding percentages were compared across the 11 subjects using a **paired t-test**.

### Why a paired t-test?

Every subject contributes two related measurements:

1. temporal coding percentage;
2. firing-rate coding percentage.

Therefore the measurements are naturally paired by subject.

Our result was:

$$
\boxed{t(10)=17.94}
$$

with:

$$
\boxed{p=6.20\times10^{-9}}.
$$

The paper reports:

$$
p<0.0001.
$$

Thus both analyses strongly support:

$$
\boxed{\text{Temporal coding detects substantially more number-coding neurons than FR coding.}}
$$


---

# 13. Regional Specificity

We next asked whether temporal number coding differed between MTL regions.

Our descriptive results were:

| Region | Total neurons | Coding neurons | Coding % |
|---|---:|---:|---:|
| HPC | 389 | 113 | 29.05 |
| ENT | 70 | 19 | 27.14 |
| AMY | 77 | 23 | 29.87 |
| PARA-HPC | 18 | 5 | 27.78 |

The proportions are visually very similar across the four regions.


## Statistical Test

The paper uses a **binomial generalized linear mixed-effects model (GLMM)**:

```matlab
IsTuned ~ Region + (1 | Subject)
```

where:

- `IsTuned` = coding / non-coding;
- `Region` = fixed effect;
- `Subject` = random intercept.

### Why use a mixed model?

Multiple neurons were recorded from the same subject.

Therefore neurons from the same person should not automatically be treated as completely independent observations.

The random subject intercept accounts for this subject-level grouping.

We exported all 554 neuron-level observations to MATLAB and used:

```matlab
fitglme
```

followed by:

```matlab
anova(glme)
```

Our result was:

$$
\boxed{F(3,550)=0.052,\quad p=0.984}
$$

while the paper reports:

$$
\boxed{F(3,550)=0.43,\quad p=0.74}.
$$

The exact statistics differ, but both clearly conclude:

$$
\boxed{\text{No significant regional difference in arithmetic number coding.}}
$$


---

# 14. Preferred Number Analysis

For each temporal number-coding neuron, decoding accuracy was calculated separately for each true number.

The neuron's **preferred number** was defined as the number with the highest class-specific decoding accuracy.

Among the 160 temporal coding neurons:

- 156 had one unique maximum;
- 4 had tied maxima.

Thus the primary preferred-number analysis used:

$$
\boxed{156\text{ uniquely assigned neurons}}.
$$

The overall preferred-number counts were:

| Preferred number | Neurons | Percentage |
|---:|---:|---:|
| 1 | 21 | 13.46% |
| 2 | 21 | 13.46% |
| 3 | 12 | 7.69% |
| 4 | 26 | 16.67% |
| 5 | 12 | 7.69% |
| 6 | 14 | 8.97% |
| 7 | 13 | 8.33% |
| 8 | 19 | 12.18% |
| 9 | 18 | 11.54% |
| **Total** | **156** | **100%** |


---

# 15. Friedman Test

The preferred-number distribution was compared across the 11 subjects using the **Friedman test**.

### Why Friedman?

Each subject contributes preferred-number counts for all nine numbers.

The Friedman test evaluates whether particular numbers consistently receive higher ranks across subjects without requiring the same assumptions as a repeated-measures parametric ANOVA.

Our result was:

$$
\boxed{\chi_F^2(8)=12.19,\quad p=0.143}.
$$

The paper reports:

$$
\boxed{p=0.0692}.
$$

Both p-values are above 0.05.

Therefore:

$$
\boxed{\text{No statistically significant population preference for a particular arithmetic number.}}
$$


---

# 16. Preferred-Number Tie Robustness

The paper did not specify how ties in preferred number were handled.

We found only four tied neurons:

```text
YFI neuron 20 -> {2, 3, 4}
YFK neuron 26 -> {7, 9}
YFK neuron 48 -> {3, 4}
YFL neuron 38 -> {4, 5}
```

There are only:

$$
3\times2\times2\times2
=
24
$$

possible ways to resolve these four ties.

Therefore, rather than arbitrarily choosing preferred numbers, **all 24 possible assignments were tested**.

Results:

```text
Minimum p-value = 0.0755
Maximum p-value = 0.2948
Assignments with p < 0.05 = 0 / 24
```

Thus:

$$
\boxed{\text{The nonsignificant preferred-number result is robust to every possible tie assignment.}}
$$


---

# 17. Figure 1N — Temporal vs Firing-Rate Decoding Accuracy

Figure 1N asks a different question from the coding-cell analysis.

The coding-cell analysis asks:

> Is this neuron significantly number-coding?

Figure 1N instead asks:

> How accurately can this neuron decode number using the temporal code compared with the firing-rate code?

Therefore Figure 1N compares:

$$
\text{Temporal decoding accuracy}
$$

against:

$$
\text{Firing-rate decoding accuracy}
$$

for individual neurons.

The paper reports that temporal coding improves single-neuron decoding accuracy by:

$$
\boxed{3.79\text{ percentage points}}.
$$


---

# 18. Our Figure 1N Result

For the 548 neurons for which both decoding accuracies could be calculated:

```text
Mean FR accuracy       = 0.10642
Mean temporal accuracy = 0.14484
```

Therefore:

$$
0.14484-0.10642
=
0.03842.
$$

Thus:

$$
\boxed{\text{Temporal improvement}=3.842\text{ percentage points}}.
$$

The paper reports:

$$
\boxed{3.79\text{ percentage points}}.
$$

Therefore:

$$
\boxed{3.842\%\text{ ours}\approx3.79\%\text{ paper}}
$$

with a difference of only:

$$
\boxed{0.052\text{ percentage points}}.
$$

This was one of the strongest numerical reproductions obtained.


---

# 19. Direction of Improvement

Among the 548 neurons with paired decoding accuracies:

```text
Temporal > FR : 484 neurons
Temporal = FR : 64 neurons
Temporal < FR : 0 neurons
```

Thus nearly all neurons either improved or remained equal when temporal information was included.

No neuron in the paired set showed lower optimized temporal decoding accuracy than firing-rate decoding.


---

# 20. Paired Accuracy Comparison

A neuron-wise paired t-test was performed between temporal and firing-rate decoding accuracies.

Our result:

$$
\boxed{t(547)=29.65}
$$

with:

$$
\boxed{p=6.90\times10^{-116}}.
$$

The mean improvement was:

$$
\boxed{3.842\pm0.130\text{ percentage points (SEM)}}.
$$

Thus temporal decoding produced a very strong improvement over firing-rate decoding.


---

# 21. Sparse-Neuron Problem in Figure 1N

The paper reports:

$$
553/554
$$

neurons in the arithmetic decoding-accuracy comparison.

The paper applies a firing-rate criterion of:

$$
>0.007\text{ Hz}.
$$

Our reconstructed task firing rates gave:

```text
552 / 554 > 0.007 Hz
2 / 554 <= 0.007 Hz
```

Six extremely sparse neurons could not produce complete LDA accuracies:

| Subject | Neuron | Total spikes | Approx. FR (Hz) |
|---|---:|---:|---:|
| YFL | 27 | 0 | 0.00000 |
| YFM | 63 | 1 | 0.00975 |
| YFM | 75 | 4 | 0.03899 |
| YFP | 41 | 1 | 0.00703 |
| YFP | 82 | 1 | 0.00703 |
| YFR | 14 | 1 | 0.00418 |

This produced:

$$
\boxed{548}
$$

neurons with paired numerical temporal and FR decoding accuracies.


---

# 22. MATLAB Sparse-Neuron Diagnostic

Initially, we suspected that the Python LDA implementation might be incorrectly rejecting these sparse neurons.

To test this directly, **YFM neuron 63** was exported to MATLAB.

Its firing-rate feature matrix contained:

```text
114 presentations
1 total spike
113 zero-spike presentations
1 nonzero presentation
```

MATLAB `fitcdiscr` was then run directly using ordinary linear discriminant analysis.

For three folds, the single spike remained in the training set and MATLAB successfully fit the classifier.

However, in one fold the single spike occurred in the test set.

Therefore the training set contained:

```text
Training spikes: 0
Training variance: 0
```

MATLAB returned:

```text
Predictor x1 has zero within-class variance.
Either exclude this predictor or set 'discrimType'
to 'pseudoLinear' or 'diagLinear'.
```

Therefore the sparse-neuron failure is **not simply a Python implementation error**.

Standard MATLAB linear `fitcdiscr` can also fail when a CV training fold contains no variance in the predictor.

We therefore did **not** artificially change the classifier to:

```text
pseudoLinear
```

or:

```text
diagLinear
```

because the paper specifies ordinary linear LDA.


---

# 23. Fixed-CV Permutation Diagnostic

One possible explanation for our elevated temporal coding percentage was the treatment of CV partitions during the permutation test.

Our primary implementation regenerates stratified CV partitions after label shuffling.

A diagnostic alternative was therefore tested for YFF:

1. construct CV folds once;
2. keep those folds fixed;
3. preserve fold class composition during permutation;
4. run 200 permutations.

The result was:

```text
Current YFF:
11 / 37 coding = 29.73%

Fixed-CV diagnostic:
11 / 37 coding = 29.73%
```

Thus:

$$
\boxed{\text{Fixed CV did not change the YFF coding percentage.}}
$$

This diagnostic therefore did not explain the temporal coding discrepancy.

Because this alternative procedure is not explicitly specified by the paper and produced no change in YFF coding count, it was not adopted as the primary method.


---

# 24. Main Results — Paper vs Reproduction

| Analysis | Paper | Reproduction | Interpretation |
|---|---:|---:|---|
| MTL neurons | 554 | **554** | Exact match |
| Temporal coding | 24.80 ± 1.73% | **28.61 ± 1.38%** | Higher, same conclusion |
| FR coding | 3.36 ± 0.71% | **4.62 ± 0.84%** | Higher, same conclusion |
| Temporal vs FR coding | p < 0.0001 | **p = 6.20 × 10⁻⁹** | Same conclusion |
| Region effect | F(3,550)=0.43, p=.74 | **F(3,550)=0.052, p=.984** | Same conclusion |
| Preferred number | p=.0692 | **p=.143** | Same conclusion |
| Temporal decoding improvement | +3.79 pp | **+3.842 pp** | Extremely close |
| Fig. 1N neuron count | 553 | **548 paired accuracies** | Sparse-neuron discrepancy |


---

# 25. Important Mistakes / Issues Encountered

## Mistake 1 — Confusing Coding Percentage with Decoding Accuracy

Initially, the reported:

$$
24.8\%
$$

could be interpreted as decoding accuracy.

This is incorrect.

The reported 24.8% is:

> the percentage of neurons classified as temporal number-coding.

Figure 1N separately analyzes actual decoding accuracy.


---

## Mistake 2 — Raw Neuron Count vs MTL Neuron Count

The raw files contain:

$$
849\text{ neurons}.
$$

However, Figure 1 uses only MTL neurons:

$$
554\text{ neurons}.
$$

Using all 849 neurons would therefore be incorrect.


---

## Mistake 3 — Region Label

The actual dataset label is:

```text
para-hpc
```

not simply:

```text
phc
```

Correct region filtering was necessary to reproduce exactly:

$$
554\text{ MTL neurons}.
$$


---

## Mistake 4 — Cue Columns Are Not Presentation Positions

The behavioral columns:

```text
cue1
cue2
```

contain operand values.

They cannot directly be interpreted as first and second displayed cues.

The actual operand onset depends on:

```text
operationFirst
```

with special handling for:

```text
YFR
YFS
```


---

## Mistake 5 — Incorrect Trials Should Not Be Removed

It initially seemed natural to analyze only correct trials.

However, Figures 1–3 concern stimulus-defined representations.

The paper intentionally retains incorrect trials.

Therefore filtering with:

```python
correct == 1
```

would be incorrect for this analysis.


---

## Mistake 6 — `toExclude` Does Not Explain the Discrepancy

Only two arithmetic trials across the full dataset had:

```text
toExclude = 1
```

Therefore `toExclude` cannot explain the difference between:

$$
28.61\%
$$

and:

$$
24.80\%.
$$


---

## Mistake 7 — Assuming scikit-learn LDA Is Identical to MATLAB

The initial implementation used scikit-learn LDA with shrinkage.

However, the paper used MATLAB:

```matlab
fitcdiscr
```

Therefore a custom MATLAB-style LDA implementation was constructed using the paper's Gamma covariance regularization:

$$
\Sigma_{\gamma}
=
(1-\gamma)\Sigma
+
\gamma\,\mathrm{diag}(\Sigma).
$$


---

## Mistake 8 — Sparse Neurons Create Genuine LDA Problems

Several neurons contain almost no spikes.

Zero-variance training folds can therefore make ordinary linear LDA undefined.

This was not merely a Python problem.

The behavior was confirmed directly using MATLAB `fitcdiscr`.


---

## Mistake 9 — Do Not Tune Arbitrary Choices to Match the Paper

Cross-validation random seeds can noticeably affect individual-neuron decoding accuracy, selected bin size, selected Gamma, and permutation significance.

We therefore did **not** search for a random seed that artificially produces exactly:

$$
24.80\%.
$$

Instead, the reconstruction uses a documented random seed and reports the remaining discrepancy transparently.


---

# 26. Important Questions Encountered

During the reconstruction, several methodological questions arose.

### Question 1

Should incorrect trials be removed?

**Answer:** No, not for Figures 1–3.


### Question 2

Should both operands be used?

**Answer:** Yes. Both valid 1–9 operands are pooled as separate presentations.


### Question 3

Should 0 and numbers above 9 be included?

**Answer:** No. The classifier is a 9-way classifier for numbers 1–9.


### Question 4

Should bin size and Gamma be optimized separately for every neuron?

**Answer:** Yes.


### Question 5

Should bin size and Gamma be re-optimized for every permutation?

**Answer:** No. The neuron's selected parameters are retained for the permutation test.


### Question 6

Should the temporal features be z-scored?

**Answer:** No additional z-scoring was introduced because it was not specified by the paper.


### Question 7

Should firing-rate coding use a separate window?

**Answer:** No. The same 0.05–0.95 s response window was used, collapsed into a single 900-ms spike-count feature.


### Question 8

Should sparse neurons automatically be removed?

**Answer:** No. They were retained and explicitly diagnosed. When the classifier could not be fit, the failure was documented.


### Question 9

Should `pseudoLinear` be used when MATLAB linear LDA fails?

**Answer:** No. That would change the classifier from the method described in the paper.


### Question 10

Should we change random seeds until the reported percentages match?

**Answer:** No. That would be tuning the reproduction to the expected answer rather than independently reproducing the analysis.


---

# 27. Main Open Question

The most important unresolved issue is:

> Why are our significance-based coding percentages somewhat higher than the paper, while the actual temporal-vs-FR decoding improvement is almost identical?

Specifically:

$$
\text{Temporal coding: }
28.61\%\text{ ours}
\quad\text{vs}\quad
24.80\%\text{ paper}
$$

and:

$$
\text{FR coding: }
4.62\%\text{ ours}
\quad\text{vs}\quad
3.36\%\text{ paper}.
$$

However:

$$
\text{Temporal decoding improvement: }
3.842\text{ pp ours}
\quad\text{vs}\quad
3.79\text{ pp paper}.
$$

This very close Figure 1N result suggests that the main preprocessing and decoding pipeline is likely close to the authors' analysis.

The remaining coding-percentage discrepancy is more likely related to details of:

- exact cross-validation realization;
- exact permutation realization;
- MATLAB implementation details;
- significance classification;
- sparse-neuron handling;
- or another undocumented analysis detail.

The fixed-CV diagnostic on YFF did not explain the discrepancy.


---

# 28. Interpretation of the Figure 1N Match

The close agreement:

$$
\boxed{3.842\text{ pp ours}\approx3.79\text{ pp paper}}
$$

is particularly informative.

If the following major components were seriously incorrect:

- operand alignment;
- neural response windows;
- spike extraction;
- temporal binning;
- firing-rate construction;
- LDA decoding;

we would generally expect the temporal-vs-FR decoding difference to deviate substantially as well.

Therefore the Figure 1N result increases confidence that the **basic neural decoding pipeline is approximately correct**.

It does not prove that every methodological detail is identical to the authors' implementation, but it helps localize the remaining disagreement toward the significance/permutation portion of the analysis.


---

# 29. Files / Outputs Saved

Important analysis outputs were saved to the project's `tables/` and `figures/` directories.

Examples include:

```text
tables/arithmetic_temporal_vs_firing_rate_subject_comparison.csv

tables/arithmetic_temporal_region_glmm_input.csv

tables/arithmetic_temporal_region_glmm_results.txt

tables/arithmetic_temporal_preferred_number_neurons.csv

tables/arithmetic_temporal_preferred_number_distribution.csv

tables/arithmetic_temporal_preferred_number_by_subject.csv

tables/arithmetic_temporal_preferred_number_tie_robustness.csv

tables/arithmetic_temporal_preferred_number_statistics.csv

tables/figure1N_temporal_vs_firing_rate_decoding.csv

tables/figure1N_summary.csv

tables/arithmetic_figure1_reproduction_checkpoint.csv
```

The Figure 1N-style plot was saved as:

```text
figures/figure1N_temporal_vs_firing_rate_decoding.png
```

The sparse-neuron MATLAB diagnostic input was saved as:

```text
tables/diagnostic_YFM_neuron63_sparse_lda.csv
```


---

# 30. Final Arithmetic Figure 1 Summary

The arithmetic portion of Figure 1 was reproduced successfully at the level of the paper's major scientific conclusions.

The reconstruction supports three main findings:

### Finding 1 — Temporal information matters

Temporal spike patterns contain substantially more arithmetic-number information than a conventional fixed-window firing-rate representation.

This appears both in:

- the percentage of neurons classified as number-coding;
- and the actual single-neuron decoding accuracy.


### Finding 2 — Coding is not strongly region-specific

There was no significant evidence that arithmetic number-coding probability differs between:

- HPC;
- ENT;
- AMY;
- PARA-HPC.


### Finding 3 — No strong preferred-number bias

The distribution of preferred numbers did not significantly differ across numbers 1–9.

This conclusion remained nonsignificant under **all 24 possible resolutions** of the four tied neurons.


---

# 31. Overall Reproduction Assessment

The strongest quantitative reproduction was Figure 1N:

$$
\boxed{
3.842\text{ percentage-point temporal improvement}
\approx
3.79\text{ percentage points reported}
}
$$

The major remaining discrepancy is the percentage of neurons passing the permutation-based number-coding significance threshold:

$$
\boxed{
28.61\%\text{ ours}
\quad\text{vs}\quad
24.80\%\text{ paper}
}
$$

for temporal coding, and:

$$
\boxed{
4.62\%\text{ ours}
\quad\text{vs}\quad
3.36\%\text{ paper}
}
$$

for firing-rate coding.

These discrepancies were investigated rather than hidden.

In particular:

- MTL neuron filtering was verified exactly;
- cue timing was corrected using the first author's clarification;
- incorrect trials were appropriately retained;
- the MATLAB-style LDA implementation was audited;
- sparse-neuron behavior was tested directly in MATLAB;
- CV-seed sensitivity was examined;
- and a fixed-CV permutation diagnostic was performed.

The remaining difference should therefore be documented as an **unresolved implementation/methodological discrepancy**, rather than changing arbitrary analysis choices to force numerical agreement with the paper.


---

# 32. Current Status

## Arithmetic Figure 1 reproduction

$$
\boxed{\text{COMPLETE}}
$$

with documented methodological discrepancies.

The next stage should preserve these results as the baseline reproduction before moving to additional analyses.

In particular, future work should distinguish between:

1. **reproducing the authors' reported analyses**, and
2. **testing whether those analyses are sufficient to establish the claimed simplex geometry**.

The latter requires additional geometric controls and tests beyond the Figure 1 decoding analyses.