# Weekly Research Update
## Number Representation / Number-Simplex Project
### Advisor Meeting — October 2, 2026

This week I worked in four directions:

1. Preliminary Bayesian analysis of number coding
2. Comparison of alternative single-neuron classifiers
3. Paired-neuron decoding analysis
4. Beginning Figure 2 population/simplex geometry reproduction

---

# 1. Preliminary Bayesian Analysis of Number Coding

## Motivation

The paper identifies number-coding neurons primarily using a permutation-test
criterion based on decoding accuracy.

I started exploring whether the evidence for number coding could instead be
described probabilistically using a Bayesian framework, rather than relying
only on a binary significance decision such as

\[
p < 0.05.
\]

The broader idea was to ask whether we can quantify our uncertainty about
whether a neuron belongs to a "null/non-coding" population or a
"number-informative" population.

---

## Initial F0/F1 idea

I considered two possible accuracy-generating distributions:

\[
F_0 = \text{distribution of decoding accuracy under no number information}
\]

and

\[
F_1 = \text{distribution of decoding accuracy when number information is present}.
\]

Conceptually:

- \(F_0\): null / non-coding neurons
- \(F_1\): signal / number-coding neurons

For the 9-way number-decoding problem, chance accuracy is

\[
\frac{1}{9} \approx 0.111.
\]

The goal is eventually to estimate something like

\[
P(\text{coding}\mid \text{observed decoding accuracy})
\]

rather than simply classifying a neuron according to a permutation-test
threshold.

---

## Preliminary model explored

I experimented with a simple mixture formulation:

\[
A_i \sim (1-\pi)F_0 + \pi F_1,
\]

where

- \(A_i\) is the observed decoding accuracy of neuron \(i\),
- \(F_0\) represents the null distribution,
- \(F_1\) represents the signal distribution,
- \(\pi\) represents the proportion of neurons belonging to the signal
  population.

As an exploratory/toy implementation, I tried assumptions including:

\[
A\mid Z=0 \sim N(1/9,\,0.02^2)
\]

for the null component, together with a shifted-Beta-type distribution for
the signal component and

\[
\pi \sim \mathrm{Beta}(1,1).
\]

However, the unconstrained fit produced a very large inferred signal
proportion (approximately 98%) and a signal mean around 14.65% accuracy.

This result was clearly highly dependent on the assumed forms of \(F_0\)
and \(F_1\).

Therefore, I have **not interpreted this as a biological result** and have
not extended it to all subjects.

---

## Current status / question

This analysis is still preliminary.

Before developing it further, I want to ask:

> Is an \(F_0/F_1\) Bayesian mixture approach a sensible way to quantify
> evidence for number coding, or should I formulate the Bayesian question
> differently?

In particular, I would like guidance on:

- how the null distribution \(F_0\) should be constructed;
- whether it should come directly from the permutation distribution;
- how the signal distribution \(F_1\) should be modeled;
- whether the target should be a posterior probability of coding,
  a credible interval for decoding performance, or another quantity.

I have deliberately not scaled this analysis to every neuron/subject yet
because I first want to make sure the statistical formulation is sensible.

---

# 2. Comparison of Different Decoders

## Motivation

The original single-neuron analysis uses regularized LDA.

The paper optimizes temporal bin size and LDA shrinkage and estimates
decoding accuracy using held-out trials.

I wanted to check whether the observed temporal number information was
specific to LDA or whether similar information could be extracted using
other classifiers.

---

## Classifiers tested

I compared several classifiers using the same general number-decoding
problem:

### Linear classifiers

1. Regularized Linear Discriminant Analysis (LDA)
2. Multinomial logistic regression
3. Linear Support Vector Machine (SVM)

I also tested a nonlinear decoder as an exploratory comparison.

---

## Main observation

The three linear classifiers produced broadly similar decoding behavior.

There were neuron-level differences:

- sometimes LDA performed slightly better;
- sometimes logistic regression performed slightly better;
- sometimes linear SVM performed slightly better.

However, I did not observe a systematic alternative linear classifier that
clearly outperformed LDA.

Therefore, the main number-decoding result does not appear to depend
strongly on the specific choice of LDA versus another standard linear
classifier.

The nonlinear decoder I tested did **not** produce a clear improvement over
the linear decoders.

---

## Interpretation

This is useful because the paper's primary single-neuron analysis is based
on regularized LDA.

The classifier comparison suggests that the temporal number information we
observe is not obviously an artifact of one particular linear classifier.

At the moment I therefore see no strong reason to replace LDA for the
reproduction.

---

# 3. Paired-Neuron Decoding Analysis

## Motivation

After analyzing individual neurons, I asked whether combining two neurons
could reveal additional number information.

For each subject, I considered all possible pairs of neurons and compared
the pair's temporal decoding performance with the decoding performance of
the individual neurons.

So far I have performed this analysis for two subjects:

- YFF
- YFI

---

# 3.1 YFF

YFF contained:

\[
37 \text{ neurons}
\]

giving

\[
\binom{37}{2}=666
\]

possible neuron pairs.

### Summary

| Quantity | Result |
|---|---:|
| Number of neurons | 37 |
| Number of neuron pairs | 666 |
| Mean single-neuron temporal accuracy | 15.55% |
| Coding single neurons | 27.03% |
| Mean paired-neuron accuracy | 16.75% |
| Coding neuron pairs | 36.04% |
| Mean gain relative to best constituent neuron | -0.79 percentage points |

The average paired accuracy was somewhat higher than the average
single-neuron accuracy.

However, this comparison alone is misleading because each pair should be
compared with its **better constituent neuron**.

When I performed that comparison, the average gain was

\[
\boxed{-0.79\text{ percentage points}}.
\]

Thus, pairing two neurons did not generally outperform simply selecting the
better neuron from that pair.

---

## Non-coding + non-coding pairs

An interesting observation was that some individually non-significant
neurons became significant when combined.

For YFF:

\[
63/351
\]

non-coding + non-coding pairs became coding pairs:

\[
\boxed{17.95\%}.
\]

Within this selected subset, the average gain relative to the better
constituent neuron was approximately

\[
\boxed{+1.82\text{ percentage points}}.
\]

This suggests that some neurons that individually contain insufficient
information for significant number decoding may contain complementary
information when combined.

---

# 3.2 YFI

YFI contained:

\[
29 \text{ neurons}
\]

giving

\[
\binom{29}{2}=406
\]

possible neuron pairs.

### Summary

| Quantity | Result |
|---|---:|
| Number of neurons | 29 |
| Number of neuron pairs | 406 |
| Mean single-neuron temporal accuracy | 16.31% |
| Coding single neurons | 27.59% |
| Mean paired-neuron accuracy | 16.53% |
| Coding neuron pairs | 30.30% |
| Mean gain relative to best constituent neuron | -1.21 percentage points |

Again, pairing neurons did not generally outperform the best neuron in the
pair:

\[
\boxed{\text{mean gain}=-1.21\text{ percentage points}}.
\]

---

## Non-coding + non-coding pairs

For YFI:

\[
34/210
\]

non-coding + non-coding pairs became coding:

\[
\boxed{16.19\%}.
\]

For this selected subset, the average gain was approximately

\[
\boxed{+1.75\text{ percentage points}}.
\]

---

# 3.3 Pairing Interpretation

The results from YFF and YFI are remarkably similar.

| Result | YFF | YFI |
|---|---:|---:|
| Neurons | 37 | 29 |
| Pairs | 666 | 406 |
| Mean single accuracy | 15.55% | 16.31% |
| Mean pair accuracy | 16.75% | 16.53% |
| Mean gain vs. best constituent | -0.79 pp | -1.21 pp |
| NC + NC pairs becoming coding | 17.95% | 16.19% |
| Gain in selected NC + NC subset | +1.82 pp | +1.75 pp |

My current interpretation is:

### 1. Pairing does not generally produce a large decoding improvement.

Although pair accuracy can exceed average single-neuron accuracy, pairs
usually do not outperform the **best neuron already contained in the pair**.

### 2. Some weak/non-significant neurons contain complementary information.

Approximately 16–18% of non-coding + non-coding pairs became significant
in these two subjects.

### 3. Combining information can also reduce decoding performance.

Some combinations involving informative neurons performed worse than the
better constituent neuron.

Therefore, adding another neuron does not automatically add useful number
information.

This suggests that the important question is not simply:

> "Does adding neurons improve decoding?"

but rather:

> "Which neurons contain complementary versus redundant or interfering
> temporal information?"

So far this analysis has only been performed for two subjects, and I have
not generalized the conclusion across all subjects.

---

# 4. Figure 2 — Beginning Population / Simplex Geometry Analysis

After the single-neuron exploratory analyses, I started reproducing the
population geometry analysis in Figure 2.

For now I have worked through one arithmetic subject:

\[
\boxed{\text{YFF}}
\]

The purpose was first to understand exactly how the single-neuron temporal
representations are converted into the population representation used for
the simplex analysis.

---

# 4.1 Selecting Neurons

For Figure 2, the paper uses a more lenient number-selectivity threshold:

\[
p \leq 0.125.
\]

For YFF:

\[
37 \text{ neurons were available}
\]

and

\[
\boxed{20 \text{ neurons passed }p\leq0.125}.
\]

The paper constructs the population representation by retaining up to the
first three optimized temporal LDA components from each selected neuron.
:chatgpt-content-reference{index="0"}

---

# 4.2 Reconstructing Each Neuron's Temporal Representation

For each selected neuron I used its previously optimized:

- temporal bin size;
- LDA shrinkage parameter \(\Gamma\).

The arithmetic analysis uses the 0.05–0.95 s interval after operand onset
and scans temporal bins from 60 to 900 ms. :chatgpt-content-reference{index="1"}

Each neuron can therefore originally have a different number of temporal
bins.

Instead of directly concatenating those raw time bins, I projected each
neuron onto its own number-discriminating LDA temporal components.

Up to three components were retained from each neuron.

---

# 4.3 Population Feature Matrix

YFF had:

\[
109 \text{ valid number presentations}.
\]

Nineteen selected neurons contributed three LDA components each.

One neuron contributed only two components because its optimized temporal
representation contained only two available dimensions.

Therefore:

\[
19(3)+1(2)=59
\]

population features.

The final population matrix was therefore

\[
\boxed{X_{\mathrm{population}}\in\mathbb{R}^{109\times59}}.
\]

Interpretation:

- rows = operand presentations;
- columns = neuron-specific temporal LDA components.

Thus each operand presentation becomes one point in a 59-dimensional
population space.

---

# 4.4 Z-scoring

Each of the 59 population features was z-scored across **all 109
presentations**:

\[
Z_{ij}
=
\frac{X_{ij}-\mu_j}{\sigma_j}.
\]

Importantly, the z-scoring was **not performed separately within each
number condition**.

After z-scoring:

- each feature had approximately zero mean;
- each feature had approximately unit standard deviation.

This z-scored 59-dimensional representation is the important space for the
subsequent geometry analysis.

---

# 4.5 Number Clouds and Centroids

The 109 population vectors were separated according to number label
\(1,\ldots,9\).

The number of presentations was:

| Number | Presentations |
|---:|---:|
| 1 | 10 |
| 2 | 15 |
| 3 | 11 |
| 4 | 7 |
| 5 | 7 |
| 6 | 16 |
| 7 | 17 |
| 8 | 12 |
| 9 | 14 |
| **Total** | **109** |

For each number \(c\), I calculated its centroid:

\[
C_c
=
\frac{1}{N_c}
\sum_{i:y_i=c} Z_i.
\]

This produced:

\[
\boxed{9 \text{ centroids in } \mathbb{R}^{59}}.
\]

---

# 4.6 PCA Visualization

For visualization only, I projected the population representation onto its
first three principal components.

The explained variance was approximately:

| PC | Explained variance |
|---|---:|
| PC1 | 7.65% |
| PC2 | 5.82% |
| PC3 | 4.88% |
| **PC1–PC3 total** | **18.35%** |

Therefore, the 3D PCA figure displays only about

\[
\boxed{18.35\%}
\]

of the total population variance.

I therefore treated PCA as a visualization and did **not** use the 3D PCA
coordinates for the main transition-angle calculation.

---

# 4.7 Number-Specific Covariance Ellipsoids

Within the 3D PCA visualization, for each number I calculated:

- its mean/centroid \(\mu_c\);
- its within-number covariance matrix \(\Sigma_c\).

I eigendecomposed the covariance matrix:

\[
\Sigma_c
=
V_c\Lambda_cV_c^T.
\]

Following the paper, the visualization ellipsoid was constructed as

\[
E_c(u)
=
\mu_c
+
0.5V_c\Lambda_c^{1/2}u,
\qquad
\|u\|=1.
\]

Interpretation:

1. \(u\): point on a unit sphere;
2. \(\Lambda_c^{1/2}\): stretches the sphere according to the
   within-number standard deviations;
3. \(V_c\): rotates it into the covariance eigenvector directions;
4. \(0.5\): sets the radii to half of the corresponding SDs;
5. \(\mu_c\): moves the ellipsoid to the number centroid.

I also constructed an interactive 3D Plotly visualization with a different
color for each of the nine number clouds.

This was useful for visually inspecting the population geometry, but the
quantitative simplex analysis was kept in the unreduced 59-dimensional
space.

---

# 4.8 Transition-Angle Analysis

I then calculated the sequential centroid transition vectors directly in
the 59-dimensional z-scored population space:

\[
d_c=C_{c+1}-C_c.
\]

With 9 centroids, this gives:

\[
8 \text{ transition vectors}.
\]

The angle between successive transitions is

\[
\theta_c
=
\cos^{-1}
\left(
\frac{d_c\cdot d_{c+1}}
{\|d_c\|\|d_{c+1}\|}
\right).
\]

There are therefore:

\[
\boxed{7 \text{ transition angles}}.
\]

The paper uses this because an ordered one-dimensional line predicts
angles near \(0^\circ\), whereas a noiseless regular simplex predicts
\(120^\circ\). :chatgpt-content-reference{index="2"}

---

## YFF Results

| Transition | Angle |
|---|---:|
| 1 → 2 → 3 | 119.87° |
| 2 → 3 → 4 | 120.59° |
| 3 → 4 → 5 | 138.47° |
| 4 → 5 → 6 | 125.77° |
| 5 → 6 → 7 | 121.59° |
| 6 → 7 → 8 | 122.19° |
| 7 → 8 → 9 | 116.41° |

Mean transition angle:

\[
\boxed{123.56^\circ}.
\]

Thus, descriptively, YFF's empirical mean transition angle is close to the
\(120^\circ\) reference value associated with a regular simplex and far
from the \(0^\circ\) value of an ideal ordered line.

However, I am **not interpreting this alone as proof of a simplex**.

---

# 4.9 Pairwise Centroid Distances

As an exploratory check, I also calculated all

\[
\binom{9}{2}=36
\]

pairwise Euclidean distances among the nine centroids.

The distances were not equal.

For example:

\[
d(C_2,C_8)=3.5295
\]

was the smallest observed pairwise distance, whereas

\[
d(C_4,C_5)=6.4544
\]

was the largest.

Number 4 was relatively far from most other centroids, which was also
qualitatively visible in the PCA visualization.

Therefore, these empirical centroids clearly do **not** form a perfect
regular simplex.

However, unequal distances do not by themselves rule out an irregular
simplex.

---

# 4.10 Important Interpretation / Open Question

The transition-angle result raises an important distinction:

\[
\text{regular simplex}
\Longrightarrow
120^\circ \text{ sequential transition angles},
\]

but the converse does not necessarily hold:

\[
120^\circ \text{ transition angles}
\not\Longrightarrow
\text{regular simplex}.
\]

Only seven local sequential angles are being tested, whereas the geometry
of nine points contains many additional relationships.

Therefore, transition angles close to \(120^\circ\) should be viewed as
evidence that the data resemble the paper's simplex reference under this
particular statistic, rather than as a mathematical proof that the nine
centroids form an 8-dimensional simplex.

I have **not yet tested affine independence/effective affine dimension**.
I plan to keep that as a separate test of the simplex claim after first
faithfully reproducing the analyses used by the authors.

---

# 4.11 Next Step: Split-Half Residual Resampling

I stopped just before implementing the paper's noise-control procedure.

This is important because neural centroids are estimated from finite,
noisy trials.

The paper explicitly notes that trial noise can inflate transition angles
even if the true centers lie on a line.

Their next analysis therefore decomposes each trial response into

\[
Z_i = C_{y_i}+R_i,
\]

where

\[
R_i=Z_i-C_{y_i}
\]

is the trial residual.

They then use split-half residual resampling to generate noisy estimates of
the empirical centers and compare the empirical geometry with matched
controls.

This procedure preserves the empirical class-specific trial noise while
allowing the underlying centroid geometry to be replaced by the candidate
geometries. 

This will be my next Figure 2 step.

---

# Questions / Discussion Points for Meeting

## 1. Bayesian analysis

Is the \(F_0/F_1\) Bayesian mixture formulation a sensible direction for
quantifying number coding?

In particular, should \(F_0\) be obtained empirically from label-shuffled
decoding distributions rather than imposing a parametric null
distribution?

What would be the most meaningful Bayesian quantity to report:
posterior probability of coding, credible interval on decoding strength,
or something else?

---

## 2. Alternative classifiers

The linear decoders I tested (LDA, multinomial logistic regression and
linear SVM) behaved broadly similarly, while the nonlinear decoder did not
show a clear advantage.

Is this enough as a robustness check, or is there another decoder/model
comparison that would be informative?

---

## 3. Paired-neuron analysis

For YFF and YFI, pairs generally did not outperform the best constituent
neuron, although approximately 16–18% of non-coding + non-coding pairs
became significant.

Is this complementary-information effect worth extending across all
subjects?

Would it be more informative to quantify redundancy/synergy explicitly
rather than continue pairwise decoding alone?

---

## 4. Simplex analysis

For YFF, the raw mean transition angle was

\[
123.56^\circ,
\]

close to the regular-simplex reference of \(120^\circ\).

However, transition angles alone do not mathematically establish that the
nine centroids form an 8-simplex.

My current plan is:

1. First faithfully reproduce the paper's split-half empirical transition
   distribution.
2. Reproduce the matched noisy number-line control.
3. Reproduce the matched noisy-simplex control.
4. Continue the remaining Figure 2 analyses.
5. Then separately test the stronger geometric claim using alternative
   tests such as affine dimension/affine independence and appropriate
   non-simplex controls.

I would like to confirm whether this is the right order of analysis.