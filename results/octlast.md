## Population-Level Decoder Comparison

We compared four decoding approaches across the 11 subjects.

| Decoder | Mean Accuracy | Main Observation |
|---|---:|---|
| **Poisson MLE** | **15.70%** | Highest mean accuracy, but only slightly above LDA |
| **LDA (100-ms bins)** | **15.53%** | Very similar to Poisson MLE |
| **Gamma–Poisson Bayesian** | **14.95%** | Similar overall performance; prior helps some subjects but hurts others |
| **Multinomial Logistic Regression (100-ms bins)** | **12.80%** | Lowest mean accuracy among the four decoders |
| **Nominal Chance (9 classes)** | **11.11%** | Reference level only |

### Main observations

- **Poisson MLE and LDA give almost identical average decoding performance.**
  - MLE: **15.70%**
  - LDA: **15.53%**

- The **Gamma–Poisson Bayesian decoder** also gives a similar average accuracy of **14.95%**.
  - The Gamma prior improves decoding for some subjects.
  - It decreases decoding accuracy for other subjects.
  - Therefore, introducing the prior does not systematically improve decoding.

- **Multinomial logistic regression** gives the lowest average accuracy at **12.80%**.

- Overall, changing the decoder does **not** produce a large increase in decoding accuracy.

### Preliminary interpretation

The three substantially different approaches — LDA, Poisson MLE, and Gamma–Poisson Bayesian decoding — all produce average accuracies around **15–16%**.

This suggests that the relatively low decoding accuracy may not be primarily caused by the choice of decoder.

Instead, an important next question is whether the limitation comes from:

1. the neural features used to represent each number,
2. the amount or reliability of number-related information in the recorded population,
3. variability across subjects and neurons, or
4. the way temporal neural responses are represented.

**Important:** These mean accuracies are descriptive results. Nominal chance is 11.11%, but statistical evidence for above-chance decoding should be evaluated using the appropriate permutation/null analysis.



## Subject-Level Comparison of Population Decoders

| Subject | LDA Accuracy | Logistic Accuracy | Poisson MLE Accuracy | Gamma–Poisson Accuracy |
|:---|---:|---:|---:|---:|
| **YFF** | 18.35% | 18.27% | 18.35% | 19.27% |
| **YFI** | 18.39% | 17.39% | 22.12% | 22.12% |
| **YFJ** | 14.88% | 11.28% | 19.30% | 16.67% |
| **YFK** | 22.82% | 13.89% | 21.93% | 16.67% |
| **YFL** | 15.83% | 12.27% | 14.91% | 15.79% |
| **YFM** | 14.87% | 14.95% | 14.04% | 12.28% |
| **YFP** | 17.12% | 13.27% | 13.29% | 15.19% |
| **YFR** | 14.65% | 10.52% | 11.28% | 11.28% |
| **YFS** | 9.12% | 6.44% | 12.88% | 10.61% |
| **YFT** | 16.34% | 12.36% | 13.63% | 13.63% |
| **YFU** | 8.50% | 10.12% | 10.93% | 10.93% |
| **Mean** | **15.53%** | **12.80%** | **15.70%** | **14.95%** |
| **Nominal Chance** | **11.11%** | **11.11%** | **11.11%** | **11.11%** |

### Preliminary observations

- **Poisson MLE:** highest mean accuracy at approximately **15.70%**.
- **LDA:** very similar mean accuracy at **15.53%**.
- **Gamma–Poisson:** mean accuracy of approximately **14.95%**.
- **Logistic regression:** lowest mean accuracy at approximately **12.80%**.
- Performance varies substantially across subjects.
- No decoder consistently dominates across every subject.
- LDA, MLE, and Gamma–Poisson remain in a broadly similar accuracy range.
- Nominal 9-class chance accuracy is **11.11%**.

> **Note:** These are descriptive decoding accuracies. Comparison with 11.11% nominal chance does not establish statistical significance; the appropriate permutation/null analysis is still required.



---
# Gamma–Poisson Bayesian Population Decoder

## Goal

Reconstruct the Gamma–Poisson Bayesian decoder from first principles and understand every statistical assumption before applying it to the complete number-decoding dataset.

The final goals are:

1. reproduce the subject-level Bayesian decoding results,
2. understand how the Gamma prior changes the Poisson rate estimates,
3. distinguish MLE, posterior-mean, and posterior-predictive decoding,
4. identify a Bayesian-derived population vector that could later be used for simplex geometry.

---

## Population representation

For each number presentation, neural activity is represented using fixed 100-ms temporal bins from 50–950 ms after operand onset.

Each neuron therefore contributes 9 spike-count features.

For \(N\) neurons,

\[
p = 9N
\]

features are used.

A single presentation is represented as

\[
\mathbf{x} = [x_1,\ldots,x_p],
\]

where each \(x_j\) is a spike count.

---

## Poisson model

For number class \(c\) and feature \(j\),

\[
X_j \mid c,\lambda_{cj}
\sim
\operatorname{Poisson}(\lambda_{cj}\Delta t),
\]

where

\[
\Delta t = 0.1 \text{ s}.
\]

Here, \(\lambda_{cj}\) is the firing rate in spikes/s associated with number \(c\) for neural feature \(j\).

---

## Gamma prior

We place a Gamma prior on each Poisson firing rate:

\[
\lambda_{cj}
\sim
\operatorname{Gamma}(\alpha,\beta),
\]

using the shape–rate parameterization.

The Gamma distribution is conjugate to the Poisson likelihood.

---

## Posterior

If class \(c\) has \(n_c\) training presentations and the total spike count for feature \(j\) is

\[
S_{cj} = \sum_i x_{ij},
\]

then

\[
\lambda_{cj}\mid D
\sim
\operatorname{Gamma}
\left(
\alpha + S_{cj},
\beta + n_c\Delta t
\right).
\]

The posterior mean is

\[
E[\lambda_{cj}\mid D]
=
\frac{
\alpha + S_{cj}
}{
\beta + n_c\Delta t
}.
\]

For comparison, the Poisson maximum-likelihood estimate is

\[
\hat{\lambda}^{MLE}_{cj}
=
\frac{
S_{cj}
}{
n_c\Delta t
}.
\]


## Posterior Prediction

After observing the training data, the firing rate for class \(c\) and feature \(j\) has posterior

\[
\lambda_{cj}\mid D
\sim
\operatorname{Gamma}
\left(
\alpha'_{cj},
\beta'_{cj}
\right).
\]

There are two ways to use this posterior for decoding.

### 1. Posterior-mean plug-in decoder

Use the posterior mean

\[
\hat{\lambda}_{cj}
=
E[\lambda_{cj}\mid D]
=
\frac{\alpha'_{cj}}{\beta'_{cj}}
\]

as the estimated firing rate and evaluate a Poisson likelihood:

\[
x_j
\sim
\operatorname{Poisson}
\left(
\hat{\lambda}_{cj}\Delta t
\right).
\]

This uses Bayesian estimation of the firing rate but reduces the posterior distribution to a single point estimate.

### 2. Full Gamma–Poisson posterior-predictive decoder

Instead of selecting one value of \(\lambda\), integrate over its complete posterior uncertainty:

\[
p(x_j\mid c,D)
=
\int
p(x_j\mid \lambda_{cj})
p(\lambda_{cj}\mid D)
\,d\lambda_{cj}.
\]

For a Gamma posterior and Poisson observation model, this integral is available analytically:

\[
p(x_j\mid c,D)
=
\frac{\Gamma(\alpha'_{cj}+x_j)}
{\Gamma(\alpha'_{cj})x_j!}
\left(
\frac{\beta'_{cj}}
{\beta'_{cj}+\Delta t}
\right)^{\alpha'_{cj}}
\left(
\frac{\Delta t}
{\beta'_{cj}+\Delta t}
\right)^{x_j}.
\]

This is the Gamma–Poisson, or negative-binomial, posterior-predictive distribution.


-----
# Gamma-Poisson Bayesian Decoding Pipeline

## 1. Temporal Population Features

For each presentation $i$, construct a temporal population vector

```math
\mathbf{x}_i=(x_{i1},x_{i2},\ldots,x_{ip})
```

where $x_{ij}$ is the spike count of feature $j$ during a fixed 100-ms time bin.

If there are $N$ neurons and 9 temporal bins per neuron,

```math
p=9N
```

For a subject with $n$ presentations,

```math
X\in\mathbb{R}^{n\times p},\qquad y_i\in\{1,2,\ldots,9\}
```

Each class $k\in\{1,\ldots,9\}$ corresponds to one presented number.

---

## 2. Poisson Likelihood

For number $k$ and feature $j$, assume

```math
X_{ij}\mid Y_i=k,\lambda_{kj}\sim\operatorname{Poisson}(\lambda_{kj})
```

where $\lambda_{kj}$ is the expected spike count for number $k$ at feature $j$.

The Poisson probability mass function is

```math
P(X_{ij}=x\mid\lambda_{kj})=\frac{\lambda_{kj}^{x}e^{-\lambda_{kj}}}{x!}
```

Here, $\lambda_{kj}$ is measured in expected spikes per 100-ms bin.

---

## 3. Gamma Prior

Place a Gamma prior on each Poisson rate:

```math
\lambda_{kj}\sim\operatorname{Gamma}(\alpha,\beta)
```

using the shape-rate parameterization.

Parameterize the prior using a prior mean $m$ and prior strength $\tau$:

```math
\alpha=m\tau,\qquad\beta=\tau
```

Therefore,

```math
E[\lambda_{kj}]=\frac{\alpha}{\beta}=m
```

and

```math
\operatorname{Var}(\lambda_{kj})=\frac{\alpha}{\beta^2}=\frac{m}{\tau}
```

Thus:

- $m$ controls the center of the prior.
- $\tau$ controls the strength of the prior.

---

## 4. Sufficient Statistics

For each number $k$ and feature $j$, calculate the total spike count

```math
S_{kj}=\sum_{i:Y_i=k}X_{ij}
```

and the number of training presentations belonging to class $k$:

```math
n_k=\#\{i:Y_i=k\}
```

The pair $(S_{kj},n_k)$ contains the information required to update the Gamma-Poisson model.

---

## 5. Posterior Distribution

Because the Gamma prior is conjugate to the Poisson likelihood,

```math
\lambda_{kj}\mid D\sim\operatorname{Gamma}(\alpha^*_{kj},\beta^*_k)
```

where

```math
\alpha^*_{kj}=\alpha+S_{kj}
```

and

```math
\beta^*_k=\beta+n_k
```

Substituting $\alpha=m\tau$ and $\beta=\tau$ gives

```math
\lambda_{kj}\mid D\sim\operatorname{Gamma}(m\tau+S_{kj},\tau+n_k)
```

---

## 6. Posterior Mean

The posterior mean of the Poisson rate is

```math
\lambda^*_{kj}=E[\lambda_{kj}\mid D]=\frac{m\tau+S_{kj}}{\tau+n_k}
```

This can also be written as

```math
\lambda^*_{kj}=\frac{\tau}{\tau+n_k}m+\frac{n_k}{\tau+n_k}\frac{S_{kj}}{n_k}
```

Since

```math
\hat{\lambda}^{\mathrm{MLE}}_{kj}=\frac{S_{kj}}{n_k}
```

the posterior mean can be interpreted as

```math
\lambda^*_{kj}=w_{\mathrm{prior}}m+w_{\mathrm{data}}\hat{\lambda}^{\mathrm{MLE}}_{kj}
```

where

```math
w_{\mathrm{prior}}=\frac{\tau}{\tau+n_k},\qquad w_{\mathrm{data}}=\frac{n_k}{\tau+n_k}
```

and

```math
w_{\mathrm{prior}}+w_{\mathrm{data}}=1
```

Thus the Bayesian estimate shrinks the class-specific MLE toward the prior mean $m$.

---

## 7. Posterior Predictive Distribution

For a new spike count $x$, the unknown rate $\lambda_{kj}$ is integrated out:

```math
P(x\mid Y=k,D)=\int P(x\mid\lambda_{kj})P(\lambda_{kj}\mid D)\,d\lambda_{kj}
```

Let

```math
a_{kj}=m\tau+S_{kj},\qquad b_k=\tau+n_k
```

Then the Gamma-Poisson posterior predictive distribution is

```math
P(x\mid Y=k,D)=\frac{\Gamma(a_{kj}+x)}{\Gamma(a_{kj})x!}\left(\frac{b_k}{b_k+1}\right)^{a_{kj}}\left(\frac{1}{b_k+1}\right)^x
```

---

## 8. Naive Bayes Population Likelihood

Assuming conditional independence across the $p$ temporal population features,

```math
P(\mathbf{x}\mid Y=k,D)=\prod_{j=1}^{p}P(x_j\mid Y=k,D)
```

For numerical stability, use log probabilities:

```math
\log P(\mathbf{x}\mid Y=k,D)=\sum_{j=1}^{p}\log P(x_j\mid Y=k,D)
```

With a uniform class prior,

```math
P(Y=k)=\frac{1}{9}
```

the predicted number is

```math
\hat{k}=\operatorname*{argmax}_{k\in\{1,\ldots,9\}}\log P(\mathbf{x}\mid Y=k,D)
```

Terms that are constant across all candidate classes, such as $-\log(x_j!)$ and $\log(1/9)$, can be omitted from the `argmax`.

---

## 9. Hyperparameter Grid

The prior parameters $m$ and $\tau$ are selected using grid search.

The prior-mean grid is

```math
m\in\operatorname{geomspace}(0.03,0.4,10)
```

and the prior-strength grid is

```math
\tau\in\operatorname{geomspace}(10,250,12)
```

Therefore, the total number of candidate priors is

```math
10\times12=120
```

For every candidate pair $(m,\tau)$,

```math
\alpha=m\tau,\qquad\beta=\tau
```

---

## 10. Nested Grouped Cross-Validation

Use 5-fold outer grouped cross-validation:

```math
D=D_{\mathrm{outer\ train}}\cup D_{\mathrm{outer\ test}}
```

The behavioral trial ID is used as the grouping variable so that presentations originating from the same trial cannot appear in both training and test sets.

For each outer fold, use 4-fold grouped cross-validation inside the outer training data:

```math
D_{\mathrm{outer\ train}}=D_{\mathrm{inner\ train}}\cup D_{\mathrm{inner\ validation}}
```

For every candidate $(m,\tau)$:

1. Fit the Gamma-Poisson model using the inner-training data.
2. Predict the inner-validation data.
3. Combine predictions across the four inner folds.
4. Calculate balanced accuracy and ordinary accuracy.

The selected hyperparameters are

```math
(m^*,\tau^*)=\operatorname*{argmax}_{m,\tau}\operatorname{BalancedAccuracy}_{\mathrm{inner}}(m,\tau)
```

with ordinary accuracy used as a tie-breaker.

The outer-test data are never used to select $m$ or $\tau$.

---

## 11. Outer-Fold Model

After selecting $(m^*,\tau^*)$, define

```math
\alpha^*=m^*\tau^*,\qquad\beta^*=\tau^*
```

Refit the Gamma-Poisson model using the complete outer-training set.

For every class $k$ and feature $j$,

```math
\lambda^*_{kj}=\frac{m^*\tau^*+S_{kj}}{\tau^*+n_k}
```

The posterior predictive distribution is then used to classify the untouched outer-test presentations.

Repeat this procedure for all five outer folds.

---

## 12. Final Decoding Accuracy

Concatenate the held-out predictions from all five outer folds:

```math
\hat{\mathbf{y}}=(\hat{\mathbf{y}}_1,\hat{\mathbf{y}}_2,\ldots,\hat{\mathbf{y}}_5)
```

The final decoding accuracy is

```math
\operatorname{Accuracy}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(\hat{y}_i=y_i)
```

Balanced accuracy is

```math
\operatorname{BalancedAccuracy}=\frac{1}{9}\sum_{k=1}^{9}\operatorname{Recall}_k
```

---

## 13. Bayesian Representation Matrix

After selecting a prior for the full subject dataset, calculate the posterior mean for every number-feature combination:

```math
\lambda^*_{kj}=\frac{m\tau+S_{kj}}{\tau+n_k}
```

Collect these posterior means into the matrix

```math
\Lambda^*=\begin{bmatrix}\lambda^*_{1,1}&\lambda^*_{1,2}&\cdots&\lambda^*_{1,p}\\\lambda^*_{2,1}&\lambda^*_{2,2}&\cdots&\lambda^*_{2,p}\\\vdots&\vdots&\ddots&\vdots\\\lambda^*_{9,1}&\lambda^*_{9,2}&\cdots&\lambda^*_{9,p}\end{bmatrix}\in\mathbb{R}^{9\times p}
```

Each row

```math
\boldsymbol{\lambda}^*_k=(\lambda^*_{k1},\lambda^*_{k2},\ldots,\lambda^*_{kp})
```

is the Bayesian posterior-mean population representation of number $k$.

Therefore, the nine numbers correspond to nine points in the $p$-dimensional neural representation space:

```math
\boldsymbol{\lambda}^*_1,\boldsymbol{\lambda}^*_2,\ldots,\boldsymbol{\lambda}^*_9\in\mathbb{R}^{p}
```

These nine posterior-mean vectors form the input for the subsequent Bayesian geometry and simplex analysis.

---

## Pipeline Summary

```text
Spike trains
    ↓
50–950 ms analysis window
    ↓
100-ms temporal bins
    ↓
Population spike-count matrix X
    ↓
Poisson likelihood
    ↓
Gamma(mτ, τ) prior
    ↓
Sufficient statistics: S_kj and n_k
    ↓
Gamma posterior
    ↓
Posterior predictive distribution
    ↓
Naive Bayes classification
    ↓
Nested grouped cross-validation
    ↓
Select m and τ
    ↓
Held-out decoding accuracy
    ↓
Posterior means λ*_kj
    ↓
9 × p Bayesian representation matrix Λ*
    ↓
Bayesian geometry / simplex analysis
```
