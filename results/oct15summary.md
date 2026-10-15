#oct 15 Summary: Classifiers and Bayesian Population Decoder

## 1

### Logistic regression

- Multiclass logistic regression directly models the probability of each number class.
- For each class `k`, it calculates a logit score:
  - `z_k = beta_k^T x + b_k`
- For the 9-class problem, the logits are converted to probabilities using **softmax**.
- The predicted number is the class with the largest probability.
- The coefficients `beta_k` and intercepts `b_k` are learned by minimizing **cross-entropy loss** with regularization.
- Binary logistic regression uses a sigmoid; our 9-class problem uses softmax.

### LDA

- LDA assumes:
  - each class is approximately Gaussian;
  - all classes share the same covariance matrix.
- It uses:
  - class means / centroids;
  - pooled within-class covariance.
- It constructs supervised discriminant directions that maximize separation between number classes.
- Important reviewer concern:
  - supervised LDA can increase class separation and may make the 9 class centroids appear more affinely independent;
  - it does **not** mathematically force a simplex, but this supervised preprocessing can affect downstream geometry.

### LDA vs logistic regression: overall result

| Measure | LDA | Logistic regression |
|---|---:|---:|
| Neurons | 554 | 554 |
| Mean decoding accuracy | 14.65% | **16.80%** |
| Mean null accuracy | 11.02% | 13.60% |
| Observed - null | **3.63 pp** | 3.20 pp |
| Coding neurons | **156 / 554 (28.16%)** | 129 / 554 (23.29%) |

- Logistic regression had higher raw decoding accuracy for **424 / 554 neurons (76.53%)**.
- LDA was higher for **88 / 554 neurons (15.88%)**.
- Ties: **42 / 554 neurons**.
- Mean logistic - LDA difference: **+2.15 percentage points**.
- Main point:
  - logistic regression gives higher raw accuracy;
  - LDA still identified more significant coding neurons;
  - therefore raw accuracy and statistical coding significance are not the same thing.

---

## 2. Unsupervised PCA

- PCA is **not a classifier**.
- PCA is an unsupervised dimensionality-reduction method.
- It uses only the neural data matrix `X`, not the number labels `y`.
- Basic idea:
  - center the data;
  - find directions of maximum variance;
  - these directions are the eigenvectors / principal components.
- PCA does not directly try to separate numbers 1-9.
- I have **not yet run the full PCA control**.
- Next control:
  - apply PCA directly to the raw population activity;
  - inspect the number geometry without supervised LDA preprocessing;
  - if PCA is used before decoding, fit PCA only on the training folds.

---

## 3. Bayesian classifier: Poisson Naive Bayes

### Why Poisson?

- Neural features are spike counts.
- For a Poisson variable:
  - `mean ≈ variance`
  - Fano factor: `F = variance / mean`
- Across the final 11-subject population analysis:
  - mean Fano factor = **1.200**
- Subject mean Fano factors ranged from approximately **1.01 to 1.51**.
- This makes a Poisson count model a reasonable first approximation.
- Important:
  - Fano near 1 supports the mean-variance relationship;
  - it does **not** prove that the full spike-count distribution is exactly Poisson.

### Main assumptions

- For number class `k` and feature `j`:
  - `X_j | Y=k ~ Poisson(lambda_kj)`
- Naive Bayes assumes conditional independence of features given the number class.
- We used a uniform prior:
  - `P(Y=k) = 1/9`
- Conditional independence is the stronger assumption that still needs to be kept in mind.

### What is estimated?

- For every number and every neuron-time-bin feature:
  - `lambda_hat_kj = mean spike count of feature j in the training trials belonging to class k`
- Therefore `lambda` is just the class-specific sample mean spike count.
- We are **not** learning:
  - an LDA projection;
  - a rotation of the neural space;
  - a covariance matrix;
  - a direction chosen to maximize number separation.

### Classification rule

- Bayes rule:
  - `P(Y=k | x) ∝ P(x | Y=k) P(Y=k)`
- Poisson likelihood:
  - `P(x | Y=k) = product_j Poisson(x_j ; lambda_kj)`
- We calculate the log score:
  - `log_score_k = log(pi_k) + sum_j [x_j log(lambda_kj) - lambda_kj]`
- The `log(x_j!)` term can be omitted because it is identical across classes for the same observation.
- The predicted number is:
  - `argmax_k log_score_k`

### Important terminology

- This is a **supervised Naive Bayes classifier** because the number labels are used to estimate each `lambda_kj`.
- It is **not** a fully Bayesian Gamma-Poisson model because we currently use a plug-in maximum-likelihood estimate of `lambda`.
- However, unlike LDA, it does not learn a supervised projection that explicitly maximizes class separation.

### Reference / motivation

- Tacikowski et al. (2024), *Human hippocampal and entorhinal neurons encode the temporal structure of experience*, **Nature 635, 160-167**.
- DOI: `10.1038/s41586-024-07973-1`
- That study used a Poisson Naive Bayes population decoder on human hippocampal-entorhinal neural activity.
- Our implementation is inspired by this approach, but it is not an exact reproduction of their pipeline.

---

## 4. Poisson Naive Bayes: single-neuron test

- First tested at the single-neuron level using 100 ms spike-count bins.
- Example: `YFU.hpc.43`
  - 9 temporal features;
  - grouped decoding accuracy ≈ **11.54%**;
  - chance = **11.11%**;
  - mean Fano ≈ **0.98**.
- Across all 62 YFU MTL neurons individually:
  - mean accuracy ≈ **11.12%**;
  - median ≈ **11.23%**;
  - maximum ≈ **15.59%**.
- Conclusion:
  - single-neuron Poisson decoding was generally weak;
  - therefore I moved to a population decoder.

---

## 5. Population Poisson Naive Bayes

### Population representation

- Arithmetic response window: **50-950 ms** after stimulus onset.
- Fixed bin width: **100 ms**.
- Number of temporal bins per neuron: **9**.
- For a subject with `N` MTL neurons:
  - population feature dimension = `9 x N`.
- Example:
  - YFU has 62 MTL neurons;
  - `62 x 9 = 558` population features.
- All MTL regions are included:
  - hippocampus;
  - entorhinal cortex;
  - amygdala;
  - parahippocampal cortex.
- Total across the 11 subjects: **554 MTL neurons**.

### Cross-validation

- Used grouped cross-validation.
- Presentations belonging to the same original arithmetic trial remain in the same fold.
- This prevents operand 1 and operand 2 from the same trial leaking across training and test sets.

### Quantities calculated for every subject

- `lambda` matrix:
  - rows = 9 number classes;
  - columns = all neuron x time-bin features.
- Fano-factor matrix.
- 9 x 9 confusion matrix.
- Normalized confusion matrix.
- Trial-level predictions.
- Posterior probability for numbers 1-9.
- Accuracy.
- Balanced accuracy.
- Permutation null distribution.

---

## 6. Final population results: all 11 subjects

| Subject | MTL neurons | Features | Accuracy | Balanced accuracy | Mean Fano | Null mean | Permutation p |
|---|---:|---:|---:|---:|---:|---:|---:|
| YFF | 37 | 333 | 18.35% | 15.30% | 1.283 | 12.84% | 0.0796 |
| YFI | 29 | 261 | **22.12%** | 15.00% | 1.065 | 16.88% | 0.0846 |
| YFJ | 45 | 405 | 19.30% | 15.29% | 1.277 | 13.68% | 0.0995 |
| YFK | 44 | 396 | **21.93%** | **17.37%** | 1.281 | 14.14% | **0.0348** |
| YFL | 57 | 513 | 14.91% | 10.92% | 1.505 | 14.31% | 0.4677 |
| YFM | 61 | 549 | 14.04% | 10.13% | 1.120 | 14.79% | 0.6020 |
| YFP | 43 | 387 | 13.29% | 10.06% | 1.118 | 14.71% | 0.6965 |
| YFR | 65 | 585 | 11.28% | 10.18% | 1.143 | 12.24% | 0.6965 |
| YFS | 59 | 531 | 12.88% | 12.35% | 1.249 | 11.31% | 0.2537 |
| YFT | 52 | 468 | 13.63% | 13.14% | 1.150 | 11.34% | 0.1194 |
| YFU | 62 | 558 | 10.93% | 10.59% | 1.007 | 11.64% | 0.7164 |

### Overall population summary

| Quantity | Final result |
|---|---:|
| Subjects | **11** |
| Total MTL neurons | **554** |
| Total subject-specific temporal features | **4,986** |
| Theoretical 9-class chance | **11.11%** |
| Mean population accuracy | **15.70% ± 1.22% SEM** |
| Median population accuracy | **14.04%** |
| Mean balanced accuracy | **12.76% ± 0.79% SEM** |
| Mean permutation-null accuracy | **13.44% ± 0.53% SEM** |
| Mean Fano factor | **1.200 ± 0.041 SEM** |
| Nominal `p < 0.05` | **YFK only** |
| Survives correction across 11 subjects | **None** |

### Main interpretation

- Raw population Poisson decoding gives approximately **15-16% mean accuracy**.
- This is clearly more reasonable than the single-neuron Poisson result, which was close to chance.
- The classifier is operating directly on raw temporal spike counts.
- No LDA temporal-component projection is used before Poisson decoding.
- Therefore number information can be decoded to some extent without first constructing supervised LDA components.
- However:
  - Poisson Naive Bayes is still supervised;
  - conditional independence is an important assumption;
  - balanced accuracy is lower than raw accuracy;
  - this analysis addresses decoding robustness, not simplex geometry by itself.

---

## 7. 

  - supervised temporal LDA may increase class separation before the simplex geometry is measured.
- Poisson Naive Bayes is different:
  - it keeps the original spike-count coordinates;
  - it estimates only class-specific `lambda` values;
  - it does not rotate the data;
  - it does not estimate a covariance matrix;
  - it does not construct discriminant axes.
- Therefore this is a useful control for the question:
  - **Can number be decoded without supervised LDA components?**
- Current answer:
  - **Yes, to some extent, using the raw temporal population spike counts.**
- But this does **not** fully answer:
  - whether the raw population geometry is specifically a simplex.
- For that reviewer concern, the important next controls are:
  - unsupervised PCA;
  - random Poisson surrogate neurons;
  - geometry analysis without supervised LDA preprocessing.

---

## 8.

- Logistic regression and LDA give broadly similar decoding behavior, although logistic regression has higher raw accuracy overall.
- PCA is an unsupervised dimensionality-reduction control, not a classifier.
- Poisson Naive Bayes provides a very different probabilistic decoder based directly on spike counts.
- With fixed 100 ms bins and all **554 MTL neurons**, population Poisson decoding gives:
  - **15.70% mean raw accuracy**;
  - **12.76% mean balanced accuracy**.
- Poisson Naive Bayes does not use the supervised LDA projection, so it helps address part of the LDA circularity concern.
- The next important step is to test the geometry directly using **unsupervised PCA and random-Poisson controls**.

---

# Bayesian Classifier Derivations

## 1. Poisson Naive Bayes with Plug-in Estimate of λ

### Setup

For one trial, $`X = (X_1, \ldots, X_d)`$ is the vector of spike-count features, and $`Y \in \{1, \ldots, 9\}`$ is the number class.

We use a uniform class prior:

```math
P(Y = k) = \pi_k = \frac{1}{9}
```

For each class $`k`$ and feature $`j`$:

```math
X_j \mid Y = k \sim \operatorname{Poisson}(\lambda_{kj})
```

The Poisson probability mass function is:

```math
P(X_j = x_j \mid Y = k) = \frac{e^{-\lambda_{kj}} \, \lambda_{kj}^{x_j}}{x_j!}
```

### Estimate of λ

Using only the training data:

```math
\hat{\lambda}_{kj} = \frac{1}{N_k} \sum_{i : y_i = k} x_{ij}
```

So $`\hat{\lambda}_{kj}`$ is the sample mean spike count for class $`k`$ and feature $`j`$.

### Conditional-independence assumption

Naive Bayes assumes the features are conditionally independent given the class:

```math
P(X = x \mid Y = k) = \prod_{j=1}^{d} P(X_j = x_j \mid Y = k)
```

Therefore:

```math
P(X = x \mid Y = k) = \prod_{j=1}^{d} \frac{e^{-\hat{\lambda}_{kj}} \, \hat{\lambda}_{kj}^{x_j}}{x_j!}
```

### Posterior over the number class

By Bayes' rule:

```math
P(Y = k \mid X = x) = \frac{P(X = x \mid Y = k) \, P(Y = k)}{\sum_{\ell=1}^{9} P(X = x \mid Y = \ell) \, P(Y = \ell)}
```

with the uniform prior $`P(Y = k) = 1/9`$.

### Log-score derivation

Multiplying many small probabilities can cause numerical underflow, so we work in log space.

Define the class log score:

```math
s_k = \log P(Y = k) + \log P(X = x \mid Y = k)
```

Using conditional independence:

```math
s_k = \log \pi_k + \sum_{j=1}^{d} \log P(X_j = x_j \mid Y = k)
```

Substituting the Poisson likelihood:

```math
s_k = \log \pi_k + \sum_{j=1}^{d} \left[ -\hat{\lambda}_{kj} + x_j \log \hat{\lambda}_{kj} - \log(x_j!) \right]
```

The term $`\log(x_j!)`$ does not depend on the class $`k`$. It is identical for all nine classes, so it can be dropped when comparing classes.

The practical log score is therefore:

```math
\boxed{\, s_k = \log \pi_k + \sum_{j=1}^{d} \left[ x_j \log \hat{\lambda}_{kj} - \hat{\lambda}_{kj} \right] \,}
```

with $`\pi_k = 1/9`$ for a uniform prior.

### Prediction

The logarithm is monotonic, so the predicted class is the one with the largest log score:

```math
\boxed{\, \hat{Y} = \arg\max_k \, s_k \,}
```

### Posterior probabilities from the log scores

The normalized posterior probabilities are:

```math
P(Y = k \mid X = x) = \frac{e^{s_k}}{\sum_{\ell=1}^{9} e^{s_\ell}}
```

For numerical stability, define the log-sum-exp:

```math
L = \log \left( \sum_{\ell=1}^{9} e^{s_\ell} \right)
```

Then:

```math
\log P(Y = k \mid X = x) = s_k - L
```

and therefore:

```math
\boxed{\, P(Y = k \mid X = x) = e^{s_k - L} \,}
```

with

```math
\sum_{k=1}^{9} P(Y = k \mid X = x) = 1
```

---

## 2. Fully Bayesian Extension: Gamma Prior on λ

Instead of replacing $`\lambda`$ with a single sample-mean estimate, we model the uncertainty in $`\lambda`$.

### Gamma prior

For one class $`k`$ and one feature $`j`$:

```math
\lambda_{kj} \sim \operatorname{Gamma}(\alpha, \beta)
```

using the shape–rate parameterization:

```math
p(\lambda_{kj}) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} \, \lambda_{kj}^{\alpha - 1} \, e^{-\beta \lambda_{kj}}
```

### Poisson likelihood for the training counts

Suppose the training spike counts for this class–feature pair are $`x_1, \ldots, x_n`$. Conditioned on $`\lambda_{kj}`$:

```math
x_i \mid \lambda_{kj} \sim \operatorname{Poisson}(\lambda_{kj})
```

Hence:

```math
P(D \mid \lambda_{kj}) = \prod_{i=1}^{n} \frac{e^{-\lambda_{kj}} \, \lambda_{kj}^{x_i}}{x_i!}
```

Ignoring terms that do not depend on $`\lambda_{kj}`$:

```math
P(D \mid \lambda_{kj}) \propto e^{-n \lambda_{kj}} \, \lambda_{kj}^{\sum_i x_i}
```

### Posterior distribution of λ

By Bayes' rule:

```math
p(\lambda_{kj} \mid D) \propto P(D \mid \lambda_{kj}) \, p(\lambda_{kj})
```

Therefore:

```math
p(\lambda_{kj} \mid D) \propto \lambda_{kj}^{\alpha + \sum_i x_i - 1} \, e^{-(\beta + n) \lambda_{kj}}
```

Thus:

```math
\boxed{\, \lambda_{kj} \mid D \sim \operatorname{Gamma}\left( \alpha + \sum_{i=1}^{n} x_i, \; \beta + n \right) \,}
```

Define

```math
a = \alpha + \sum_{i=1}^{n} x_i, \qquad b = \beta + n
```

so that $`\lambda_{kj} \mid D \sim \operatorname{Gamma}(a, b)`$.

### Posterior predictive distribution

For a new spike count $`x`$:

```math
P(X_{\text{new}} = x \mid D) = \int_0^{\infty} P(X_{\text{new}} = x \mid \lambda) \, p(\lambda \mid D) \, d\lambda
```

Substituting the Poisson likelihood and the Gamma posterior:

```math
P(X_{\text{new}} = x \mid D) = \int_0^{\infty} \frac{e^{-\lambda} \lambda^{x}}{x!} \cdot \frac{b^{a}}{\Gamma(a)} \lambda^{a-1} e^{-b\lambda} \, d\lambda
```

Collecting terms:

```math
P(X_{\text{new}} = x \mid D) = \frac{b^{a}}{x! \, \Gamma(a)} \int_0^{\infty} \lambda^{x + a - 1} \, e^{-(b+1)\lambda} \, d\lambda
```

Using the identity

```math
\int_0^{\infty} \lambda^{c-1} e^{-d\lambda} \, d\lambda = \frac{\Gamma(c)}{d^{c}}
```

we obtain:

```math
P(X_{\text{new}} = x \mid D) = \frac{b^{a}}{x! \, \Gamma(a)} \cdot \frac{\Gamma(x + a)}{(b + 1)^{x + a}}
```

Therefore:

```math
\boxed{\, P(X_{\text{new}} = x \mid D) = \frac{\Gamma(x + a)}{\Gamma(a) \, x!} \left( \frac{b}{b+1} \right)^{a} \left( \frac{1}{b+1} \right)^{x} \,}
```

This is the Gamma–Poisson posterior predictive distribution, written in Negative Binomial form.

### Full Bayesian classifier

For every class $`k`$ and feature $`j`$:

```math
\lambda_{kj} \mid D \sim \operatorname{Gamma}(a_{kj}, b_{kj})
```

The predictive probability for one feature is:

```math
P(x_j \mid Y = k, D) = \int_0^{\infty} P(x_j \mid \lambda_{kj}) \, p(\lambda_{kj} \mid D) \, d\lambda_{kj}
```

Under the same conditional-independence assumption:

```math
P(X = x \mid Y = k, D) = \prod_{j=1}^{d} P(x_j \mid Y = k, D)
```

Finally:

```math
\boxed{\, P(Y = k \mid X = x, D) = \frac{P(Y = k) \prod_{j=1}^{d} P(x_j \mid Y = k, D)}{\sum_{\ell=1}^{9} P(Y = \ell) \prod_{j=1}^{d} P(x_j \mid Y = \ell, D)} \,}
```

with the same uniform class prior $`P(Y = k) = 1/9`$.

The final prediction is:

```math
\boxed{\, \hat{Y} = \arg\max_k \, P(Y = k \mid X = x, D) \,}
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>gamma_accuracy</th>
      <th>mle_accuracy</th>
      <th>delta_accuracy</th>
      <th>gamma_balanced_accuracy</th>
      <th>mle_balanced_accuracy</th>
      <th>delta_balanced</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>YFP</td>
      <td>0.151899</td>
      <td>0.132911</td>
      <td>0.018987</td>
      <td>0.105580</td>
      <td>0.100632</td>
      <td>0.004948</td>
    </tr>
    <tr>
      <th>0</th>
      <td>YFF</td>
      <td>0.192661</td>
      <td>0.183486</td>
      <td>0.009174</td>
      <td>0.148992</td>
      <td>0.153032</td>
      <td>-0.004040</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YFL</td>
      <td>0.157895</td>
      <td>0.149123</td>
      <td>0.008772</td>
      <td>0.109545</td>
      <td>0.109250</td>
      <td>0.000295</td>
    </tr>
    <tr>
      <th>10</th>
      <td>YFU</td>
      <td>0.109312</td>
      <td>0.109312</td>
      <td>0.000000</td>
      <td>0.093849</td>
      <td>0.105901</td>
      <td>-0.012052</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YFI</td>
      <td>0.221154</td>
      <td>0.221154</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.150022</td>
      <td>-0.038911</td>
    </tr>
    <tr>
      <th>9</th>
      <td>YFT</td>
      <td>0.136268</td>
      <td>0.136268</td>
      <td>0.000000</td>
      <td>0.126056</td>
      <td>0.131361</td>
      <td>-0.005305</td>
    </tr>
    <tr>
      <th>7</th>
      <td>YFR</td>
      <td>0.112782</td>
      <td>0.112782</td>
      <td>0.000000</td>
      <td>0.102472</td>
      <td>0.101830</td>
      <td>0.000642</td>
    </tr>
    <tr>
      <th>5</th>
      <td>YFM</td>
      <td>0.122807</td>
      <td>0.140351</td>
      <td>-0.017544</td>
      <td>0.087633</td>
      <td>0.101289</td>
      <td>-0.013655</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YFS</td>
      <td>0.106061</td>
      <td>0.128788</td>
      <td>-0.022727</td>
      <td>0.097012</td>
      <td>0.123494</td>
      <td>-0.026482</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YFJ</td>
      <td>0.166667</td>
      <td>0.192982</td>
      <td>-0.026316</td>
      <td>0.118040</td>
      <td>0.152883</td>
      <td>-0.034843</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YFK</td>
      <td>0.166667</td>
      <td>0.219298</td>
      <td>-0.052632</td>
      <td>0.125762</td>
      <td>0.173736</td>
      <td>-0.047974</td>
    </tr>
  </tbody>
</table>
</div>




Mean Gamma accuracy: 0.1494700785602819
Mean MLE accuracy: 0.15695052558520278
Mean Gamma balanced accuracy: 0.11145924526448975
Mean MLE balanced accuracy: 0.12758451334669166