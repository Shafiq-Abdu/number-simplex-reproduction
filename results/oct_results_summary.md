# Bayesian Population Decoding and Number-Geometry Analysis

## 1. Population decoding comparison

I compared four population decoders across all 11 subjects. The input representation was the same across the main analyses: spike counts in 100-ms bins from 50–950 ms after operand onset.

| Decoder | Mean accuracy |
|---|---:|
| Poisson MLE | **15.70%** |
| LDA | **15.53%** |
| Gamma–Poisson Bayesian | **14.95%** |
| Multinomial logistic regression | **12.80%** |
| Nominal chance | **11.11%** |

Poisson MLE, LDA, and Gamma–Poisson Bayesian decoding all give similar average performance, around 15–16%. Logistic regression is lower. Changing the decoder therefore does not produce a large improvement in population decoding accuracy.

### Subject-level results

| Subject | LDA | Logistic | Poisson MLE | Gamma–Poisson |
|---|---:|---:|---:|---:|
| YFF | 18.35% | 18.27% | 18.35% | **19.27%** |
| YFI | 18.39% | 17.39% | **22.12%** | **22.12%** |
| YFJ | 14.88% | 11.28% | **19.30%** | 16.67% |
| YFK | **22.82%** | 13.89% | 21.93% | 16.67% |
| YFL | 15.83% | 12.27% | 14.91% | 15.79% |
| YFM | 14.87% | **14.95%** | 14.04% | 12.28% |
| YFP | **17.12%** | 13.27% | 13.29% | 15.19% |
| YFR | **14.65%** | 10.52% | 11.28% | 11.28% |
| YFS | 9.12% | 6.44% | **12.88%** | 10.61% |
| YFT | **16.34%** | 12.36% | 13.63% | 13.63% |
| YFU | 8.50% | 10.12% | **10.93%** | **10.93%** |
| **Mean** | **15.53%** | **12.80%** | **15.70%** | **14.95%** |

There is substantial subject-to-subject variability, and no decoder is best for every subject.

---

# 2. Gamma–Poisson Bayesian decoder

For each number presentation, I used 9 temporal bins of width 100 ms for every neuron.

If a subject has \(N\) neurons, the total number of features is:

```math
p = 9N
```

For a spike-count feature \(j\) and number class \(k\), the model is:

```math
X_{j} \mid k,\lambda_{kj} \sim \operatorname{Poisson}(\lambda_{kj})
```

where \(\lambda_{kj}\) is the expected spike count for that number and neural-temporal feature.

## Gamma prior

A Gamma prior is placed on each Poisson rate:

```math
\lambda_{kj} \sim \operatorname{Gamma}(\alpha,\beta)
```

I parameterized it using a prior mean \(m\) and prior strength \(\tau\):

```math
\alpha = m\tau,\qquad \beta=\tau
```

Therefore,

```math
E[\lambda_{kj}] = m
```

and

```math
\operatorname{Var}(\lambda_{kj}) = \frac{m}{\tau}
```

So the two parameters have simple roles:

- \(m\) sets the center of the prior.
- \(\tau\) controls how strongly the estimate is pulled toward that prior.

For class \(k\), let \(S_{kj}\) be the total spike count for feature \(j\), and let \(n_k\) be the number of training presentations.

The posterior is:

```math
\lambda_{kj}\mid D
\sim
\operatorname{Gamma}(m\tau+S_{kj},\,\tau+n_k)
```

and the posterior mean is:

```math
\lambda^*_{kj}
=
\frac{m\tau+S_{kj}}{\tau+n_k}
```

This can also be written as:

```math
\lambda^*_{kj}
=
\frac{\tau}{\tau+n_k}m
+
\frac{n_k}{\tau+n_k}\hat{\lambda}^{\mathrm{MLE}}_{kj}
```

So the Bayesian estimate is basically a weighted combination of the prior mean and the observed MLE rate.

## Posterior-predictive decoder

For decoding, instead of fixing \(\lambda\) at one value, the full Gamma posterior can be integrated out:

```math
p(x\mid k,D)
=
\int
p(x\mid\lambda_{kj})
p(\lambda_{kj}\mid D)
\,d\lambda_{kj}
```

This gives the Gamma–Poisson posterior-predictive distribution.

Assuming conditional independence across features:

```math
p(\mathbf{x}\mid k,D)
=
\prod_{j=1}^{p}
p(x_j\mid k,D)
```

In practice I use log probabilities and select the class with the largest total log posterior-predictive probability.

## Prior grid

The prior parameters were selected from:

```math
m \in \operatorname{geomspace}(0.03,0.4,10)
```

and

```math
\tau \in \operatorname{geomspace}(10,250,12)
```

giving 120 possible prior settings.

The prior is selected inside the training data using grouped nested cross-validation. Trial ID is used as the group so presentations from the same behavioral trial do not enter both training and test sets.

---

# 3. Bayesian posterior-mean representation

After fitting the Gamma–Poisson model, each number has a posterior-mean population vector:

```math
\boldsymbol{\lambda}_k
=
(\lambda^*_{k1},\lambda^*_{k2},\ldots,\lambda^*_{kp})
```

The nine numbers therefore give:

```math
\Lambda
=
\begin{bmatrix}
\boldsymbol{\lambda}_1\\
\boldsymbol{\lambda}_2\\
\vdots\\
\boldsymbol{\lambda}_9
\end{bmatrix}
\in
\mathbb{R}^{9\times p}
```

I used these nine vectors as the condition-level Bayesian representation for the geometry analysis.

---

# 4. Bayesian geometry: YFF

YFF has 37 neurons and 9 temporal bins:

```math
p = 37\times9 = 333
```

so:

```math
\Lambda_{\mathrm{YFF}}\in\mathbb{R}^{9\times333}
```

Each row is the posterior-mean population representation of one number.

## 4.1 Centering

First, I removed the common centroid:

```math
\bar{\lambda}
=
\frac{1}{9}\sum_{k=1}^{9}\boldsymbol{\lambda}_k
```

```math
C_k
=
\boldsymbol{\lambda}_k-\bar{\lambda}
```

This is only centering. The features were not divided by their standard deviations.

Centering preserves the relative geometry and gives:

```math
\sum_{k=1}^{9}C_k=0
```

Therefore:

```math
\operatorname{rank}(C)\leq8
```

for nine condition centers.

---

## 4.2 SVD and affine independence

I used the SVD:

```math
C=U\Sigma V^\top
```

For YFF:

| Dimension | Singular value | Variance | Cumulative |
|---|---:|---:|---:|
| 1 | 7.185 | 38.81% | 38.81% |
| 2 | 4.408 | 14.61% | 53.42% |
| 3 | 3.680 | 10.18% | 63.60% |
| 4 | 3.431 | 8.85% | 72.45% |
| 5 | 3.292 | 8.15% | 80.60% |
| 6 | 3.138 | 7.40% | 88.00% |
| 7 | 2.915 | 6.39% | 94.39% |
| 8 | 2.732 | 5.61% | 100% |

The numerical rank is:

```math
\operatorname{rank}(C)=8
```

Nine points are affinely independent when the eight difference vectors

```math
\boldsymbol{\lambda}_2-\boldsymbol{\lambda}_1,\ldots,
\boldsymbol{\lambda}_9-\boldsymbol{\lambda}_1
```

are linearly independent.

YFF therefore has the maximum possible affine dimension for nine points. Its nine posterior-mean number centers form a nondegenerate 8-dimensional simplex.

This does not mean that the simplex is regular.

---

# 5. Representational distance matrix

For every pair of numbers:

```math
D_{ij}
=
\|\boldsymbol{\lambda}_i-\boldsymbol{\lambda}_j\|_2
```

This gives a \(9\times9\) representational distance matrix.

For YFF:

| Quantity | Value |
|---|---:|
| Mean pairwise distance | 5.683 |
| SD | 0.979 |
| Minimum | 4.476 |
| Maximum | 8.815 |
| Distance CV | **0.172** |
| Closest pair | 3, 8 |
| Farthest pair | 2, 5 |

I used the coefficient of variation of pairwise distances:

```math
\mathrm{CV}_D
=
\frac{\mathrm{SD}(D_{ij})}
{\mathrm{Mean}(D_{ij})}
```

A regular simplex has equal pairwise distances and therefore:

```math
\mathrm{CV}_D=0
```

YFF has unequal pairwise distances, so it is not a regular simplex.

---

# 6. PCA visualization

PCA was used mainly to visualize the centered geometry.

For YFF:

| Component | Variance explained |
|---|---:|
| PC1 | 38.81% |
| PC2 | 14.61% |
| PC3 | 10.18% |
| PC1–PC3 | **63.60%** |

The path from 1 to 9 zigzags in the PCA projections rather than following a clear straight numerical ordering.

The quantitative geometry calculations were still performed in the full 333-dimensional space, not only in the first three PCs.

---

# 7. Gram-matrix geometry

For the centered representations:

```math
G=CC^\top
```

where:

```math
G_{ij}=C_i^\top C_j
```

The diagonal gives squared distances from the centroid:

```math
G_{ii}=\|C_i\|_2^2
```

and the off-diagonal entries contain the angular relationships:

```math
G_{ij}
=
\|C_i\|_2\|C_j\|_2\cos(\theta_{ij})
```

For a regular 8-simplex with matched radial scale:

```math
G_{ii}=r^2
```

and:

```math
G_{ij}=-\frac{r^2}{8},
\qquad i\neq j
```

For YFF the matched regular-simplex values are:

| Quantity | Value |
|---|---:|
| Ideal diagonal | 14.781 |
| Ideal off-diagonal | -1.848 |
| Diagonal RMSE | 5.58 |
| Off-diagonal RMSE | 4.11 |

The relative Gram error is:

```math
E_G
=
\frac{\|G-G_{\mathrm{regular}}\|_F}
{\|G_{\mathrm{regular}}\|_F}
```

For YFF:

```math
E_G=0.822
```

Most of the total squared Gram deviation comes from the off-diagonal structure, so the irregularity is not only due to unequal radii. The relationships between different number representations are also heterogeneous.

---

# 8. Centroid-based angles

The angle between two centered number representations is:

```math
\theta_{ij}
=
\cos^{-1}
\left(
\frac{G_{ij}}
{\sqrt{G_{ii}G_{jj}}}
\right)
```

For a regular 8-simplex:

```math
\theta_{\mathrm{regular}}
=
\cos^{-1}\left(-\frac{1}{8}\right)
\approx97.18^\circ
```

For YFF:

| Quantity | Value |
|---|---:|
| Mean centroid angle | **97.20°** |
| SD | **12.95°** |
| Minimum | 70.67° |
| Maximum | 134.85° |

The mean is almost exactly the regular-simplex value, but the individual angles have a large spread. So the mean angle alone does not imply regularity.

---

# 9. Simplex volume and conditioning

Using number 1 as a reference, I formed the edge matrix:

```math
A
=
\begin{bmatrix}
\boldsymbol{\lambda}_2-\boldsymbol{\lambda}_1\\
\boldsymbol{\lambda}_3-\boldsymbol{\lambda}_1\\
\vdots\\
\boldsymbol{\lambda}_9-\boldsymbol{\lambda}_1
\end{bmatrix}
```

For YFF:

```math
\operatorname{rank}(A)=8
```

The 8D simplex volume is:

```math
V_8
=
\frac{\sqrt{\det(AA^\top)}}{8!}
```

YFF has:

```text
Empirical volume = 2.449
Matched regular-simplex volume = 5.689
Normalized volume = 0.430
```

So its volume is about 43% of a regular simplex matched to the same overall radial scale.

## Conditioning and isotropy

Using the centered singular values:

```math
\kappa
=
\frac{\sigma_1}{\sigma_8}
```

and:

```math
I
=
\frac{\sigma_8}{\sigma_1}
```

For YFF:

```text
Condition number = 2.63
Isotropy ratio = 0.380
```

For a regular isotropic simplex:

```text
Condition number = 1
Isotropy ratio = 1
```

So YFF is full rank but clearly anisotropic.

---

# 10. Sequential transition angles

For the canonical numerical sequence, define:

```math
d_k
=
\boldsymbol{\lambda}_{k+1}
-
\boldsymbol{\lambda}_k
```

The angle between consecutive transitions is:

```math
\theta_k
=
\cos^{-1}
\left(
\frac{d_k^\top d_{k+1}}
{\|d_k\|_2\|d_{k+1}\|_2}
\right)
```

The two useful references are:

```text
Straight number line: 0°
Regular simplex:      120°
```

### YFF

| Consecutive transitions | Angle |
|---|---:|
| 1→2 vs 2→3 | 137.58° |
| 2→3 vs 3→4 | 99.37° |
| 3→4 vs 4→5 | 123.36° |
| 4→5 vs 5→6 | 116.54° |
| 5→6 vs 6→7 | 110.38° |
| 6→7 vs 7→8 | 124.19° |
| 7→8 vs 8→9 | 118.86° |
| **Mean** | **118.61°** |

The canonical 1→2→...→9 path is therefore strongly bent rather than straight. Its mean angle is close to the 120° regular-simplex reference, although the individual angles are variable.

---

# 11. Number-line tests

## 11.1 Numerical distance vs neural distance

For a simple linear number line:

```math
D_{\mathrm{num}}(i,j)=|i-j|
```

and:

```math
D_{\mathrm{neural}}(i,j)
=
\|\boldsymbol{\lambda}_i-\boldsymbol{\lambda}_j\|_2
```

If the geometry followed a number line, larger numerical separation should generally correspond to larger neural separation.

For YFF:

```text
Pearson r = -0.266
```

For logarithmic numerical distance:

```math
D_{\log}(i,j)
=
|\log(i)-\log(j)|
```

the result is:

```text
Pearson r = -0.221
```

Both correlations are slightly negative rather than positive.

For example:

```text
Neural distance between 1 and 9 ≈ 5.012
Neural distance between 2 and 5 ≈ 8.815
```

even though 1 and 9 are much farther apart numerically.

So the posterior-mean Euclidean geometry does not show the distance structure expected from a simple linear or logarithmic number line.

---

# 12. One-dimensional magnitude-axis fit

The previous test uses pairwise distances. I also asked a slightly different question: even if the complete geometry is high-dimensional, is there a one-dimensional direction inside it that follows numerical magnitude?

For the linear model:

```math
x_k=k-\bar{k}
```

and:

```math
C_k\approx x_kb
```

where \(b\) is the fitted neural magnitude direction.

The explained fraction of the centered geometry is:

```math
R^2
=
\frac{\|\hat C\|_F^2}
{\|C\|_F^2}
```

For YFF:

| One-dimensional model | Variance explained |
|---|---:|
| Linear magnitude | **8.13%** |
| Log magnitude | **7.79%** |
| PC1 | **38.81%** |

PC1 is the best unconstrained one-dimensional direction, whereas the magnitude axis is forced to follow numerical order.

For YFF:

```text
Linear magnitude / PC1 ≈ 0.21
Log magnitude / PC1    ≈ 0.20
```

So numerical magnitude explains only about one fifth as much variance as the best unconstrained one-dimensional direction.

The exhaustive comparison with all possible \(9!\) one-dimensional orderings is not included here because that analysis is not yet completed in the current results.

---

# 13. Cross-subject Bayesian geometry

The same posterior-mean geometry analysis was then applied to all 11 subjects.

## 13.1 Affine dimension and PC1

| Subject | Features | Affine rank | PC1 variance |
|---|---:|---:|---:|
| YFF | 333 | **8** | 38.81% |
| YFI | 261 | **8** | 34.73% |
| YFJ | 405 | **8** | 48.38% |
| YFK | 396 | **8** | 34.24% |
| YFL | 513 | **8** | 40.78% |
| YFM | 549 | **8** | 43.21% |
| YFP | 387 | **8** | 52.74% |
| YFR | 585 | **8** | 22.47% |
| YFS | 531 | **8** | 23.61% |
| YFT | 468 | **8** | 23.83% |
| YFU | 558 | **8** | 45.77% |

All 11 subjects have:

```math
\operatorname{rank}(C)=8
```

So the nine Bayesian posterior-mean number centers are affinely independent in every subject.

The important point is that full affine rank does not imply a regular simplex. The shapes can still be very different.

---

## 13.2 Simplex shape across subjects

| Subject | Distance CV | Gram error | Centroid-angle SD | Normalized volume | Condition number | Isotropy |
|---|---:|---:|---:|---:|---:|---:|
| YFF | 0.172 | 0.822 | 12.95° | 0.431 | 2.630 | 0.380 |
| YFI | 0.208 | 0.690 | 6.81° | 0.550 | 2.268 | 0.441 |
| YFJ | 0.261 | 1.095 | 15.67° | 0.274 | 3.228 | 0.310 |
| YFK | 0.151 | 0.688 | 11.39° | 0.527 | 2.409 | 0.415 |
| YFL | 0.192 | 0.869 | 13.08° | 0.413 | 2.729 | 0.366 |
| YFM | 0.226 | 0.935 | 12.34° | 0.386 | 2.713 | 0.369 |
| YFP | 0.241 | **1.223** | **19.69°** | **0.214** | **3.484** | **0.287** |
| YFR | **0.081** | **0.351** | **6.64°** | **0.804** | **1.763** | **0.567** |
| YFS | **0.080** | **0.383** | 7.73° | **0.779** | **1.803** | **0.555** |
| YFT | **0.089** | **0.423** | 8.14° | **0.740** | **1.763** | **0.567** |
| YFU | 0.194 | 1.010 | **18.34°** | 0.343 | 2.781 | 0.360 |

For a perfectly regular simplex:

```text
Distance CV       = 0
Gram error        = 0
Centroid-angle SD = 0
Normalized volume = 1
Condition number  = 1
Isotropy ratio    = 1
```

YFR, YFS, and YFT are the closest to an isotropic/regular geometry among the subjects.

YFP, YFJ, and YFU are more anisotropic and irregular.

---

## 13.3 Transition angles across subjects

| Subject | Mean transition angle |
|---|---:|
| YFF | 118.61° |
| YFI | 123.13° |
| YFJ | 119.29° |
| YFK | 119.42° |
| YFL | 118.48° |
| YFM | 115.12° |
| YFP | 121.82° |
| YFR | 118.06° |
| YFS | 119.04° |
| YFT | 120.16° |
| YFU | 124.83° |

All subjects are roughly between 115° and 125°, close to the 120° regular-simplex transition-angle reference and far from the 0° straight-line reference.

However, transition angle alone does not tell us that the geometry is regular. For example, YFR and YFP both have transition angles near 120°, but their Gram error, volume, conditioning, and isotropy are very different.

---

## 13.4 Numerical distance vs neural distance

Using only the Pearson correlation:

| Subject | Pearson \(r\) |
|---|---:|
| YFF | -0.266 |
| YFI | +0.095 |
| YFJ | -0.154 |
| YFK | -0.060 |
| YFL | +0.022 |
| YFM | -0.175 |
| YFP | -0.028 |
| YFR | -0.142 |
| YFS | -0.102 |
| YFT | -0.178 |
| YFU | -0.013 |

There is no consistent positive relationship across subjects. Most values are close to zero and several are negative.

For YFF, \(r=-0.266\) means that larger numerical differences tend, if anything, to have slightly smaller neural distances. This is opposite to the prediction of a simple number-line representation.

---

## 13.5 Magnitude-axis variance across subjects

| Subject | Linear magnitude \(R^2\) | Log magnitude \(R^2\) | PC1 variance |
|---|---:|---:|---:|
| YFF | 8.13% | 7.79% | 38.81% |
| YFI | 14.08% | 13.44% | 34.73% |
| YFJ | 8.38% | 9.95% | 48.38% |
| YFK | 13.67% | 13.39% | 34.24% |
| YFL | 15.16% | 17.64% | 40.78% |
| YFM | 9.82% | 11.42% | 43.21% |
| YFP | 10.17% | 10.96% | 52.74% |
| YFR | 13.30% | 14.13% | 22.47% |
| YFS | 11.94% | 11.53% | 23.61% |
| YFT | 10.33% | 10.17% | 23.83% |
| YFU | 14.13% | 11.95% | 45.77% |

The linear magnitude axis explains about 8–15% of the total centered variance across subjects, and the logarithmic axis explains about 8–18%.

In comparison, PC1 alone explains about 22–53%.

So there is some variance along a numerical-magnitude direction, but numerical magnitude is not the dominant one-dimensional direction in these posterior-mean representations.

---

# 14. Current summary

At the posterior-mean condition-center level:

1. Population decoding remains modest across different classifiers. Poisson MLE, LDA, and Gamma–Poisson Bayesian decoding all average around 15–16%.

2. The Bayesian representation gives nine posterior-mean population vectors, one for each number.

3. After centering, all 11 subjects have affine rank 8. Therefore the nine number centers are affinely independent in every subject.

4. Full rank does not mean regular-simplex geometry. Distance regularity, Gram error, volume, conditioning, and isotropy vary substantially across subjects.

5. YFR, YFS, and YFT have the most regular/isotropic geometries among the subjects, while YFP, YFJ, and YFU are more anisotropic.

6. Sequential transition angles are consistently around 115–125°, much closer to the 120° simplex reference than to a straight 0° trajectory.

7. Neural distance does not consistently increase with numerical distance.

8. Linear and logarithmic magnitude axes explain much less variance than the best unconstrained one-dimensional direction given by PC1.

So the current posterior-mean geometry is high-dimensional and clearly non-number-line-like, but the present analyses alone do not tell us whether this geometry is specifically simplex-like or is what we would expect from generic high-dimensional neural representations.

---

# 15. Next steps

### 1. Recovery analysis

Build synthetic data with a known ground-truth geometry and pass it through the same analysis.

For example, start with a known one-dimensional representation and test whether the current estimation and geometry pipeline recovers dimension 1 or artificially produces a higher-dimensional structure.

The synthetic data should be matched as closely as possible to the real data in number of neurons, trial counts, firing-rate scale, temporal bins, and realistic spike-count variability.

This will tell us what geometric distortions are introduced by finite sampling, Poisson variability, Bayesian shrinkage, and the analysis procedure itself.

### 2. Transition-angle controls across subjects

Compare the empirical transition angles with matched control representations rather than interpreting proximity to 120° by itself.

The controls should preserve important properties of the neural data while removing the specific numerical geometry.

### 3. Full Bayesian posterior geometry

The current geometry uses only:

```math
E[\Lambda\mid D]
```

Instead, sample:

```math
\Lambda^{(s)}\sim p(\Lambda\mid D)
```

and repeat the geometric measurements for each posterior sample.

This will show how certain we actually are about the observed distances, angles, rank structure, Gram geometry, volume, and conditioning.

### 4. Control the effect of cluster noise

The current condition centers may look high-dimensional partly because each number representation is estimated from noisy neural responses.

We need a control that separates:

```text
true separation between number representations
```

from:

```text
trial-to-trial / estimation noise around each representation
```

This should help determine whether the apparent high-dimensional simplex structure remains after accounting for within-number variability.
