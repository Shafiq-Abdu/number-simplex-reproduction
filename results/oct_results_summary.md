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

For a spike-count feature \(j\) and number class \(k\):

```math
X_j \mid k,\lambda_{kj} \sim \mathrm{Poisson}(\lambda_{kj})
```

Here, \(\lambda_{kj}\) is the expected spike count for number \(k\) at feature \(j\).

## Gamma prior

A Gamma prior is placed on each Poisson rate:

```math
\lambda_{kj} \sim \mathrm{Gamma}(\alpha,\beta)
```

I parameterized the prior using a prior mean \(m\) and prior strength \(\tau\):

```math
\alpha=m\tau,\qquad \beta=\tau
```

Therefore:

```math
E[\lambda_{kj}]=m
```

and:

```math
\mathrm{Var}(\lambda_{kj})=\frac{m}{\tau}
```

So \(m\) controls where the prior is centered, while \(\tau\) controls how strongly the rate estimate is pulled toward that prior.

For number \(k\), let:

```math
S_{kj}=\sum_{i:y_i=k}x_{ij}
```

be the total spike count for feature \(j\), and let \(n_k\) be the number of training presentations of number \(k\).

Because the Gamma distribution is conjugate to the Poisson likelihood:

```math
\lambda_{kj}\mid D
\sim
\mathrm{Gamma}(m\tau+S_{kj},\,\tau+n_k)
```

The posterior mean is:

```math
\lambda^*_{kj}
=
\frac{m\tau+S_{kj}}
{\tau+n_k}
```

Since the MLE is:

```math
\hat{\lambda}^{\mathrm{MLE}}_{kj}
=
\frac{S_{kj}}{n_k}
```

the posterior mean can also be written as:

```math
\lambda^*_{kj}
=
\frac{\tau}{\tau+n_k}m
+
\frac{n_k}{\tau+n_k}
\hat{\lambda}^{\mathrm{MLE}}_{kj}
```

This makes the shrinkage interpretation clear. With little data, the prior contributes more. With more observations, the estimate moves closer to the MLE.

## Posterior-predictive decoding

For decoding, I integrate over the posterior uncertainty in \(\lambda\):

```math
p(x_j\mid k,D)
=
\int
p(x_j\mid\lambda_{kj})
p(\lambda_{kj}\mid D)
\,d\lambda_{kj}
```

This gives a Gamma–Poisson posterior-predictive distribution.

Assuming conditional independence between the neural-temporal features:

```math
p(\mathbf{x}\mid k,D)
=
\prod_{j=1}^{p}
p(x_j\mid k,D)
```

The predicted number is the class with the largest posterior-predictive score.

## Prior grid

The prior parameters were selected from:

```math
m \in \mathrm{geomspace}(0.03,0.4,10)
```

and:

```math
\tau \in \mathrm{geomspace}(10,250,12)
```

giving 120 possible prior settings.

The prior parameters were selected inside the training data using grouped nested cross-validation. Trial ID was used as the group so presentations from the same behavioral trial did not appear in both training and test data.

---

# 3. Bayesian posterior-mean representation

After fitting the Gamma–Poisson model, each number has a posterior-mean population vector:

```math
\boldsymbol{\lambda}_k
=
(\lambda^*_{k1},\lambda^*_{k2},\ldots,\lambda^*_{kp})
```

The nine numbers give:

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
p=37\times9=333
```

Therefore:

```math
\Lambda_{\mathrm{YFF}}\in\mathbb{R}^{9\times333}
```

Each row represents one number.

## 4.1 Centering

First, I calculated the centroid of the nine number representations:

```math
\bar{\boldsymbol{\lambda}}
=
\frac{1}{9}
\sum_{k=1}^{9}
\boldsymbol{\lambda}_k
```

Then each representation was centered:

```math
C_k
=
\boldsymbol{\lambda}_k
-
\bar{\boldsymbol{\lambda}}
```

This removes the common population activity shared by all numbers while preserving their relative geometry.

Because:

```math
\sum_{k=1}^{9}C_k=0
```

the centered matrix can have at most rank 8:

```math
\mathrm{rank}(C)\leq8
```

---

## 4.2 SVD and affine independence

I decomposed the centered matrix using:

```math
C=U\Sigma V^\top
```

The diagonal entries of \(\Sigma\) are the singular values.

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
\mathrm{rank}(C)=8
```

Another way to check affine independence is to choose one point as a reference and form:

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

If:

```math
\mathrm{rank}(A)=8
```

then the eight difference vectors are linearly independent, so the nine points are affinely independent.

For YFF this rank is 8. Therefore the nine posterior-mean number centers form a nondegenerate 8-dimensional simplex.

This only establishes affine independence. It does not mean that the simplex is regular.

---

# 5. Representational distance matrix

For each pair of numbers \(i\) and \(j\), I calculated the Euclidean distance between their posterior-mean population vectors:

```math
D_{ij}
=
\|\boldsymbol{\lambda}_i-\boldsymbol{\lambda}_j\|_2
```

Explicitly:

```math
D_{ij}
=
\sqrt{
\sum_{f=1}^{p}
(\lambda_{if}-\lambda_{jf})^2
}
```

Doing this for all number pairs gives the \(9\times9\) representational distance matrix:

```math
D
=
[D_{ij}]_{i,j=1}^{9}
```

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

I summarized the regularity of the 36 unique pairwise distances using:

```math
\mathrm{CV}_D
=
\frac{\mathrm{SD}(D_{ij})}
{\mathrm{Mean}(D_{ij})}
```

For an exact regular simplex, all pairwise distances are equal, so:

```math
\mathrm{CV}_D=0
```

YFF has \(\mathrm{CV}_D=0.172\), showing that its edge lengths are not equal.

---

# 6. PCA visualization

PCA was applied to the centered matrix \(C\).

Because:

```math
C=U\Sigma V^\top
```

the variance explained by PC \(r\) is:

```math
\mathrm{VarFrac}_r
=
\frac{\sigma_r^2}
{\sum_{\ell=1}^{8}\sigma_\ell^2}
```

For YFF:

| Component | Variance explained |
|---|---:|
| PC1 | 38.81% |
| PC2 | 14.61% |
| PC3 | 10.18% |
| PC1–PC3 | **63.60%** |

The PCA plots are used mainly for visualization. All main distance and geometric calculations are performed in the full feature space.

---

# 7. Gram-matrix geometry

The centered Gram matrix is:

```math
G=CC^\top
```

with:

```math
G_{ij}=C_i^\top C_j
```

The diagonal contains the squared centroid radii:

```math
G_{ii}=\|C_i\|_2^2
```

The off-diagonal entries describe the relationship between different number vectors:

```math
G_{ij}
=
\|C_i\|_2
\|C_j\|_2
\cos(\theta_{ij})
```

For a regular 8-simplex:

```math
G_{ii}=r^2
```

and:

```math
G_{ij}
=
-\frac{r^2}{8},
\qquad i\neq j
```

I matched a regular simplex to the empirical radial scale and compared its Gram matrix with the empirical Gram matrix.

For YFF:

| Quantity | Value |
|---|---:|
| Ideal diagonal | 14.781 |
| Ideal off-diagonal | -1.848 |
| Diagonal RMSE | 5.58 |
| Off-diagonal RMSE | 4.11 |

The overall relative Gram error is:

```math
E_G
=
\frac{
\|G-G_{\mathrm{regular}}\|_F
}{
\|G_{\mathrm{regular}}\|_F
}
```

For YFF:

```math
E_G=0.822
```

So YFF differs substantially from an exact matched regular simplex.

---

# 8. Centroid-based angular structure

The angle between two centered number representations is obtained directly from the Gram matrix:

```math
\cos(\theta_{ij})
=
\frac{G_{ij}}
{\sqrt{G_{ii}G_{jj}}}
```

and therefore:

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
\cos(\theta)
=
-\frac{1}{8}
```

so:

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

The mean is almost exactly the regular-simplex value, but the individual angles vary substantially. So the average angle alone is not enough to call the geometry regular.

---

# 9. Simplex volume and conditioning

Using the edge matrix \(A\), the 8-dimensional simplex volume is:

```math
V_8
=
\frac{
\sqrt{\det(AA^\top)}
}{
8!
}
```

Equivalently, if \(s_1,\ldots,s_8\) are the singular values of \(A\):

```math
V_8
=
\frac{
\prod_{i=1}^{8}s_i
}{
8!
}
```

For YFF:

```text
Empirical volume = 2.449
Matched regular-simplex volume = 5.689
```

I therefore used the scale-free normalized volume:

```math
V_{\mathrm{norm}}
=
\frac{
V_{\mathrm{empirical}}
}{
V_{\mathrm{regular}}
}
```

giving:

```math
V_{\mathrm{norm}}=0.430
```

So YFF occupies about 43% of the 8D volume of the matched regular simplex.

## Conditioning

Using the largest and smallest centered singular values:

```math
\kappa_{\mathrm{sym}}
=
\frac{\sigma_1}{\sigma_8}
```

For YFF:

```math
\kappa_{\mathrm{sym}}=2.63
```

I also calculated the inverse quantity as an isotropy measure:

```math
I
=
\frac{\sigma_8}{\sigma_1}
```

giving:

```math
I=0.380
```

For an isotropic regular simplex:

```math
\kappa_{\mathrm{sym}}=1,
\qquad
I=1
```

So YFF is full rank, but the geometry is stretched unevenly across dimensions.

---

# 10. Sequential transition angles

This analysis asks how the representation moves when following the natural numerical sequence:

```text
1 → 2 → 3 → ... → 9
```

First, I formed the eight transition vectors:

```math
d_k
=
\boldsymbol{\lambda}_{k+1}
-
\boldsymbol{\lambda}_k,
\qquad
k=1,\ldots,8
```

Then, for each pair of consecutive transitions, I calculated:

```math
\cos(\theta_k)
=
\frac{
d_k^\top d_{k+1}
}{
\|d_k\|_2\|d_{k+1}\|_2
}
```

and:

```math
\theta_k
=
\cos^{-1}
\left(
\frac{
d_k^\top d_{k+1}
}{
\|d_k\|_2\|d_{k+1}\|_2
}
\right)
```

The reference values are:

```text
Straight number line = 0°
Regular simplex      = 120°
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

The trajectory is therefore strongly bent rather than close to a straight number line.

---

# 11. Number-line tests

I used two related tests here.

The first asks whether **pairwise neural distance increases with numerical distance**.

The second asks whether the complete population geometry contains a **one-dimensional axis aligned with numerical magnitude**.

These are different questions.

## 11.1 Numerical distance vs neural distance

For every unordered number pair:

```text
(1,2), (1,3), ..., (8,9)
```

there are:

```math
\binom{9}{2}=36
```

unique pairs.

For each pair \((i,j)\), I calculated two quantities.

### Step 1: numerical distance

For a linear number line:

```math
x_{ij}=|i-j|
```

For example:

```text
|1 - 2| = 1
|1 - 5| = 4
|1 - 9| = 8
```

### Step 2: neural distance

Using the Bayesian population representations:

```math
y_{ij}
=
\|\boldsymbol{\lambda}_i-\boldsymbol{\lambda}_j\|_2
```

or explicitly:

```math
y_{ij}
=
\sqrt{
\sum_{f=1}^{p}
(\lambda_{if}-\lambda_{jf})^2
}
```

This gives two vectors of length 36:

```math
\mathbf{x}
=
(x_{12},x_{13},\ldots,x_{89})
```

and:

```math
\mathbf{y}
=
(y_{12},y_{13},\ldots,y_{89})
```

### Step 3: Pearson correlation

I then calculated the Pearson correlation between numerical distance and neural distance:

```math
r
=
\frac{
\sum_q(x_q-\bar{x})(y_q-\bar{y})
}{
\sqrt{
\sum_q(x_q-\bar{x})^2
}
\sqrt{
\sum_q(y_q-\bar{y})^2
}
}
```

where \(q\) indexes the 36 number pairs.

The interpretation is simple:

```text
r > 0  : larger numerical distance tends to mean larger neural distance
r ≈ 0  : little linear relationship
r < 0  : larger numerical distance tends to mean smaller neural distance
```

A simple number-line representation would therefore predict a positive \(r\).

For YFF:

```math
r=-0.266
```

So the relationship is slightly negative rather than positive.

For example:

| Number pair | Numerical distance | Neural distance |
|---|---:|---:|
| 1 and 9 | 8 | ≈ 5.012 |
| 2 and 5 | 3 | ≈ 8.815 |

Even though 1 and 9 are much farther apart numerically, their neural representations are closer than those of 2 and 5.

---

## 11.2 Logarithmic numerical distance

I repeated exactly the same procedure, but replaced linear numerical distance with logarithmic numerical distance.

For each pair:

```math
x^{\log}_{ij}
=
|\log(i)-\log(j)|
```

The neural distance stays unchanged:

```math
y_{ij}
=
\|\boldsymbol{\lambda}_i-\boldsymbol{\lambda}_j\|_2
```

I then calculated:

```math
r_{\log}
=
\mathrm{corr}
\left(
x^{\log}_{ij},
y_{ij}
\right)
```

using the same Pearson correlation formula.

For YFF:

```math
r_{\log}=-0.221
```

So using logarithmic numerical spacing does not produce the positive distance relationship expected from a logarithmic number line either.

---

# 12. One-dimensional magnitude-axis fit

The distance-correlation analysis above asks whether **pairwise distances** behave like a number line.

Here I asked a different question:

> How much of the full neural geometry can be explained by a single direction that is explicitly aligned with numerical magnitude?

## 12.1 Linear magnitude axis

Start with the numerical values:

```math
x_k=k,
\qquad
k=1,\ldots,9
```

Since the neural matrix \(C\) has already been centered, I also center the numerical values:

```math
\tilde{x}_k
=
x_k-\bar{x}
```

For numbers 1 through 9:

```math
\bar{x}=5
```

so:

```math
\tilde{\mathbf{x}}
=
(-4,-3,-2,-1,0,1,2,3,4)^\top
```

Now fit the centered neural representation using one numerical predictor:

```math
C
\approx
\tilde{\mathbf{x}}b^\top
```

where:

```math
b\in\mathbb{R}^{p}
```

is the neural direction associated with numerical magnitude.

The least-squares estimate is:

```math
\hat b^\top
=
\frac{
\tilde{\mathbf{x}}^\top C
}{
\tilde{\mathbf{x}}^\top\tilde{\mathbf{x}}
}
```

The fitted neural representation is then:

```math
\hat C
=
\tilde{\mathbf{x}}\hat b^\top
```

So \(\hat C\) is the part of the neural geometry that can be represented along the canonical numerical axis.

## 12.2 Variance explained

I measured the fraction of the centered neural geometry captured by this fit using:

```math
R^2_{\mathrm{mag}}
=
\frac{
\|\hat C\|_F^2
}{
\|C\|_F^2
}
```

where the Frobenius norm is:

```math
\|C\|_F^2
=
\sum_{k=1}^{9}
\sum_{f=1}^{p}
C_{kf}^2
```

For YFF:

```math
R^2_{\mathrm{linear}}=0.0813
```

or:

```text
Linear magnitude variance = 8.13%
```

So only about 8% of the total centered population geometry lies along the canonical linear number axis.

---

## 12.3 Logarithmic magnitude axis

I repeated the same fit using:

```math
x_k=\log(k)
```

and then centered these values:

```math
\tilde{x}^{\log}_k
=
\log(k)
-
\frac{1}{9}
\sum_{j=1}^{9}\log(j)
```

The fitted direction is:

```math
\hat b_{\log}^\top
=
\frac{
(\tilde{\mathbf{x}}^{\log})^\top C
}{
(\tilde{\mathbf{x}}^{\log})^\top
\tilde{\mathbf{x}}^{\log}
}
```

and:

```math
\hat C_{\log}
=
\tilde{\mathbf{x}}^{\log}
\hat b_{\log}^\top
```

Then:

```math
R^2_{\log}
=
\frac{
\|\hat C_{\log}\|_F^2
}{
\|C\|_F^2
}
```

For YFF:

```text
Linear magnitude = 8.13%
Log magnitude    = 7.79%
```

Neither numerical transformation captures a large fraction of the population geometry.

---

# 13. Comparison with PC1

There is an important comparison here.

The magnitude-axis analysis asks:

> How much variance can a specific one-dimensional numerical axis explain?

PCA asks:

> What is the best possible one-dimensional direction for explaining variance, without requiring it to follow numerical magnitude?

For PC1:

```math
R^2_{\mathrm{PC1}}
=
\frac{\sigma_1^2}
{\sum_{j=1}^{8}\sigma_j^2}
```

For YFF:

```text
Linear magnitude = 8.13%
Log magnitude    = 7.79%
PC1              = 38.81%
```

Therefore:

```math
\frac{
R^2_{\mathrm{linear}}
}{
R^2_{\mathrm{PC1}}
}
=
\frac{0.0813}{0.3881}
\approx0.21
```

and:

```math
\frac{
R^2_{\log}
}{
R^2_{\mathrm{PC1}}
}
=
\frac{0.0779}{0.3881}
\approx0.20
```

So the canonical numerical axis explains only about 20% of the variance captured by the best unconstrained one-dimensional direction.

This is useful because the data clearly contain strong one-dimensional variation, but that dominant direction is not simply numerical magnitude.

---

# 14. Cross-subject Bayesian geometry

The same posterior-mean geometry analysis was then applied to all 11 subjects.

## 14.1 Affine dimension and PC1

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
\mathrm{rank}(C)=8
```

So the nine Bayesian posterior-mean number centers are affinely independent in every subject.

The main point is that full affine rank does not imply regular-simplex geometry. The shape can still be very different between subjects.

---

## 14.2 Simplex shape across subjects

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

| Measure | Regular simplex |
|---|---:|
| Distance CV | 0 |
| Gram error | 0 |
| Centroid-angle SD | 0 |
| Normalized volume | 1 |
| Condition number | 1 |
| Isotropy ratio | 1 |

YFR, YFS, and YFT are the closest to regular/isotropic geometry among the subjects.

YFP, YFJ, and YFU are more anisotropic.

---

## 14.3 Transition angles across subjects

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

All subjects fall roughly between 115° and 125°.

This is close to the 120° regular-simplex reference and very different from the 0° straight-number-line reference.

However, this result should not be interpreted by itself. For example, YFR and YFP both have transition angles close to 120°, but they have very different Gram errors, volumes and conditioning.

---

## 14.4 Numerical distance vs neural distance

For each subject, I repeated the same 36-pair calculation:

```math
x_{ij}=|i-j|
```

```math
y_{ij}
=
\|\boldsymbol{\lambda}_i-\boldsymbol{\lambda}_j\|_2
```

and calculated:

```math
r
=
\mathrm{corr}(\mathbf{x},\mathbf{y})
```

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

There is no consistent positive relationship across subjects. Most correlations are close to zero and several are negative.

So increasing numerical separation does not consistently produce increasing neural separation in the posterior-mean geometry.

---

## 14.5 Magnitude-axis variance across subjects

For each subject, I fitted:

```math
C
\approx
\tilde{\mathbf{x}}b^\top
```

using both:

```math
x_k=k
```

and:

```math
x_k=\log(k)
```

and measured:

```math
R^2
=
\frac{\|\hat C\|_F^2}
{\|C\|_F^2}
```

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

The linear magnitude direction explains about 8–15% of the centered geometry and the logarithmic direction about 8–18%.

PC1 explains about 22–53%.

So the population activity contains much stronger one-dimensional variation than what is captured specifically by numerical magnitude.

---

# 15. Current summary

1. Poisson MLE, LDA, and Gamma–Poisson Bayesian decoding give similar population decoding accuracies, averaging around 15–16%.

2. The Gamma–Poisson model gives a posterior distribution for each number × neural-temporal feature. The posterior means provide nine population vectors that can be used for condition-level geometry.

3. After centering, all 11 subjects have affine rank 8. Therefore the nine posterior-mean number centers are affinely independent.

4. Rank 8 alone does not mean that the geometry is a regular simplex. Pairwise distances, Gram structure, centroid angles, volume, conditioning and isotropy show substantial deviations from regularity.

5. YFR, YFS and YFT are the most regular/isotropic subjects by these measures. YFP, YFJ and YFU are more anisotropic.

6. The sequential numerical trajectory has mean transition angles around 115–125° across subjects, far from a straight 0° number-line trajectory and close to the 120° regular-simplex reference.

7. Pairwise neural distance does not consistently increase with numerical distance. The Pearson correlations are mostly near zero or negative.

8. Linear and logarithmic magnitude axes explain only a small part of the full population geometry. They also explain much less variance than PC1.

Overall, the posterior-mean number representations are high-dimensional and clearly not organized as a simple number line. However, the current results do not yet establish whether the observed high-dimensional geometry is specifically a structured neural simplex or a more generic consequence of high-dimensional noisy population activity.

---

# 16. Next steps

## 1. Recovery analysis

The immediate next step is a recovery analysis.

The idea is to generate synthetic data where the true geometry is known, for example a true one-dimensional number representation:

```math
\boldsymbol{\mu}_k
=
a+k\mathbf{v}
```

or more generally:

```math
\boldsymbol{\mu}_k
=
a+z_k\mathbf{v}
```

where \(z_k\) controls the known position of number \(k\) on the ground-truth line.

Then generate spike counts from these known rates:

```math
X_{ikj}
\sim
\mathrm{Poisson}(\lambda_{kj})
```

while matching the real data as closely as possible in:

- number of neurons,
- number of presentations per number,
- firing-rate distribution,
- temporal-bin structure,
- overall spike-count scale.

Then run exactly the same Bayesian estimation and geometry analysis on the synthetic data.

The main question is:

```text
Known ground truth
        ↓
Generate realistic neural data
        ↓
Apply the same analysis
        ↓
Compare recovered geometry with true geometry
```

For example, if the true representation is one-dimensional but the analysis consistently recovers rank 8 and transition angles near 120°, that would show that some of the observed high-dimensional geometry can be produced by estimation noise or the analysis procedure itself.

If the one-dimensional geometry is recovered correctly, then the empirical rank-8 result becomes more informative.

## 2. Matched transition-angle controls

The empirical transition angles should be compared with matched controls rather than interpreting 120° by itself.

Useful controls include:

```text
1D number line
random high-dimensional points
regular simplex
irregular simplex
matched Poisson population data
```

The same transition-angle calculation should be applied to each control.

## 3. Full Bayesian posterior geometry

The current geometry uses only the posterior mean:

```math
E[\Lambda\mid D]
```

Instead, draw samples from the posterior:

```math
\Lambda^{(s)}
\sim
p(\Lambda\mid D)
```

For each posterior draw, repeat the geometric measurements:

```text
affine rank
pairwise distances
Gram geometry
centroid angles
transition angles
volume
conditioning
isotropy
number-line measures
```

This gives uncertainty distributions for the geometric quantities instead of treating the posterior mean as exact.

## 4. Control for noise in the number clusters

Each number is actually represented by a cloud of trial-level neural responses, not a single point.

Conceptually:

```math
X_{ik}
=
\boldsymbol{\mu}_k
+
\boldsymbol{\epsilon}_{ik}
```

where:

```math
\boldsymbol{\mu}_k
```

is the underlying number representation and:

```math
\boldsymbol{\epsilon}_{ik}
```

is trial-to-trial variability and measurement noise.

The current analysis mainly compares estimates of the centers \(\boldsymbol{\mu}_k\).

The next analysis should determine how much of the apparent separation and high-dimensional structure remains after accounting for the within-number variability:

```text
between-number geometry
vs.
within-number noise
```

This is important for deciding whether the observed geometry reflects stable number representations or noisy estimates of those representations.
