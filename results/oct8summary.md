## Paired-Neuron Analysis Across Subjects

### 1. Parallel Computing Setup

- Last week, I had completed the paired-neuron analysis only for a small number of subjects.

- To scale the analysis, I learned how to use the UH research computing environment:
  - connected to the UH server through SSH; (username@login.math.uh.edu)
  - learned how the available compute nodes work; 
  - used separate compute nodes to run different subjects in parallel;
  - used `tmux` sessions so that analyses continued running after disconnecting from SSH.
- This allowed the subject-level analyses to run simultaneously rather than processing all subjects sequentially.

---

### 2. Paired-Neuron Analysis

For each subject:

- Considered all unique pairs of neurons.
- For each pair, performed:
  - firing-rate (FR) decoding;
  - temporal-code (TC) decoding;
  - permutation testing for statistical significance.
- For the temporal decoder, I then compared the pair with its two constituent neurons.

For a pair containing neurons $i$ and $j$, I defined the better individual-neuron accuracy as

$$
A_{\mathrm{best}} = \max(A_i,A_j).
$$

The gain obtained by pairing the neurons was

$$
\Delta A = A_{\mathrm{pair}} - A_{\mathrm{best}}.
$$

Therefore:

- $\Delta A > 0$: pairing improves decoding.
- $\Delta A = 0$: no improvement.
- $\Delta A < 0$: the better individual neuron performs better.

---

### 3. Current Status

- Paired-neuron analyses were launched for all **11 subjects**.
- **10 subjects are currently completed and included in this summary.**
- **YFR is still running**, so it is not included in the results below.
- The 10 completed subjects contain a total of:

$$
\boxed{12,209\text{ neuron pairs}}
$$

| Quantity | Result |
|---|---:|
| Subjects included | 10 |
| Total neuron pairs | 12,209 |
| Mean pair TC accuracy | 15.3% |
| Mean accuracy of better individual neuron | 16.1% |
| Mean pair gain | -0.736 percentage points |
| Median pair gain | -0.629 percentage points |
| Pairs outperforming better individual neuron | 3,320 |
| Percentage of pairs showing improvement | 27.2% |

---

### 4. Does Pairing Improve Decoding?

The main result is that **pairing does not improve decoding accuracy on average**.

Across the 12,209 pairs,

$$
\text{Mean pair accuracy} = 15.3\%
$$

while

$$
\text{Mean better-single-neuron accuracy} = 16.1\%.
$$

Thus,

$$
\boxed{\text{Mean pair gain} = -0.736\text{ percentage points}}
$$

- On average, the paired decoder performs slightly worse than the better constituent neuron.
- However, this is not true for every pair.
- **27.2% of all pairs** outperform their better individual constituent.
- Therefore, there is substantial pair-to-pair variability even though the population-average gain is negative.

> **Main conclusion:** Randomly combining two neurons does not generally improve decoding relative to selecting the better individual neuron.

---

### 5. Does the Result Depend on the Coding Status of the Individual Neurons?

I separated the pairs into three categories based on the temporal-coding significance of their constituent neurons:

1. **Noncoding + Noncoding**
2. **Coding + Noncoding**
3. **Coding + Coding**

| Constituent neurons | Pair classified as coding | Mean gain over better constituent |
|---|---:|---:|
| Noncoding + Noncoding | 13.3% | -0.33 pp |
| Coding + Noncoding | 50.2% | -1.27 pp |
| Coding + Coding | 76.7% | -0.51 pp |

#### Noncoding + Noncoding

- There were **6,062** pairs in which neither constituent neuron was individually classified as temporally coding.
- Of these, **809 pairs** became statistically significant when the two neurons were decoded jointly.


$$
\frac{809}{6062}\times100
$$
$$
\boxed{13.35\%}
$$

- Thus, two neurons that individually fail the coding significance criterion can sometimes produce a significant decoder when considered jointly.
- However, their mean decoding gain relative to the better individual neuron is still slightly negative:

$$
\Delta A \approx -0.33\text{ percentage points}.
$$

#### Coding + Noncoding

- When one neuron was individually coding and the other was noncoding, approximately **50.2%** of the resulting pairs were classified as coding.
- This category showed the largest decrease relative to the better constituent:

$$
\Delta A \approx -1.27\text{ percentage points}.
$$

#### Coding + Coding

- When both constituent neurons were individually coding, approximately **76.7%** of the pairs were classified as coding.
- Nevertheless, even these pairs did not improve decoding on average:

$$
\Delta A \approx -0.51\text{ percentage points}.
$$

---

### 6. Interpretation

- **Pairing is not generally better than individual-neuron decoding.**
- The better individual neuron has slightly higher decoding accuracy on average.
- This result is consistent across all three constituent-neuron categories.
- Nevertheless, particular neuron pairs can improve decoding:
  - approximately **27% of all pairs** outperform their better constituent;
  - approximately **13% of noncoding + noncoding pairs** become statistically significant when considered jointly.
- Therefore, the paired-neuron analysis reveals **heterogeneity across neuron pairs** that is not visible from the population-average accuracy alone.

### Important Caution

- Here, **"noncoding" does not mean that the neuron contains absolutely no numerical information**.
- It only means that the neuron did not pass the statistical significance criterion used in the single-neuron analysis.
- Therefore, the observation

$$
\text{noncoding} + \text{noncoding} \rightarrow \text{coding pair}
$$

should **not yet be interpreted as proof of neural synergy**.
- A more specific statistical analysis would be required to establish whether the pair contains genuinely synergistic or complementary information.

---

### 7. Take-Home Message

> **Across the 10 completed subjects and 12,209 neuron pairs, combining two neurons does not improve temporal decoding on average relative to the better constituent neuron. However, the effect is heterogeneous: about 27% of pairs improve, and about 13% of pairs formed from two individually non-significant neurons become significant when decoded jointly. Thus, pairing is not universally advantageous for decoding accuracy, but some neuron pairs reveal jointly detectable information that is not apparent from individual-neuron significance alone.**

**Remaining:** YFR paired-neuron analysis is still running and can be added once completed.