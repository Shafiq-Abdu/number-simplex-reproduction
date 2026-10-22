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