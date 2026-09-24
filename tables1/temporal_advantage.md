--------------------------------------
FINAL FIGURE 1N DATASET
-------------------------------------------------

Total MTL neurons: 554

Status:
status
success               552
threshold_excluded      2
Name: count, dtype: int64


-----
Successful: 552
Missing FR: 0
Missing temporal: 0
----
Mean FR accuracy: 0.1059245440118961
Mean temporal accuracy: 0.14582265663791502
Mean improvement: 0.039898112626018944
Mean improvement (percentage points): 3.9898112626018944
----
Temporal > FR: 496
Temporal = FR: 56
Temporal < FR: 0
---
Saved:
c:\Users\shafi\number-simplex-reproduction\tables1\figure1N_all_subjects_decoding.csv



===========================================================

========================================
FIGURE 1N PAIRED T-TEST
========================================
Number of neurons: 552

Mean firing-rate accuracy: 10.59245440118961 %
Mean temporal accuracy: 14.582265663791501 %
Mean temporal advantage: 3.9898112626018944 percentage points

Paired t statistic: 29.826296921183435
Degrees of freedom: 551
p-value: 4.397928681493731e-117

========================================
CONCLUSION
========================================
Reject H0 at alpha = 0.05.
Temporal-component decoding accuracy is significantly higher than firing-rate decoding accuracy.
This provides statistical evidence for a temporal decoding advantage.

see fig1n reproduced

==================================================


| Subject   | Neurons tested | FR-coding | TC-coding |      FR coding % |       TC coding % |
| --------- | -------------: | --------: | --------: | ---------------: | ----------------: |
| YFF       |             37 |         1 |        11 |            2.70% |            29.73% |
| YFI       |             29 |         2 |        10 |            6.90% |            34.48% |
| YFJ       |             45 |         2 |        15 |            4.44% |            33.33% |
| YFK       |             44 |         2 |        17 |            4.55% |            38.64% |
| YFL       |             56 |         1 |         9 |            1.79% |            16.07% |
| YFM       |             61 |         2 |        16 |            3.28% |            26.23% |
| YFP       |             43 |         3 |         9 |            6.98% |            20.93% |
| YFR       |             64 |         3 |        15 |            4.69% |            23.44% |
| YFS       |             59 |         4 |        20 |            6.78% |            33.90% |
| YFT       |             52 |         4 |        13 |            7.69% |            25.00% |
| YFU       |             62 |         2 |        25 |            3.23% |            40.32% |
| **Total** |        **552** |    **26** |   **160** | **4.71% pooled** | **28.99% pooled** |


======================


========================================
ANOVA: REGION EFFECT
========================================

    ANOVA MARGINAL TESTS: DFMETHOD = 'RESIDUAL'

    Term                    FStat     DF1    DF2    pValue    
    {'(Intercept)' }        35.105    1      548    5.5234e-09
    {'region_clean'}        2.1673    3      548      0.090875

========================================
FINAL FIGURE 1Q RESULT
========================================
    FStat     DF1    DF2     pValue 
    ______    ___    ___    ________

    2.1673     3     548    0.090875


Final result:
Region main effect: F(3,548) = 2.1673, p = 0.090875

Our null hypothesis is:

$$ H_0: \text{the probability of being a TC numeral-coding neuron does not differ by MTL region.} $$

We obtained

$$ p=0.0909. $$

Since

$$ 0.0909>0.05, $$

we fail to reject \(H_0\).