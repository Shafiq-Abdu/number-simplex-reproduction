| Representation                 | Mean observed accuracy | Mean permutation null | Group permutation \(p\) | Significant individual subjects |
| ------------------------------ | ---------------------: | --------------------: | ----------------------: | ------------------------------: |
| **Firing rate (900-ms count)** |             **12.59%** |      subject-specific |                      —* |                        **0/11** |
| **Raw temporal (60-ms bins)**  |             **14.87%** |      subject-specific |                      —* |                        **0/11** |
| **Temporal − FR advantage**    |           **+2.28 pp** |          **+0.86 pp** |              **0.1188** |                        **0/11** |
| **LDA temporal components**    |             **13.39%** |            **12.85%** |              **0.3267** |                        **2/11** |


raw temporal decoding increased from

$$ 12.59\%\rightarrow14.87\%, $$

and 7/11 subjects improved over FR

hypothesis :
H0​:neural population activity has no relationship with number labels.


LDA beat raw temporal in only 3/11 subjects.


Individual neurons show strong evidence for temporal number coding, but simply pooling simultaneously recorded MTL neurons within subjects does not automatically produce robust population-level number decoding with the population decoders we tested.
For these permutation decoding tests, conceptually:

$$ \boxed{H_0:\text{neural population activity has no decodable relationship with number labels}} $$

versus



| Analysis                     |              Permutation result | Decision                   |
| ---------------------------- | ------------------------------: | -------------------------- |
| FR population decoding       | **0/11 subjects had \(p<.05\)** | **Fail to reject \(H_0\)** |
| Raw 60-ms temporal decoding  | **0/11 subjects had \(p<.05\)** | **Fail to reject \(H_0\)** |
| Temporal improvement over FR |     Group \(p=\mathbf{0.1188}\) | **Fail to reject \(H_0\)** |
| LDA-component decoding       |     Group \(p=\mathbf{0.3267}\) | **Fail to reject \(H_0\)** |
| LDA — YFF                    |           \(p=\mathbf{0.0297}\) | **Reject \(H_0\)** for YFF |
| LDA — YFK                    |           \(p=\mathbf{0.0396}\) | **Reject \(H_0\)** for YFK |



I did not find robust evidence for above-null population decoding


