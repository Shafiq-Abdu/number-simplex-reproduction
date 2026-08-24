## Progress Summary Before Simplex Analysis

| Representation      | Number-coding neurons |   Pooled % |      Subject mean |         Paper |
| ------------------- | --------------------: | ---------: | ----------------: | ------------: |
| Firing rate         |          **17 / 554** |  **3.07%** |  **3.34 ± 0.96%** |  3.36 ± 0.71% |
| Temporal components |         **151 / 554** | **27.26%** | **27.06 ± 2.08%** | 24.80 ± 1.73% |


#  Short Notes

- Reproducing *A number simplex in the human medial temporal lobe* independently from the Methods, without using the authors' analysis code.

- The dataset contains **11 subjects**: YFF, YFI, YFJ, YFK, YFL, YFM, YFP, YFR, YFS, YFT, and YFU.

- Successfully reconstructed the arithmetic-task MTL population: **554 neurons** — HPC 389, ENT 70, AMY 77, and para-HPC 18.

- MATLAB v7.3 neural files are read using `h5py`; behavioral tables were inspected in MATLAB and exported to CSV for Python analysis.

- Reconstructed Operand 1 and Operand 2 presentations for numbers **1–9** and pooled both operand presentations for the arithmetic analysis.

- Added special handling for **YFR and YFS**, where the operation sign is always `Cue2`.

- Reproduced the main single-neuron analyses for **YFU.hpc.43**, including firing-rate tuning, raster plots, temporal firing, LDA temporal components, and TC1–TC2 projections.

- Rebuilt the **firing-rate decoder** using firing rate during the stimulus window, stratified cross-validation, uniform class treatment, and permutation testing.

- Across all **554 neurons**, we obtained **17 FR-coding neurons (3.07%)**, with a subject-level mean of **3.34 ± 0.96%**, very close to the paper's **3.36 ± 0.71%**.

- For **temporal-code decoding**, spike timing is preserved by dividing the stimulus period into temporal bins and applying regularized LDA.
- Ran the **temporal-component number-coding analysis across all 554 MTL neurons**.

- We obtained **151/554 = 27.26% temporal-coding neurons (pooled)**.

- The subject-level temporal-coding proportion was **27.06 ± 2.08% (mean ± SEM)**.

- The paper reports **24.80 ± 1.73%**, so our reproduction is slightly higher but broadly consistent with the reported temporal-coding result.

- Therefore, both major population analyses are now complete:
  - **Firing-rate coding:** 17/554 = **3.07%**, subject mean **3.34 ± 0.96%** (paper: **3.36 ± 0.71%**).
  - **Temporal coding:** 151/554 = **27.26%**, subject mean **27.06 ± 2.08%** (paper: **24.80 ± 1.73%**).

- The reproduction therefore recovers the paper's main qualitative result: **number information is much more prevalent in temporal spike patterns than in mean firing rates**.
- Temporal parameters are optimized **separately for each neuron** using a grid search over bin size and shrinkage `Gamma`; therefore, there is no single optimal bin size for all neurons.

- Important mistakes fixed so far include the `para-hpc` region-label error (**536 → 554 neurons**), different spike keys across files, YFR/YFS cue structure, MATLAB/Python indexing differences, and degenerate LDA folds.

- Figures are saved in `figures/`, while population results and summary DataFrames are saved in `tables/`.

- **Current checkpoint:** firing-rate reproduction is complete; the population-level temporal-code accuracy and FR-vs-temporal comparison are the next results to finalize.

- After completing Figure 1 reproduction, the next major goal is to test whether the **nine number centroids genuinely form a simplex-like geometry**, including alternative/random controls rather than forcing agreement with the paper.

- *Next:** complete the remaining arithmetic population-level Figure 1 analyses — FR vs temporal coding proportions (M), neuron-wise decoding accuracy comparison (N), regional tuned-neuron proportions (Q), and preferred-number distribution (S) — then begin the simplex analysis: centroid geometry, dimensionality/affine independence, paper tests, and alternative Gaussian/random controls.

