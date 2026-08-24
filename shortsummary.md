## Progress Summary Before Simplex Analysis

| Representation      | Number-coding neurons |   Pooled % |      Subject mean |         Paper |
| ------------------- | --------------------: | ---------: | ----------------: | ------------: |
| Firing rate         |          **17 / 554** |  **3.07%** |  **3.34 ± 0.96%** |  3.36 ± 0.71% |
| Temporal components |         **151 / 554** | **27.26%** | **27.06 ± 2.08%** | 24.80 ± 1.73% |



1. **Loaded and organized data from all 11 patients** using `h5py`, including both `spikes.mat` and `spikesArithmetic.mat` formats.

2. **Recovered the exact 554 MTL neurons** reported in the paper: HPC = 389, ENT = 70, AMY = 77, para-HPC = 18.

3. **Aligned arithmetic trials with spike data** and pooled Operand 1 and Operand 2 presentations for numbers 1–9, including the special cue structure for YFR/YFS.

4. **Reproduced single-neuron firing-rate tuning** for the example neuron YFU.hpc.43, including mean firing rate ± SEM across number conditions.

5. **Reproduced spike-timing analyses** using raster/spike-train plots and Gaussian-smoothed temporal firing rates, showing that numbers with similar average firing rates can still have different temporal response patterns.

6. **Reproduced the temporal-component/LDA analysis** for YFU.hpc.43: temporal bins, LDA weights, TC1/TC2 projections, number centroids, variability ellipses, and reduced overlap between number representations.

7. **Ran temporal-component number-coding analysis across all 554 MTL neurons.** We obtained **151/554 = 27.26% temporal-coding neurons (pooled)** and a subject-level mean of **27.06 ± 2.08%**, compared with the paper's **24.80 ± 1.73%**.

8. **Rebuilt firing-rate-only decoding independently** using one firing-rate feature, stratified cross-validation, and 200-shuffle permutation testing; validated it first on individual neurons and subjects before running all 554 neurons.

9. **Reproduced the firing-rate coding prevalence:** 17/554 neurons were FR-coding (**3.07% pooled**); subject-level mean = **3.34 ± 0.96%**, compared with the paper's **3.36 ± 0.71%**.

10. **Next:** complete the remaining arithmetic population-level Figure 1 analyses — FR vs temporal coding proportions (M), neuron-wise decoding accuracy comparison (N), regional tuned-neuron proportions (Q), and preferred-number distribution (S) — then begin the simplex analysis: centroid geometry, dimensionality/affine independence, paper tests, and alternative Gaussian/random controls.

11. # Number Simplex Reproduction — Short Progress Notes

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

- Temporal parameters are optimized **separately for each neuron** using a grid search over bin size and shrinkage `Gamma`; therefore, there is no single optimal bin size for all neurons.

- Important mistakes fixed so far include the `para-hpc` region-label error (**536 → 554 neurons**), different spike keys across files, YFR/YFS cue structure, MATLAB/Python indexing differences, and degenerate LDA folds.

- Figures are saved in `figures/`, while population results and summary DataFrames are saved in `tables/`.

- **Current checkpoint:** firing-rate reproduction is complete; the population-level temporal-code accuracy and FR-vs-temporal comparison are the next results to finalize.

- After completing Figure 1 reproduction, the next major goal is to test whether the **nine number centroids genuinely form a simplex-like geometry**, including alternative/random controls rather than forcing agreement with the paper.
