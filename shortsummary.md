## Progress Summary Before Simplex Analysis

1. **Loaded and organized data from all 11 patients** using `h5py`, including both `spikes.mat` and `spikesArithmetic.mat` formats.

2. **Recovered the exact 554 MTL neurons** reported in the paper: HPC = 389, ENT = 70, AMY = 77, para-HPC = 18.

3. **Aligned arithmetic trials with spike data** and pooled Operand 1 and Operand 2 presentations for numbers 1–9, including the special cue structure for YFR/YFS.

4. **Reproduced single-neuron firing-rate tuning** for the example neuron YFU.hpc.43 and calculated mean firing rate and SEM across number conditions.

5. **Visualized spike timing** using raster/spike-train plots and Gaussian-convolved firing rates, showing that numbers with similar mean firing rates can still have different temporal responses.

6. **Reproduced the temporal-component/LDA analysis** for the example neuron, including temporal-bin weights, LDA components, TC1/TC2 projections, number centroids, and response variability.

7. **Learned and verified the interpretation of LDA/TC analysis:** projection, discriminant directions, decision boundaries, dimensionality reduction, centroids, component sign ambiguity, and class overlap.

8. **Rebuilt firing-rate number decoding independently**, using stratified cross-validation and permutation testing, first validating individual neurons/subjects and then running all 554 MTL neurons.

9. **Reproduced the firing-rate coding prevalence:** 17/554 neurons were FR-coding (3.07% pooled); subject-level mean = **3.34 ± 0.96%**, compared with the paper's **3.36 ± 0.71%**.

10. **Next:** finish/check the remaining population-level Figure 1 comparisons (FR vs temporal coding, decoding, region and preferred-number analyses), then move to the simplex analysis: centroid geometry, affine independence/dimension, paper tests, and alternative/random Gaussian controls.