# Number Simplex Reproduction --- Progress Summary

**Project:** *A number simplex in the human medial temporal lobe*\
**Goal:** Independently reproduce the main neural coding analyses before
starting the simplex-geometry analysis.

## 1. Data setup and exploration

-   Set up the reproduction project in Python/Jupyter/VS Code.
-   Worked with data from **11 subjects**:
    -   YFF, YFI, YFJ, YFK, YFL, YFM, YFP, YFR, YFS, YFT, YFU.
-   MATLAB `.mat` files were MATLAB v7.3/HDF5, so `scipy.io.loadmat` was
    not suitable.
-   Used `h5py` to read spike matrices and region labels.
-   Used MATLAB to inspect behavioral tables because MATLAB displays the
    original structures more clearly.
-   Converted behavioral tables to CSV for easier use in Python.
-   Confirmed spike sampling rate and aligned behavioral event times
    with the spike timeline.
-   Handled both:
    -   `spikes.mat` with key `spikes`
    -   `spikesArithmetic.mat` with key `spikesArithmetic`

## 2. Important data checks

-   Recovered the paper's exact arithmetic MTL population:

  Region        Neurons
  ----------- ---------
  HPC               389
  ENT                70
  AMY                77
  para-HPC           18
  **Total**     **554**

-   Initially obtained **536 neurons** because `para-hpc` was mistakenly
    written as `phc`.
-   Correcting the region label recovered the missing **18 neurons** and
    gave the exact total **554**.
-   Confirmed subject-specific neuron counts before running the full
    analysis.
-   Built pooled arithmetic presentations from **Operand 1 + Operand
    2**.
-   Restricted number labels to **1--9**.
-   Confirmed that every subject contained all 9 number classes.

## 3. Behavioral alignment

-   Used `operationFirst` to determine which cue contained Operand 1,
    operation sign, and Operand 2.
-   Special handling was required for **YFR and YFS**:
    -   operation sign always occurred at `Cue2`
    -   Operand 1 = `tCue1`
    -   Operand 2 = `tCue3`
-   Presentation counts differed by subject, so cross-validation folds
    were allowed to adapt to the smallest number class.

## 4. Single-neuron firing-rate tuning --- Figure 1D

-   Reproduced the firing-rate tuning curve for the example neuron
    **YFU.hpc.43**.
-   Calculated mean firing rate for numbers 1--9.
-   Added SEM/error bars.
-   Compared number conditions such as **5 and 6**, which can have
    similar average firing rates.
-   This helped illustrate the main motivation of the paper:
    -   similar mean firing rates can hide differences in spike timing.

**Relevant figure/table names used during the work:** - YFU.hpc.43 mean
firing-rate tuning figure - `figure1M_YFU_FR_neuron_results.csv` -
`figure1M_YFU_FR_summary.csv` - `figure1M_YFF_FR_neuron_results.csv` -
`figure1M_YFF_FR_summary.csv`

Tables are stored under the project `tables_dir`.

## 5. Raster plots and spike trains --- Figures 1E/F concepts

-   Learned that a raster plot is a direct display of spike times across
    repeated trials.
-   Each small mark corresponds to a spike.
-   Replotted spike trains in a cleaner signal-like format with
    horizontal trial lines.
-   Used Gaussian convolution to transform discrete spike events into a
    smooth time-resolved firing-rate signal.
-   Compared numbers **5 and 6** for YFU.hpc.43:
    -   their average firing rates were similar,
    -   but their time-resolved responses differed.
-   This provided an intuitive example of why temporal information may
    contain information lost by a single average firing rate.

**Relevant figures:** - YFU.hpc.43 raster/spike-train plots - YFU.hpc.43
Gaussian-convolved temporal response - Numbers 5 vs 6
firing-rate/temporal-response comparison

Figures are stored under the project `figures_dir`.

## 6. Temporal representation and LDA --- Figures 1G/H

-   Divided the neural response into temporal bins.
-   Represented each presentation using firing rates across the bins
    rather than one single firing-rate value.
-   Applied multiclass Linear Discriminant Analysis (LDA).
-   Learned the meaning of:
    -   projection
    -   discriminant direction
    -   decision boundary
    -   LDA components
    -   temporal components (TC1, TC2, ...)
    -   centroids
    -   eigenvalues/eigenvectors
    -   dimensionality reduction
-   Clarified that an LDA/TC component does **not** correspond to one
    number.
-   TC1 and TC2 are directions that summarize discriminative temporal
    patterns.
-   Projected number responses into the TC1--TC2 plane.
-   Calculated the centroid of each number class.
-   Added ellipses/circles around number clusters to visualize
    variability/overlap.
-   Learned why overlap is expected: neural responses to different
    numbers are variable and are not perfectly separable.
-   Noted that component signs can flip without changing the underlying
    solution; LDA/eigenvector directions are sign-indeterminate.

**Relevant table:** - `figure1H_YFU_hpc43_TC_centroids.csv`

**Relevant figures:** - YFU.hpc.43 temporal discriminant/LDA weights -
YFU.hpc.43 all temporal LDA components - YFU.hpc.43 TC1--TC2
number-centroid plot - YFU.hpc.43 TC1--TC2 plot with variability
ellipses

## 7. Firing-rate decoding

-   Rebuilt the firing-rate-only decoder separately from the temporal
    decoder.
-   Used:
    -   one firing-rate feature per presentation
    -   number labels 1--9
    -   stratified cross-validation
    -   permutation/shuffle testing
-   Chance level for nine numbers is approximately:

\[ 1/9 `\approx 11.11`{=tex}% \]

-   Validated the method first on individual neurons and then on
    individual subjects.

### Example validation

**YFU.hpc.43** - 494 pooled presentations. - FR accuracy ≈ **11.13%**. -
Permutation p ≈ **0.478**. - Classified as **not FR coding**.

### Subject checks

-   **YFU**
    -   62 MTL neurons
    -   4 FR-coding neurons
    -   **6.45%**
-   **YFF**
    -   37 MTL neurons
    -   1 FR-coding neuron
    -   **2.70%**
-   **YFI**
    -   29 MTL neurons
    -   3 FR-coding neurons
    -   **10.34%**

These checks were used to validate the pipeline before running all
neurons.

## 8. Full arithmetic firing-rate analysis --- Figure 1M

-   Ran firing-rate decoding across all **554 MTL neurons**.
-   Final results:

  Result                                     Value
  ------------------------------------ -----------
  Total neurons                                554
  FR-coding neurons                             17
  Pooled FR-coding percentage            **3.07%**
  Neurons with undefined FR decoding             5

-   Calculated subject-level coding percentages.
-   Our subject-level result:

\[ `\boxed{3.34\% \pm 0.96\%}`{=tex} \]

-   Paper value:

\[ 3.36% `\pm 0.71`{=tex}% \]

-   The reproduced mean differs from the paper by only **0.02 percentage
    points**.
-   This is a strong reproduction of the firing-rate side of Figure 1M.

**Saved table:** - `figure1_arithmetic_FR_all554.csv` -
`figure1M_arithmetic_FR_subject_summary.csv`

Stored under `tables_dir`.

## 9. Problems/mistakes encountered and fixed

-   `scipy.io.loadmat` could not read MATLAB v7.3 files → switched to
    `h5py`.
-   MATLAB tables were difficult to interpret directly in Python →
    inspected/exported them through MATLAB.
-   Different subjects used different spike filenames/keys → loader
    updated to detect both formats.
-   Initially used the wrong region name `phc` → corrected to
    `para-hpc`.
-   Initially recovered 536 instead of 554 MTL neurons → correction
    recovered all 18 para-HPC neurons.
-   YFR/YFS had a different cue arrangement → handled separately.
-   MATLAB/Python neuron indexing caused confusion → kept track of
    Python 0-based vs paper/MATLAB 1-based indexing.
-   LDA component signs sometimes appeared reversed relative to the
    paper → learned that component direction can be multiplied by -1
    without changing the representation.
-   Full 554-neuron run crashed around YFK because some sparse
    firing-rate data caused an SVD-LDA failure.
-   Updated the FR decoder to handle degenerate/zero-variation folds and
    invalid shuffled realizations.
-   Resumed the analysis rather than restarting the completed neurons.
-   Final batch completed all 554 neurons.
-   Five neurons produced undefined FR decoding; these were retained and
    flagged rather than silently discarded.

## 10. Main concepts learned

-   Spike train vs raster plot.
-   Mean firing rate vs temporally resolved firing.
-   Gaussian convolution of spikes.
-   Why two conditions can have similar mean firing rates but different
    temporal responses.
-   Temporal binning.
-   LDA as projection + dimensionality reduction + classification.
-   Discriminant direction vs decision boundary.
-   TC1/TC2 and higher temporal components.
-   Class centroids and cluster overlap.
-   Stratified cross-validation.
-   Permutation/shuffle significance testing.
-   Subject-level percentages vs pooled-neuron percentages.
-   Why temporal coding can reveal information missed by a single
    firing-rate value.

## 11. What has been reproduced so far

-   Data loading and anatomical region counts.
-   Exact **554-neuron arithmetic MTL population**.
-   Behavioral/spike alignment.
-   Operand 1 + Operand 2 pooled presentation construction.
-   Example single-neuron firing-rate tuning (Figure 1D style).
-   Raster/spike-train visualization (Figure 1E/F concepts).
-   Gaussian-convolved temporal responses.
-   Temporal bin representation.
-   LDA/temporal-component weights (Figure 1G style).
-   TC1--TC2 number projections and centroids (Figure 1H style).
-   Temporal decoding explored earlier.
-   Firing-rate decoding rebuilt independently.
-   Full 554-neuron FR analysis.
-   Firing-rate side of Figure 1M reproduced closely:
    -   **ours: 3.34 ± 0.96%**
    -   **paper: 3.36 ± 0.71%**

## 12. Remaining work before simplex analysis

-   Finish/check the population-level temporal-component results needed
    for the remaining Figure 1 comparisons.
-   Reproduce/organize the relevant arithmetic panels among Figure
    1M--T:
    -   FR vs temporal-component coding proportion.
    -   neuron-wise decoding comparison.
    -   region-wise tuned-neuron proportions.
    -   preferred-number distribution.
-   Save final clean figures and summary tables.
-   Document any differences from the paper rather than forcing exact
    agreement.
-   After the Figure 1 reproduction is sufficiently checked, move to the
    main geometry question.

## 13. Next stage --- simplex analysis

The next research stage is to test the paper's main geometric claim
using the independently reconstructed neural representations.

Planned questions include:

-   Do the nine number centroids form the claimed simplex-like geometry?
-   What dimension do the centroids span?
-   Are the centroid vectors affinely independent?
-   How strong is the simplex structure compared with alternative/random
    configurations?
-   What happens for Gaussian/randomly generated centroid
    configurations?
-   Reproduce the paper's geometric tests and add alternative tests
    suggested by the advisor.

------------------------------------------------------------------------

**Current checkpoint:** The preprocessing, example-neuron temporal
analyses, LDA/TC interpretation, and full arithmetic firing-rate
analysis are working. The exact 554-neuron MTL population was recovered,
and the firing-rate coding prevalence closely reproduces the paper. The
next goal is to finish the remaining population-level Figure 1 checks
before beginning the simplex analysis.
