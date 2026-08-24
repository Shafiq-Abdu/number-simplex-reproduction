# Number Simplex Reproduction :  Notes

**Paper:** *A number simplex in the human medial temporal lobe*   

https://www.biorxiv.org/content/10.64898/2026.06.25.734462v1
---

## 1. Data setup

- Working with **11 subjects**:
  - YFF, YFI, YFJ, YFK, YFL, YFM, YFP, YFR, YFS, YFT, YFU
- Raw neural data are MATLAB v7.3 (`.mat` / HDF5) files.
- Used `h5py` in Python to read spike matrices and region labels.
- Used MATLAB to inspect the behavioral tables, then exported them to CSV for easier processing in Python.
- Built the analysis independently from the Methods rather than using the authors' code.

### Arithmetic MTL neurons

| Region | Neurons |
|---|---:|
| HPC | 389 |
| ENT | 70 |
| AMY | 77 |
| para-HPC | 18 |
| **Total** | **554** |

This exactly matches the arithmetic-task MTL population reported in the paper.

---

## 2. Behavioral alignment

For the arithmetic task:

- Reconstructed Operand 1 and Operand 2 presentations from the behavioral event tables.
- Pooled **Operand 1 + Operand 2** presentations.
- Used number labels **1–9**.
- Used `operationFirst` to determine the cue order.

### Special case

YFR and YFS use a different cue structure:

- Operand 1 → `Cue1`
- Operation sign → `Cue2`
- Operand 2 → `Cue3`

This had to be handled separately from the other subjects.

---

## 3. Single-neuron reproduction

Started with the example neuron:

**YFU.hpc.43**

Reproduced the main ideas behind Figure 1D–H:

- firing-rate tuning across numbers 1–9
- mean ± SEM
- spike raster
- Gaussian-smoothed temporal firing
- temporal binning
- LDA temporal components
- TC1–TC2 projections
- number centroids

### Figures

Saved under `figures/`.

Main reproduced figures include:

- Figure 1D-style firing-rate tuning
- Figure 1E-style raster
- Figure 1F-style temporal firing response
- Figure 1G-style temporal/LDA representation
- Figure 1H-style TC1–TC2 projection

---

## 4. Firing-rate decoding

I rebuilt the firing-rate decoder independently.

For every presentation, the feature is simply:

> firing rate during the stimulus window

Then:

- labels = numbers 1–9
- stratified cross-validation
- uniform class treatment
- permutation/shuffle test for significance

Chance accuracy for nine classes is:

`1/9 ≈ 11.11%`

### Example: YFU.hpc.43

- Presentations: **494**
- FR decoding accuracy: **~11.13%**
- permutation p-value: **~0.478**
- Result: **not firing-rate coding**

This makes sense because the accuracy is essentially at chance.

---

## 5. Full firing-rate result

Ran the analysis across the complete **554-neuron arithmetic MTL population**.

| Result | Our reproduction |
|---|---:|
| Total neurons | 554 |
| FR-coding neurons | 17 |
| Pooled FR-coding proportion | 3.07% |
| Subject-level mean ± SEM | **3.34 ± 0.96%** |
| Paper | **3.36 ± 0.71%** |

So our subject-level mean differs from the paper by only **0.02 percentage points**.

This gives us confidence that the behavioral alignment, neuron selection, firing-rate calculation, CV and permutation pipeline are working correctly.

### Tables

Saved under `tables/`:

- `figure1_arithmetic_FR_all554.csv`
- `figure1M_arithmetic_FR_subject_summary.csv`

---

## 6. Temporal-code decoding

The temporal code keeps the **timing of spikes**, instead of reducing each trial to one firing-rate value.

For the arithmetic task, I bin the stimulus period using candidate bin sizes and run regularized LDA.

### Grid search

For each neuron:

- candidate bin sizes:
  - 60, 75, 90, 100, 150, 180, 225, 300, 450, 900 ms
- shrinkage:
  - `Gamma = {0.2, 0.5, 0.8}`
- stratified cross-validation
- uniform class prior
- zero-variance features removed

The important point is that there is **not one best bin size for the whole population**.

For each neuron, I test every:

`bin size × Gamma`

combination and choose the pair giving the **highest cross-validated decoding accuracy for that neuron**.

Therefore different neurons can have different optimal temporal resolutions.

### Temporal accuracy

**Current reproduced temporal result:**

- Mean temporal-code accuracy: **[ADD OUR FINAL VALUE]**
- Mean firing-rate accuracy: **[ADD OUR FINAL VALUE]**
- Improvement: **[ADD OUR FINAL VALUE]**
- Temporal coding neurons: **[ADD FINAL COUNT / %]**

These population-level temporal results still need to be finalized/checked before moving fully to the simplex analysis.

For reference, the paper reports that temporal decoding improves arithmetic single-neuron decoding accuracy by **3.79 percentage points**.

---

## 7. Mistakes / issues found during reproduction

These were useful because most of them were data-structure or implementation issues rather than conceptual problems.

### MATLAB file loading

**Problem:** `scipy.io.loadmat` did not work.

**Why:** the files are MATLAB v7.3/HDF5.

**Fix:** switched to `h5py`.

---

### Different spike keys

**Problem:** initially assumed every file contained the same `spikes` key.

**Why:** subjects/files do not all use exactly the same naming convention.

**Fix:** loader now handles both `spikes` and `spikesArithmetic` structures.

---

### Wrong region label

**Problem:** initially obtained only **536 MTL neurons**.

**Why:** I used `phc`, but the dataset label is `para-hpc`.

**Fix:** corrected the label.

Result:

`536 + 18 = 554 neurons`

This recovered the exact population reported in the paper.

---

### YFR / YFS cue structure

**Problem:** the standard cue reconstruction does not apply to YFR and YFS.

**Why:** the operation sign is always Cue2 for these subjects.

**Fix:** added subject-specific handling.

---

### MATLAB vs Python neuron indexing

**Problem:** neuron IDs initially looked inconsistent.

**Why:** MATLAB/paper indexing is 1-based, while Python is 0-based.

**Fix:** explicitly keep track of both indexing systems.

---

### LDA component sign

**Problem:** some temporal components looked flipped relative to the paper.

**Why:** an LDA direction can be multiplied by `-1` without changing the solution.

**Fix:** do not interpret a sign flip as a disagreement with the paper.

---

### FR decoder crash

**Problem:** the full 554-neuron run crashed around YFK.

**Why:** some sparse/degenerate firing-rate folds caused LDA/SVD failures.

**Fix:**

- handle zero-variation/degenerate folds
- handle invalid shuffle realizations
- retain problematic neurons instead of silently deleting them
- resume completed calculations instead of restarting everything

Final run completed all **554 neurons**.

Five neurons had undefined FR decoding and were explicitly flagged.

---

## 8. Current checkpoint

So far I have reproduced:

- [x] Data loading
- [x] Behavioral/spike alignment
- [x] Exact 554-neuron arithmetic MTL population
- [x] Operand 1 + Operand 2 pooling
- [x] Figure 1D-style firing-rate tuning
- [x] Raster / temporal firing analysis
- [x] Temporal binning
- [x] LDA temporal components
- [x] TC1–TC2 projections
- [x] Firing-rate decoding
- [x] Full 554-neuron FR analysis
- [x] Figure 1M firing-rate coding proportion
- [ ] Final population temporal-code analysis/check
- [ ] FR vs temporal decoding comparison
- [ ] Region-wise coding proportions
- [ ] Preferred-number analysis
- [ ] Simplex geometry analysis

---

## 9. Next step

Before testing the simplex claim, I want to finish and clean up the remaining Figure 1 population results.

After that, the main question becomes:

> **Do the nine number centroids actually form a simplex-like geometry?**

Things I plan to test:

- dimension spanned by the nine centroids
- affine independence
- pairwise/transition-angle structure
- comparison with random configurations
- comparison with Gaussian-generated configurations
- reproduce the paper's geometric tests
- add alternative tests instead of relying only on the paper's simplex metrics

The main principle for this reproduction is to **document differences rather than force the results to match the paper**.
