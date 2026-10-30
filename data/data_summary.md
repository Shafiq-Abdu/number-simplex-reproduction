# Dataset Summary

## 1. Dataset Overview

This dataset contains intracranial single-neuron recordings from **11 human subjects** performing numerical tasks.

| Feature | Summary |
|---|---|
| Number of subjects | **11** |
| Subjects | YFF, YFI, YFJ, YFK, YFL, YFM, YFP, YFR, YFS, YFT, YFU |
| Main tasks | Arithmetic task and dot/numerosity task |
| Total recorded neurons | **849 neurons** |
| MTL neurons used in the main analysis | **554 neurons** |
| Brain regions | Hippocampus (HPC), Entorhinal Cortex (EC), Amygdala (AMY), Parahippocampal Cortex (PHC) |
| Number classes | **1–9** |
| Main behavioral/event file | `photoBehavEvents.mat` |
| Main neural file | `spikes.mat` or task-specific spike files |
| Raw data directory | `data/raw/<subject>/<task>/` |

---

## 2. MTL Neurons by Subject

These are the neurons retained from the medial temporal lobe for the main population analyses.

| Subject | MTL Neurons | 100-ms Temporal Features |
|---|---:|---:|
| YFF | 37 | 333 |
| YFI | 29 | 261 |
| YFJ | 45 | 405 |
| YFK | 44 | 396 |
| YFL | 57 | 513 |
| YFM | 61 | 549 |
| YFP | 43 | 387 |
| YFR | 65 | 585 |
| YFS | 59 | 531 |
| YFT | 52 | 468 |
| YFU | 62 | 558 |
| **Total** | **554** | **4,986** |

For the 100-ms temporal representation:

```text
9 time bins × number of neurons = number of features
```

Example:

```text
YFF: 37 neurons × 9 bins = 333 features
```

---

## 3. MTL Neurons by Brain Region

| Region | Number of Neurons |
|---|---:|
| Hippocampus (HPC) | 389 |
| Entorhinal Cortex (EC) | 70 |
| Amygdala (AMY) | 77 |
| Parahippocampal Cortex (PHC) | 18 |
| **Total** | **554** |

> **Important:** The raw spike files can contain more neurons than the 554 neurons used in the MTL analysis.  
> For example, YFU contains **78 recorded neurons**, while **62 MTL neurons** are retained for the main analysis.

---

## 4. Arithmetic Task

All **11 subjects** contain arithmetic-task data.

### Main analysis window

```text
50 ms → 950 ms after operand onset
```

Total duration:

```text
900 ms
```

For the fixed 100-ms temporal representation:

```text
50–150 ms
150–250 ms
250–350 ms
350–450 ms
450–550 ms
550–650 ms
650–750 ms
750–850 ms
850–950 ms
```

This gives **9 temporal bins per neuron**.

### Numerical classes

```text
1, 2, 3, 4, 5, 6, 7, 8, 9
```

Each operand presentation is treated as a numerical presentation belonging to one of these nine classes.

---

## 5. Dot / Numerosity Task

The primary analysis window for the dot task is:

```text
100 ms → 1900 ms after stimulus onset
```

| Subject Group | Dot-task Neural Data |
|---|---|
| YFI, YFJ, YFR, YFS, YFT, YFU | `photoBehavEvents.mat` + `spikes.mat` |
| YFK, YFL, YFM | `photoBehavEvents.mat` + `spikesDot.mat` |
| YFF | Behavioral/event file available, but no dot-task spike file |
| YFP | Dot-task data absent |

Therefore, **9 subjects contain usable dot-task spike recordings**.

---

## 6. Raw File Structure

The general organization is:

```text
data/
└── raw/
    ├── YFF/
    │   ├── arithmetic/
    │   └── dot/
    ├── YFI/
    ├── YFJ/
    ├── YFK/
    ├── YFL/
    ├── YFM/
    ├── YFP/
    ├── YFR/
    ├── YFS/
    ├── YFT/
    └── YFU/
```

### Behavioral / Event Data

```text
photoBehavEvents.mat
```

contains the behavioral and event information needed to align the neural recordings with the experiment, including quantities such as:

- stimulus / operand onset times
- presented numbers
- arithmetic operation information
- responses
- trial information
- correctness
- exclusion indicators

For YFF arithmetic, a converted version is also available:

```text
photoBehavEvents.csv
```

The `.mat` file is the original data source.

---

## 7. Spike Data

Depending on the subject, neural recordings are stored as:

```text
spikes.mat
```

or

```text
spikesArithmetic.mat
spikesDot.mat
```

The spike files contain information such as:

```text
spikes
regionsVect
```

The spike matrix is stored in a **sparse representation**, so the raw neural data should not be interpreted as an ordinary dense trial × neuron table.

The basic processing structure is:

```text
Continuous spike recording
        ↓
Behavioral/event onset times
        ↓
Extract stimulus-specific time window
        ↓
Count/bin spikes for each neuron
        ↓
Construct presentation × neural-feature matrix
```

---

## 8. File Differences Across Subjects

### Arithmetic

| Subjects | Files |
|---|---|
| YFF | `photoBehavEvents.mat`, `photoBehavEvents.csv`, `spikes.mat` |
| YFI, YFJ, YFP, YFR, YFS, YFT, YFU | `photoBehavEvents.mat`, `spikes.mat` |
| YFK, YFL, YFM | `photoBehavEvents.mat`, `spikesArithmetic.mat` |

### Dot Task

| Subjects | Files |
|---|---|
| YFI, YFJ, YFR, YFS, YFT, YFU | `photoBehavEvents.mat`, `spikes.mat` |
| YFK, YFL, YFM | `photoBehavEvents.mat`, `spikesDot.mat` |
| YFF | `photoBehavEvents.mat` only |
| YFP | No dot-task data |

---

## 9. Example: YFF Arithmetic Data

YFF provides a useful reference for understanding the data structure.

| Feature | Value |
|---|---:|
| Behavioral trials | 100 |
| Behavioral variables | 22 |
| Arithmetic operations | Addition and subtraction |
| Number range analyzed | 1–9 |
| Number presentations after filtering | 109 |
| MTL neurons used | 37 |
| Temporal bins | 9 |
| Temporal population features | 333 |

The number of **presentations can exceed the number of trials** because an arithmetic trial contains multiple numerical operands.

---

## 10. Important Analysis Considerations

1. **Trial and presentation are not identical.**  
   One arithmetic trial can contribute multiple numerical presentations.

2. **Train/test leakage must be avoided.**  
   Presentations originating from the same trial should not be split between training and testing when trial dependence is relevant.

3. **Raw neuron count and MTL neuron count are different.**  
   The complete recordings contain more neurons, while the main analysis focuses on the selected MTL population.

4. **Spike-file formats are not completely uniform across subjects.**  
   Some subjects use `spikes.mat`, while others use task-specific spike files.

5. **The main arithmetic analysis window is fixed at 50–950 ms.**

6. **The current fixed temporal representation uses 100-ms bins**, giving nine temporal features per neuron.

7. **Nine numerical classes are analyzed:** numbers 1 through 9.

8. Behavioral/event information and spike recordings must always be aligned before constructing firing-rate or temporal neural features.

---

## 11. Dataset at a Glance

```text
11 subjects
│
├── Arithmetic: 11 subjects
│   ├── Numbers 1–9
│   ├── 50–950 ms analysis window
│   └── 554 MTL neurons
│
├── Dot task: 9 subjects with neural recordings
│   └── 100–1900 ms analysis window
│
├── Brain regions
│   ├── HPC: 389
│   ├── EC:   70
│   ├── AMY:  77
│   └── PHC:  18
│
└── Main data sources
    ├── photoBehavEvents.mat
    ├── spikes.mat
    ├── spikesArithmetic.mat
    └── spikesDot.mat
```

### Main analysis dataset

**11 subjects → 554 MTL neurons → 9 numerical classes → stimulus-aligned spike responses**

For the current 100-ms temporal representation:

**554 neurons across subjects → 4,986 subject-specific temporal neural features.**