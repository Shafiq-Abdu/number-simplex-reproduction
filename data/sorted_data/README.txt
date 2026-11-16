Unmerged spike sorting: YFM, YFS, YFT
Prepared 2026-11-15

This package replaces the previously shared spike matrices with the new,
unmerged sorting. All source units and all behavioral rows are retained.
There is no MTL-region, tuning-p-value, or decoding-accuracy filter. No proposed
unit combinations have been applied. Unit row identity is shared across the
dot and arithmetic files of each subject.

The directory and spike-variable names match the previous MathData.zip:

YFM/
  dot/
    spikesDot.mat             (spike variable: spikesDot)
    photoBehavEvents.mat
  arithmetic/
    spikesArithmetic.mat      (spike variable: spikesArithmetic)
    photoBehavEvents.mat
YFS/
  dot/
    spikes.mat                (spike variable: spikes)
    photoBehavEvents.mat
  arithmetic/
    spikes.mat                (spike variable: spikes)
    photoBehavEvents.mat
YFT/
  dot/
    spikes.mat                (spike variable: spikes)
    photoBehavEvents.mat
  arithmetic/
    spikes.mat                (spike variable: spikes)
    photoBehavEvents.mat

ZIP entries start directly with these subject folders, as in MathData.zip.
Extract into a new data directory. The original shared data remain separate.

Spike MAT files
--------------
MATLAB v7.3. Spike matrices are sparse logical, units by 1-ms bins, using the
same bin-indexing convention as the previously shared files. These are the
stored millisecond-binned spikes, not sub-millisecond timestamp lists.
regionsVect remains an N-by-1 cell array of character vectors.

New fields, all matched to spike-matrix rows:
  isSingleUnit             N-by-1 uint8: 1 = meets the SU rule; 0 = does not.
  sourceUnitIndex          N-by-1 uint32, the original new-sorting row (1-based).
  electrodesVect           Original new-sorting electrode labels.
  isiViolationPct3ms       100 * isiViolationCount3ms / isiIntervalCount.
  isiViolationCount3ms     Number of within-recording ISIs <=3 ms, pooled.
  isiIntervalCount         Total within-recording ISIs, pooled.
  exportInfo               File schema, clock mapping, padding and source hash.

SU rule: inclusive ISI <=3 ms percentage <=2%, with at least one interval.
Calculate intervals separately within each physical recording, then add the
violation counts and interval counts. Do not concatenate timestamps before
calculating ISIs. This is an ISI-based label; it does not assert manual waveform
quality or significant number coding. The label vector is identical in the
two task files for each subject.

Subject       All units       isSingleUnit=1       isSingleUnit=0
YFM               198                   180                   18
YFS               244                   201                   43
YFT               174                   151                   23

Clock compatibility
-------------------
All six photoBehavEvents.mat files are byte-for-byte copies of the previously
shared files. The table values, column names, column types, order and trial
flags are unchanged. No new behavioral timestamp conversion is required.

YFM: arithmetic and dot use their separate, local recording clocks, matching
the earlier package. Arithmetic contains source columns 1:808864. Dot contains
source columns 808865:1645349, reindexed locally to start at column 1. One empty
column is appended at the end of dots to preserve the old 836486-bin length.
The exported arithmetic length is 808864 bins. The exact mapping for dot is
source column = exported column +808864, excluding the appended empty column.
This integer shift preserves the earlier round(event-time) bin alignment.

YFS and YFT: each task folder contains the same full neural recording on the
original shared/global clock; the task-specific behavior file selects its
events. Do not concatenate the dot and arithmetic spike matrices for these
subjects: they are copies of the same full recording. Full new lengths are
2215514 bins for YFS and 3167227 for YFT. They are 3 and 5 ms longer than the old
matrices; the extra tail bins are retained because they contain spikes.

Loading examples
----------------
% YFM dots:
D = load(fullfile(dataRoot,'YFM','dot','spikesDot.mat'));
B = load(fullfile(dataRoot,'YFM','dot','photoBehavEvents.mat'));
keep = D.isSingleUnit == 1;
spikesSU = D.spikesDot(keep,:);
regionsSU = D.regionsVect(keep);
sourceRowsSU = D.sourceUnitIndex(keep);
photoBehavEvents = B.photoBehavEvents;

% YFS/YFT, either task, use D.spikes instead of D.spikesDot.
% YFM arithmetic uses D.spikesArithmetic.
% Keep all rows by using the full spike matrix without the keep mask.

Validation
----------
All exported spike coordinates match the designated source ranges, with no
merging or row reordering. Fresh pooled ISI calculations agree for all 616
units. Native preprocessing of the exported files reproduces the previously
verified dots and arithmetic aligned spike counts exactly for every staged
unit and observation. Original raw files remain unchanged.
MANIFEST.csv records each file's dimensions, unit counts, clock mapping and
SHA256 hashes. SHA256SUMS.txt covers the 12 MAT files, this README and manifest.
