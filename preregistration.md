# Preregistration: defect-free commissioning of PatchCore

Status: frozen before the first full MVTec run. After that run, this file is not edited. Anything added later is marked exploratory in the report.

Code: `python -m commissioning`. The archived notebook `archive/projekt_v3_0s_poprawione.py` is not imported.

## Topic

Title: **How Many Good Parts Are Enough? Defect-Free Commissioning of Memory-Bank Anomaly Detection**

A line that has just been installed has no catalog of defects. It has a stream of parts that the process currently treats as good. The paper asks how to commission a PatchCore memory bank from that stream alone: how many good images to store, how to set the reject threshold without ever seeing a defect, and when collecting further good images stops changing the bank.

The contribution is the commissioning protocol, not a new detector. PatchCore is used as specified below and is not modified. The protocol has three parts. The memory bank grows only from good images held out from calibration. The threshold is the maximum score on a fixed calibration set of 20 other good images, which has a marginal false-reject level of 1/21 under exchangeability. Image-level diversity selection is compared with a random nested bank, and a normal-only stopping rule is compared with the observed threshold N*.

What the paper can claim even if the two selection strategies do not differ: the dependence of TPR and realized FPR on the number of good images, the gap between the realized FPR and 1/21, and the per-class observed threshold N*, including classes where that threshold is never reached. There is no universal N. N* is an observed commissioning threshold for a class and a repetition design. It is not a certified sample size for a production line. The camera–computer–decision sketch in the paper is a concept. The protocol was not run on a line.

Target: Applied Sciences, section Computing and Artificial Intelligence, or Applied Industrial Technologies. The manuscript is in English, MDPI format, about 12–16 pages.

## 1. Questions

RQ1. With the threshold fixed from good images only, how do TPR and the realized FPR change as the memory bank grows through N ∈ {1, 2, 4, 8, 16, 32, 64, |P|}? For which classes, and in how many of the 20 repetitions, is TPR ≥ 0.95 and FPR ≤ 0.05, and from which N onward?

RQ2. Across the 20 repetitions, how far is the realized FPR from the marginal level 1/21 at each N? The comparison uses the mean, median, and spread of the 20 realized rates. It does not treat those repetitions as independent Bernoulli trials.

RQ3. At the same N and the same split, does farthest-first selection on global-average-pooled features improve on a random nested bank in AULC_TPR, AULC_FPR, N*, and spread? The pre-specified practical rule is in section 5. A non-significant test is not evidence that the strategies are equivalent.

RQ4, secondary. Does the median score on the calibration set C stabilize along the grid, and how does that stopping point compare with the retrospective N*? A negative answer does not invalidate RQ1–RQ3.

Out of scope: coreset subsampling inside the RQ runs, reflection masking, PaDiM, EfficientAD, CPU timing, projection to 1024 dimensions, the 0.604 and 1.000 results from earlier drafts, and deployment on a real line.

## 2. Data and splits

All 15 MVTec AD classes. For every class and repetition, C is exactly 20 training images drawn from `train/good`, P is the rest of `train/good`, and the test set is the official `test` directory.

C is used for the threshold and for the stopping statistic. C never enters the memory bank. P is the only source of bank images. The test set is not used to set the threshold, to choose images, or to choose N_stop.

Assertions, checked in code: |C| = 20, |C| + |P| = |train|, C, P, and test are disjoint, and no two files in the class have the same SHA-1. Official train and good-test counts are hard-coded. A mismatch stops the run. Splits are written once to `splits/{class}/rep_XX.json` and are not overwritten if a later config hash disagrees.

C and P change across repetitions. Inside one repetition they are identical for both strategies and for every N. Seeds are `SeedSequence([42, class_index, repetition])`, split into independent streams for the split, the random permutation, and the first k-center image. Class indices follow the order in `config.yaml` and do not change when a subset of classes is launched.

## 3. Grid, repetitions, strategies

N ∈ {1, 2, 4, 8, 16, 32, 64, |P|}. Points with N > |P| are dropped. Toothbrush has 60 training images, so |P| = 40 and N = 64 is dropped. The last point is |P| even when it is not a doubling: 64 → 193 for transistor, 32 → 40 for toothbrush.

20 paired repetitions and 2 strategies: 15 × 20 × 2 = 600 nested trajectories, up to 4800 rows. There is no extra set of 10 draws.

Random: one permutation of P per repetition. The bank of size N is the first N images. The banks are nested.

K-center: farthest-first on 1536-dimensional global average pooled features. The first image is random, then each next image maximizes the distance to the nearest already chosen image. The text calls this image-level diversity selection using global average pooled features. It is not an optimal selection of local PatchCore descriptors, and it is not the PatchCore coreset.

## 4. PatchCore used for RQ1–RQ4

Wide-ResNet-50-2, layers `layer2` and `layer3`, ImageNet weights from the same timm tag Anomalib's `TimmFeatureExtractor` uses when Anomalib is installed. Input pipeline, matching `Patchcore.configure_pre_processor`: resize to 256×256 (bilinear, antialias), center crop 224×224, ImageNet normalization with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).

Each map is average-pooled with a 3×3 kernel, stride 1, padding 1. `layer3` is bilinearly upsampled (`align_corners=False`) to the 28×28 grid of `layer2`, then concatenated. The descriptor is 784 patches × 1536 dimensions. No coreset and no dimensionality reduction in the RQ runs.

The image score is

s(x | S) = max over patches p of min over bank images j in S of ||f_p(x) − m||₂

where the inner minimum runs over patches m of image j. Equivalently, with

D[q, p, j] = min over patches m of image j of ||f_p(q) − m||₂,

s(q | S) = max over p of min over j in S of D[q, p, j].

An image is rejected when s > τ, strict. A tie with the calibration maximum is not a reject. Scores are never inverted and never min-max scaled before the decision. A non-finite score or a corrupt image raises `RuntimeError` with the class name and the path. It is not replaced by zero.

Patches are stored as float16, one class at a time. Distances and scores are computed in float32. The self-distance of a training image is stored as +∞ so an image cannot become its own neighbor; C and P are disjoint, so this entry is not used by a valid bank.

The RQ configuration does not have to match E0a. Each comparison inside E0 uses the same image score on both sides of that comparison.

## 5. Calibration, metrics, AULC, N*

Threshold: n = 20, α = 0.05, k = ceil((n+1)(1−α)) = 20, so τ = max s(C | S).

Reading of the threshold, for the paper: with continuous scores and exchangeability, the probability that a new good image exceeds the maximum of 20 calibration scores is 1/21 ≈ 4.76%. Ties make the procedure conservative. The guarantee is marginal. It covers the randomness of the calibration set and of a new good image. It does not control the FPR inside every repetition or for a chosen production class.

Scale of that caveat: conditional on C, the true exceedance rate is Beta(1, 20). The mean is 1/21, the median is about 3.4%, and with probability 0.95^20 ≈ 0.358 the true rate exceeds 5%. About one third of calibration draws can therefore have a true FPR above 5% even under perfect exchangeability. That is the reference for RQ2, not a bug.

For every class, strategy, N, and repetition the report stores: point TPR with a one-sided lower 95% Clopper–Pearson bound, point FPR with a one-sided upper 95% bound, image AUROC, and the oracle FPR at TPR = 0.95. The oracle uses test labels and is not an operational method. Clopper–Pearson lower/upper bounds use the beta quantiles: lower is 0 at zero successes, otherwise Beta(α; k, n−k+1); upper is 1 at k = n, otherwise Beta(1−α; k+1, n−k).

FPR across repetitions (RQ2): for each N, class, and strategy, the mean and median of the realized FPR over 20 repetitions, the IQR, the min–max, and the fraction of repetitions with point FPR > 0.05. The Clopper–Pearson interval stays attached to a single test result. It does not represent uncertainty from drawing C and P; the 20 repetitions do. All 20 repetitions use the same test images, so the 20 × n_good trials are not pooled into one Clopper–Pearson interval.

AULC uses the log2 N axis and the trapezoid rule, divided by log2(32) − log2(1) = 5. The main grid is N ∈ {1, 2, 4, 8, 16, 32}, shared by every class. N = 64 and N = |P| are an extended analysis and do not enter the main AULC. The reported class value is the mean of the 20 per-repetition AULCs, which equals the AULC of the mean curve because the trapezoid is linear. AULC_TPR is higher-better. AULC_FPR is lower-better.

Practical rule for k-center, fixed before the run: k-center is called favorable when the median across classes of AULC_TPR(k-center) − AULC_TPR(random) is positive and the median of AULC_FPR(k-center) − AULC_FPR(random) is at most +0.01. The margin is a practical rule, not a formal non-inferiority test. Significance tests are reported separately.

N* is the smallest grid N at which at least 16 of 20 repetitions have TPR ≥ 0.95 and FPR ≤ 0.05, and every larger grid point also meets that count. If the full pool never does, N* is right-censored, reported as > |P|, and the class stays in the analysis. N* is an observed commissioning threshold, not a certified sample size. For each N the report also gives the fraction of repetitions that meet the criterion.

## 6. Stopping rule (RQ4)

m(N) is the median score of the 20 calibration images given bank B_N.

Δ(N_i) = |m(N_i) − m(N_{i−1})| / |m(N_{i−1})|,

where N_{i−1} is the previous grid point, not N/2. If the previous median is 0, Δ is 0 when the current median is 0 and infinite otherwise.

N_stop is the grid point that ends the second consecutive transition with Δ < δ. Example: if 8 → 16 and 16 → 32 both qualify, N_stop = 32. The rule is evaluated only on the grid. There is no interpolation. If it never fires, N_stop is missing.

Sensitivity: δ ∈ {0.025, 0.05, 0.10}, with 0.05 as the main value.

Reported: N_stop / N* where both exist, the fraction of premature stops, the extra images collected, and the cases where N* was not reached. Categories are premature (N_stop < N*), concordant (equal), excessive (N_stop > N*), N* not reached, and rule not fired.

Limitation, to be written in the paper: the rule uses the same C as the threshold, so the stopping statistic and the threshold can be optimistically coupled.

## 7. False-reject budget

The largest k with k / n_good ≤ 0.05 on the official good test images is floor(0.05 × n_good), computed as (5 × n_good) // 100. The expected table is toothbrush 0 of 12, wood 0 of 19, one error for bottle, grid, metal_nut, capsule, pill, carpet, leather, zipper, and tile, two errors for hazelnut, screw, and cable, and three for transistor. The code recomputes this from the official counts and checks it.

With zero errors the upper one-sided 95% Clopper–Pearson bound is 1 − 0.05^(1/n). It falls below 0.05 only for n ≥ 59: about 0.0503 at n = 58 and 0.0487 at n = 60. Among MVTec classes only transistor has that many good test images. These are observed counts. They do not certify an industrial FPR.

## 8. E0 gate

RQ does not start unless `e0/gate.json` has status PASS for the current config hash. `--exploratory` can override that lock; every row and the report are then marked exploratory.

E0a. Anomalib PatchCore in the reference configuration: the same preprocessing, Wide-ResNet-50-2, `layer2` + `layer3`, coreset ratio 0.1, `num_neighbors = 9`, including the paper's reweighted image score. The published mean image AUROC of about 0.991 is the sanity reference. The operational floor is a mean AUROC of at least 0.97 across the classes that were run. This is a sanity floor, not a claim of exact reproduction.

E0b. The feature cache versus Anomalib, full memory bank on both sides, `num_neighbors = 1`. Both sides use the raw maximum patch distance. Anomalib's coreset sampler is not run: `subsample_embedding` at ratio 1 still executes k-center over every patch and only returns a permutation, at quadratic cost. The bypass sets `memory_bank` to the stacked embeddings. The installed Anomalib version and the source of `subsample_embedding` are written to `e0/sampler_source.txt` before this comparison.

Criteria for E0b: the absolute difference of the mean image AUROC is at most 0.005, and Spearman ρ of the test scores is at least 0.99 inside each class. One correlation pooled across classes is not used. The report gives the median and the minimum ρ over the 15 classes. Classes that fail are diagnosed before any RQ run.

Integrity checks: the same file order, labels, tensor shape 784 × 1536, no duplicate files, no overlap of C, P, and test, no NaN or Inf, a corrupt file raises `RuntimeError`, and the distance tensor matches a direct 1-NN on transistor repetition 0 at N ∈ {1, 8, |P|} for held-out queries (or on the first available class if transistor was not requested).

## 9. Statistics

Two primary tests, on the 15 per-class paired differences (k-center minus random): the exact two-sided Wilcoxon signed-rank test for AULC_TPR and for AULC_FPR. Zero differences are dropped by the Wilcoxon procedure. If every difference is zero, the p-value is reported as 1.

Holm correction is applied to these two primary p-values together. It is not reserved for secondary comparisons. This protocol has no additional confirmatory tests. N* is described with the censoring flag and is not put into a mean.

Sensitivity analysis, not the primary test: an exact two-sided sign-flip test of the mean paired difference, enumerating all 2^15 sign assignments.

The bootstrap percentile interval of the median paired difference resamples classes with replacement, 10 000 times, with fixed seeds (42 for TPR, 43 for FPR). With 15 classes it is unstable. The paper presents it as an uncertainty display, not as strong evidence. A non-significant test is not evidence that the strategies are equivalent.

The report also states the median difference, its sign, and the number of classes on which each strategy was better.

## 10. Exploratory display, declared before the run

The fraction of repetitions with sample FPR > 0.05 is plotted against 0.358. Because each class has a small good test set, that fraction is not the same thing as the probability that the true conditional FPR exceeds 0.05. The report may also show the beta-binomial probability P(K/n > 0.05) for K ~ BetaBinomial(n, 1, 20). That column is exploratory. It is not a gate and it does not change N*, AULC, or the primary tests.

## 11. What a finished run writes

`cache/features/{class}.npz` and `.json`, `cache/dist/{class}.npy`, `runs/parts/*.parquet`, `runs/scores/*.npz`, `analysis/summary.csv`, `analysis/statistics.json`, and `report/index.html` with PNG and PDF figures and CSV/LaTeX tables. Each artifact stores the config hash. Finished trajectories are not recomputed. A file from another hash is not reused.
