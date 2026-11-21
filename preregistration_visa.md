# Preregistration v2: confirmatory replication on VisA

Status: **FROZEN** on 20 November 2026, before the first VisA image is scored by the pipeline. After this line, this file is not edited. Anything added later is marked exploratory.

Relation to v1: `preregistration.md` (MVTec AD, config hash `ec70992336b4c4a4`, E0 PASS, 600 trajectories) and `exploratory_addendum.md` stay frozen and are reported first and unchanged. This document turns the main exploratory findings of the addendum into confirmatory hypotheses on a second dataset that the v1 analysis never touched.

Prior exposure, declared before freezing: the authors have not computed PatchCore scores on VisA before.

Code: `python -m commissioning.visa`. The MVTec package path, `config.yaml`, and the v1 hash are not modified; a test checks that the v1 hash is still `ec70992336b4c4a4`.

## Why a second preregistration

The MVTec run produced three results that a single dataset cannot settle:

1. Image-level k-center selection beat a random nested bank in AULC_TPR on 13 of 15 classes (Holm p = 0.0005) without an FPR penalty. This is the confirmatory result to replicate.
2. The criterion behind N* (16 of 20 repetitions with TPR ≥ 0.95 and FPR ≤ 0.05, threshold max of 20) was infeasible by construction: with 12–60 good test images even a perfect detector is expected to pass in only 10–15 of 20 repetitions (addendum E1). 13 of 15 classes were censored.
3. Where good train and good test scores were exchangeable, the realized FPR matched the Beta(1, 20) theory (mean 0.051–0.057 vs 1/21, fraction above 5% 0.31–0.40 vs 0.32). Where they were not (carpet, leather, wood, partly zipper), the FPR was 0.18–0.28 (addendum E3).

v2 therefore (a) replicates the k-center comparison, (b) adds a larger calibration set whose feasibility is checked analytically before the run (section 8), and (c) adds a large held-out set of exchangeable good images, so that the calibration error and the train/test shift can be measured separately.

## 1. Questions and hypotheses

H1, primary (replication of v1 RQ3). With the v1 threshold R20, does farthest-first selection on GAP features give a higher AULC_TPR than a random nested bank? Prediction: positive median paired difference.

H2, primary (replication of v1 RQ3). Under R20, does k-center change AULC_FPR? Prediction: no practically relevant increase (the v1 margin of +0.01).

H3, primary estimation, no significance test. At N = |P|, what TPR does the larger calibration set cost? Because max(C60) ≥ max(C20) by construction, TPR(R60) ≤ TPR(R20) in every trajectory, so a test of the sign is uninformative. The pre-specified practical rule: R60 is called **recommended** when the median over classes of the mean (over 20 repetitions) TPR(R20) − TPR(R60) is at most 0.03.

Pre-declared predictions, descriptive, not tests:

P1 (feasibility). With R20, N* (official test) is censored for most classes, as section 8 predicts. With R60, N* is reached for every class whose TPR at N = |P| reaches 0.95 in at least 16 of 20 repetitions.

P2 (calibration under exchangeability). On the held-out good set H: the median over classes of the mean realized FPR (mean over every grid point, both strategies, and 20 repetitions; every bank is drawn from P, so C and H stay exchangeable given the bank) lies in [0.03, 0.07] for R20 (theory 1/21 ≈ 0.048) and in [0.005, 0.03] for R60 (theory 1/61 ≈ 0.016). The fraction of repetitions with FPR_H > 0.05 is compared with 0.358 (R20) and 0.046 (R60).

P3 (shift). A class whose official good test scores differ from its held-out good scores (section 9, Holm p < 0.05) has a higher mean FPR on the official good test than on H under R20 (same averaging as P2).

Secondary: the v1 stopping rule (RQ4) and the budget split (section 10).

Out of scope, as in v1: coreset inside the RQ runs, other detectors or backbones, CPU timing, pixel-level metrics, deployment on a line.

## 2. Data

VisA (Zou et al., ECCV 2022, CC BY 4.0), archive `VisA_20220922.tar`, the official one-class split `split_csv/1cls.csv`. All 12 classes, in this order (seed index 0–11): candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum. Groups for figures (from the VisA paper): multiple instances (candle, capsules, macaroni1, macaroni2), single instance (cashew, chewinggum, fryum, pipe_fryum), complex structure (pcb1–pcb4).

Official counts, hard-coded; a mismatch stops the run:

| class | train good | test good | test anomaly | official FR budget | |H| |
|---|---|---|---|---|---|
| candle | 900 | 100 | 100 | 5 | 600 |
| capsules | 542 | 60 | 100 | 3 | 242 |
| cashew | 450 | 50 | 100 | 2 | 150 |
| chewinggum | 453 | 50 | 100 | 2 | 153 |
| fryum | 450 | 50 | 100 | 2 | 150 |
| macaroni1 | 900 | 100 | 100 | 5 | 600 |
| macaroni2 | 900 | 100 | 100 | 5 | 600 |
| pcb1 | 904 | 100 | 100 | 5 | 604 |
| pcb2 | 901 | 100 | 100 | 5 | 601 |
| pcb3 | 905 | 101 | 100 | 5 | 605 |
| pcb4 | 904 | 101 | 100 | 5 | 604 |
| pipe_fryum | 450 | 50 | 100 | 2 | 150 |

The raw archive is converted once into the MVTec layout (`{class}/train/good`, `{class}/test/good`, `{class}/test/bad`) by copying files listed in `1cls.csv`; nothing is resized or re-encoded at this step.

Duplicates (SHA-1 of file content), decided before any score exists: a duplicate across train and test, or between test good and test anomaly, stops the run and is reported. Duplicates inside train/good keep the first file in sorted path order and drop the rest. Duplicates inside test good or inside test anomaly are reported and kept, because the official test set is used unchanged. The official counts above are checked before removal; |H| is then the deduplicated train/good count minus 300. Everything is listed in `data_check.json`. The data check runs before feature extraction and does not compute scores.

## 3. Commissioning universe, splits, grid

For each class, once: the commissioning universe U is 300 images drawn without replacement from train/good with `SeedSequence([42, 1000 + class_index])`. The held-out good set H is train/good minus U (150–605 images). H never enters a bank or a threshold. U and H are fixed across repetitions.

For each class and repetition r = 0..19, `SeedSequence([42, 1000 + class_index, r])` is spawned into independent streams for the split, the random permutation, and the first k-center image. From the split stream, C60 is an ordered random sample of 60 images of U; C20 is the first 20 images of C60, so C20 ⊂ C60. The pool is P = U \ C60, |P| = 240 for every class.

Assertions, checked in code: |U| = 300, |C60| = 60, |C20| = 20, C20 ⊂ C60, |P| = 240, and U, H, and the official test set are pairwise disjoint. Splits are written once to `splits/{class}/rep_XX.json` and never overwritten.

Grid: N ∈ {1, 2, 4, 8, 16, 32, 64, 128, 240}. 12 classes × 20 repetitions × 2 strategies = 480 nested trajectories, 4320 rows per threshold rule. Random and k-center are defined as in v1 section 3; k-center runs on P only.

## 4. PatchCore

Identical to v1 section 4: Wide-ResNet-50-2 (same timm weights), `layer2` + `layer3`, 3×3 average pooling, bilinear upsampling, 784 × 1536 descriptors, no coreset, image score = max over patches of the min distance to the bank, reject when strictly above the threshold. Preprocessing as v1: resize to 256 × 256 (aspect ratio not preserved; VisA images are about 1.5k × 1k), center crop 224, ImageNet normalization. This matches the Anomalib default and is a stated limitation: the crop removes about 6% of each border.

The distance tensor has U as columns (300) and U ∪ H ∪ test as rows.

## 5. Threshold rules

Both rules are evaluated on the same bank in every trajectory, so the comparison is paired.

- R20: τ = max s(C20 | S). This is the v1 rule (k = ceil(21 × 0.95) = 20).
- R60: τ = max s(C60 | S). With continuous scores and exchangeability the conditional FPR is Beta(1, 60): mean 1/61, P(FPR > 0.05) = 0.95^60 ≈ 0.046. R60 is also the PAC rank of the addendum at m = 60, ε = 0.05, δ = 0.10 (the rank 59 gives 0.19 > δ).

## 6. Metrics

For every class, rep, strategy, N, and rule:

- TPR on official test anomalies, with the one-sided lower 95% Clopper–Pearson bound;
- FPR_off on official good test images and FPR_H on H, each with the one-sided upper 95% bound;
- image AUROC and oracle FPR at TPR = 0.95 on the official test set (rule-independent);
- criterion_off: TPR ≥ 0.95 and false rejects ≤ floor(0.05 × n_good_test) on the official good test;
- criterion_H: TPR ≥ 0.95 and false rejects on H ≤ floor(0.05 × |H|).

AULC: as v1, trapezoid on log2 N over {1, 2, 4, 8, 16, 32}, divided by 5, mean over 20 repetitions.

N*: as v1 (smallest grid N from which at least 16 of 20 repetitions meet the criterion at every larger grid point; otherwise right-censored and reported as > |P|). Reported for every rule × criterion combination. The headline N* is **R60 with criterion_off**, the combination that section 8 shows to be feasible.

## 7. Statistics

H1 and H2: the exact two-sided Wilcoxon signed-rank test on the 12 per-class paired differences (k-center minus random) of AULC_TPR and AULC_FPR (R20, official test). Holm over these two p-values. The v1 practical rule for k-center is reused unchanged: favorable when the median ΔAULC_TPR > 0 and the median ΔAULC_FPR ≤ +0.01. Sign-flip test (2^12 assignments) and the bootstrap of the median (10 000, seeds 42 and 43) as sensitivity displays. With 12 classes the smallest attainable two-sided Wilcoxon p is 2/4096 ≈ 0.0005.

The same two tests repeated with R60 and with FPR_H are reported as secondary.

H3: the per-class mean of TPR(R20) − TPR(R60) at N = |P| (random strategy; at |P| both strategies give the same bank), its median over classes against 0.03, and a class bootstrap interval of the median (10 000, seed 44). The same quantity at N = 32 is secondary.

A non-significant test is not evidence of equivalence.

## 8. Feasibility before the run

For a detector with TPR = 1 and exchangeable good scores, P(a repetition meets the FPR budget) = BetaBinomialCDF(budget; n, 1, m), where m is the size of the calibration set. Expected passing repetitions of 20, and P(at least 16 of 20) under an independence approximation (repetitions share the test set):

| test set | rule | expected of 20 | P(≥ 16) |
|---|---|---|---|
| official good, n = 50 | R20 | 12.8 | 0.10 |
| official good, n = 100–101 | R20 | 13.4–13.5 | 0.16–0.17 |
| official good, n = 50–101 | R60 | 18.2–18.9 | 0.97–1.00 |
| H, |H| = 150–605 | R20 | 12.7–13.1 | 0.09–0.12 |
| H, |H| = 150–605 | R60 | 18.7–19.0 | 0.99–1.00 |

So N* with R20 is predicted to be censored by design (P1), and N* with R60 is limited only by the detector.

## 9. Exchangeability check

Bank = U (all 300 images). Scores of H against U and of the official good test against U. Per class: AUROC of official-good versus H scores (0.5 = no shift), two-sided Mann–Whitney p with Holm over 12 classes, medians, and the fraction of official good images with conformal p ≤ 0.05 against H. This replaces the leave-one-out construction of addendum E3; H and U are disjoint, so no image is scored against itself.

## 10. Secondary analyses

Stopping rule: v1 section 6 unchanged, computed on C20, δ ∈ {0.025, 0.05, 0.10}, main 0.05, compared with N* of R20 and of R60.

Budget split (from addendum E2, now pre-declared): per class and repetition, a random permutation of U from `SeedSequence([42, 1000 + class_index, r, 20261120])`; for B ∈ {80, 120, 160, 200, 300}, bank = first N, calibration = next B − N, N ∈ {1, 2, 4, 8, 16, 32, 64, 128} with B − N ≥ 20; rules: conformal rank and PAC rank (ε = 0.05, δ = 0.10); evaluated on the official test and on H. Hypothesis carried over from MVTec, descriptive: for B ≥ 120 the split with the most passing repetitions reserves at least 40 images for calibration in most classes.

## 11. E0 gate

As v1 section 8, with the bank restricted to U:

- E0a: Anomalib PatchCore, coreset 0.1, `num_neighbors = 9`, trained on U, official test. Operational floor: mean image AUROC ≥ 0.90. Published full-training-set PatchCore results on VisA are around 0.95; a smaller bank is expected to be slightly lower.
- E0b: own cache versus Anomalib, full bank U, `num_neighbors = 1`; |Δ mean AUROC| ≤ 0.005 and Spearman ρ ≥ 0.99 in every class.

Integrity checks as v1, plus: rows of H and of the official test never appear as columns of the distance tensor.

## 12. Compute and artifacts

Features and distances on a GPU (Colab). Per class, the distance tensor is at most 1106 × 784 × 300 float32 ≈ 1.0 GB; it is kept on the runtime's local disk and is not required on Drive after the trajectories, the exchangeability check, and the budget split are finished. Drive keeps `splits/`, `e0/`, `runs/parts/`, `runs/scores/`, `analysis/`, `report/`, and `environment.json`. Every artifact stores the v2 config hash.
