# Exploratory addendum (post hoc)

Written on 20 November 2026, after the confirmatory run (config hash `ec70992336b4c4a4`, E0 PASS, 600 trajectories) had been analysed. The analyses below were chosen after seeing those results. They are exploratory. They do not change N*, AULC, the stopping rule, the primary Wilcoxon tests, or anything else in `preregistration.md`, which stays frozen. Every parameter below is fixed in `commissioning/exploratory.py` before E2–E5 are computed on the real caches. Code: `python -m commissioning explore`. Outputs: `exploratory/`.

## Why

13 of 15 classes are right-censored for N*. Three observations from the confirmatory report motivate the addendum:

1. Under exchangeability of good scores, the threshold max(C) with |C| = 20 has conditional FPR ~ Beta(1, 20). With 12–60 good test images and a false-reject budget of 0–3, even a detector with TPR = 1 is expected to meet the FPR budget in fewer than 16 of 20 repetitions in every class (E1, analytic).
2. For textures the oracle FPR at TPR = 0.95 is 0 at every N, including N = 1, while the realized FPR at max(C) is well above 1/21 for carpet, leather and wood. This points to a difference between train/good and test/good scores rather than to the memory bank.
3. In practice, commissioning has a fixed budget of good parts. The confirmatory protocol fixes |C| = 20 and grows only the bank, so it cannot say how that budget should be split.

## Fixed parameters

- Target conditional FPR: epsilon = 0.05. Allowed exceedance probability: delta = 0.10.
- Threshold rules on m calibration scores, reject when the score is strictly above the k-th smallest:
  - conformal: k = ceil((m + 1)(1 − alpha)), alpha = 0.05 (the confirmatory rule; at m = 20 this is the maximum);
  - PAC: smallest k with P(Beta(m − k + 1, k) > epsilon) ≤ delta; if no k ≤ m qualifies, k = m and the rule is flagged infeasible.
- Practical criterion per repetition: as in the confirmatory protocol (TPR ≥ 0.95 and false rejects ≤ floor(0.05 · n_good) on the official test set); 16 of 20 repetitions.

## E1 Feasibility ceiling (analytic, no data)

For each class and calibration size m ∈ {20, 30, 40, 45, 60, 80, 100, 150, 200} (m < |train/good|) and each rule: P(rep meets the FPR budget) = BetaBinomial-CDF(budget; n_good, m − k + 1, k), its expectation over 20 repetitions, and P(at least 16 of 20) under an independence approximation. The repetitions share one test set, so the last quantity is labelled as an approximation. A separate curve gives P(conditional FPR > epsilon) against m for the maximum, conformal and PAC rules.

## E2 Fixed budget of good parts: bank versus calibration

Per class and repetition r = 0..19, a random permutation of all train/good images from `SeedSequence([42, class_index, r, 20261120])`, which is independent of the confirmatory splits. For a budget B ∈ {40, 60, 80, 120, 160, 200} with B ≤ |train/good|, the first N images of the permutation form the bank and the next B − N images form the calibration set, for N ∈ {1, 2, 4, 8, 16, 32, 64, 128} with B − N ≥ 20. Scores come from the existing distance tensor `cache/dist/{class}.npy`, with no new features. Both rules are evaluated on the official test set. Reported: passing repetitions, repetitions meeting TPR, repetitions meeting FPR, median TPR, mean FPR, and the best N per (class, B, rule), ranked by passing repetitions, then median TPR, then smaller N. Only random banks are used. A k-center bank would make the calibration set the non-selected images and break their exchangeability with new parts.

## E3 Exchangeability of train/good and test/good

The full training set is the bank. Train/good images are scored leave-one-out, through the +inf diagonal of the distance tensor. Test/good images are scored against all training images. Per class: the AUROC of test/good versus train/good scores (0.5 means no shift), a two-sided Mann–Whitney p-value with Holm correction over 15 classes, the medians, and the share of test/good images whose conformal p-value against train/good is ≤ 0.05, with a binomial tail p-value that is descriptive only. The defect AUROC with the full bank is given for context.

## E4 Good test images rejected repeatedly

From `runs/scores/*.npz`: for each good test image, the share of the 40 confirmatory trajectories (20 repetitions × 2 strategies) in which its score is above max(C), at N = |P| and at N = 32. Images rejected in at least half of the trajectories are shown in the model view (resize 256, center crop 224).

## E5 Censoring decomposition

From `analysis/metrics.parquet`: per class and strategy at N = |P|, the repetitions meeting TPR, meeting FPR, and meeting both. The limiting factor is the one below 16 of 20 (tpr, fpr, both, none). Also reported: the N with the most passing repetitions, and the E1 ideal expectation for comparison.

## How the paper uses this

The confirmatory results are reported first and unchanged. E1–E5 appear in a separate section titled as exploratory. Any recommendation drawn from E2 (for example, how many good parts to reserve for the threshold) is stated as a hypothesis for a new preregistered study, not as a validated rule.

The report figures for TPR, FPR and oracle FPR were redrawn after the run so that the full pool, whose size differs by class, appears as one separate point instead of class-specific x positions. This changes the display only.
