# Detector portability protocol — operating thresholds and decision rules

This document records the operating thresholds and decision rules
used in `transferability_audit/`. The thresholds were chosen at the
time of the original CV2024 audit (commit `b2c7b37`, before any of
the external benchmarks were hashed) and held fixed across all
subsequent additions.

## Detector parameters (FIXED across all benchmarks)

| Component | Setting |
| --- | --- |
| Hash algorithm | `imagehash.phash` + `imagehash.dhash` |
| Hash size | 8 (64-bit each, 128-bit joint) |
| Joint Hamming threshold | $\le 6$ |
| Strict-mode variant | joint $= 0$ (pHash-exact) |
| NCC pixel-confirmation | $256\times256$ grayscale, threshold $\ge 0.99$ |
| NCC stage applies to | joint $\le 6$ candidate pairs |

These were the operating thresholds for the CV2024 audit
(`src/audit/01_phash_dhash_audit.py`); the same thresholds were
reused without modification for ISIC 2019, HyperKvasir, and
Kvasir-SEG runs in this directory.

## Endpoints and denominators (PRE-REGISTERED)

| Endpoint | Numerator | Denominator |
| --- | --- | --- |
| CV2024 within-split rate (joint=0) | val rows with pHash-exact training match | KVASIR validation rows (n=11,581) |
| CV2024 within-split rate (NCC≥0.99) | val rows with both pHash-exact AND NCC≥0.99 training match | same n=11,581 |
| ISIC cross-source (joint≤6) | unordered cross-source pairs | n=25,331 indexed rows |
| ISIC cross-source (NCC≥0.99) | NCC-confirmed cross-source pairs | n=25,331 |
| HK × CV2024-X cross-bench | unordered cross-bench pairs | HyperKvasir n=10,662 |
| Kvasir-SEG × CV2024-X cross-bench | unordered cross-bench pairs | Kvasir-SEG n=1,000 |

## Statistical procedures (PRE-REGISTERED)

- Wilson 95% CI for binomial proportions (with explicit anti-conservative caveat for clustered data)
- Cluster-bootstrap (video-level resampling) for CV2024 within-split rate, $n_{\text{boot}}{=}20{,}000$, percentile method (chosen for conservativeness on lower bound)
- Two-proportion z-test reported in the v1 draft was DROPPED after adversarial review (anti-conservative under within-video clustering)

## Decisions made AFTER seeing data (DISCLOSED)

The following framings were revised after running the measurements:

1. The "three orders of magnitude" headline was downgraded to "$\sim 90\text{--}1{,}500\times$ depending on endpoint" after computing the actual range.
2. The "transferability evidence" framing was downgraded to "portability run" after the ISIC cross-source pairs were identified as metadata-gap re-attributions.
3. The HyperKvasir × Kvasir-Capsule cross-bench was relabeled "negative control across procedures" rather than "transferability evidence."
4. The "specificity at 627M comparisons" framing was relabeled as a "sanity check" after the random-collision null expectation was computed (≈ 1e-20 hits at this volume).

## Measurements NOT taken (decision: SKIP)

After multi-agent adversarial pre-mortem before deadline:

- Cassidy 2022 dedup-list intersection: SKIPPED (outcome distribution bimodal-bad, no time to spin either tail responsibly).
- DINOv2 semantic feature stage on ISIC + HyperKvasir: SKIPPED (highest backfire probability × longest runtime; cited as future work).
