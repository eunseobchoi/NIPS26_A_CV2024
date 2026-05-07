# Detector portability runs

This directory documents how the perceptual-hash + NCC component of
our CV2024 audit behaves on three external benchmarks and one
clustered-CV2024 re-resampling. All thresholds were fixed in the
original CV2024 audit (see `PROTOCOL.md`); none were tuned on the
external runs.

## Operating point (FIXED)

- `imagehash 4.3.2` pHash + dHash, `hash_size=8` (128-bit joint)
- Joint Hamming threshold $\le 6$; strict-mode variant joint $= 0$
- NCC pixel-confirmation: $256\times256$ grayscale, threshold $\ge 0.99$

## Runs

### A. CV2024 within-split cluster bootstrap (`run_cluster_bootstrap.py`)

Frame-level Wilson CI is anti-conservative because frames cluster by
Kvasir-Capsule source video. We resample 41 unique video prefixes
with replacement.

- 11,581 KVASIR validation frames, 1,381 flagged (point: 11.92%)
- Cluster percentile 95% CI: **[6.90%, 18.00%]**
- BCa 95% CI: [7.68%, 20.01%]
- Leave-cluster-out robustness:
  - Drop top-1 video → 9.79% (1082/11057)
  - Drop top-2 → 7.78% (809/10394)
  - Drop top-3 → 7.49% (741/9886)
- Top 2 videos contribute 41.4% of flagged rows
- File: `cv2024_cluster_bootstrap.json`

### B. ISIC 2019 within-benchmark cross-source (`run_audit.py`, `run_joint_le6.py`, `run_ncc.py`)

25,331 dermoscopy images, 4 declared sources from `lesion_id`
prefix (BCN 12,413 / HAM 10,015 / ISIC-archive 2,084 / MSK 819).

- Intra-source pHash-exact: BCN 78, HAM 19, ISIC-archive 3, MSK 1
  (each $\le 1\%$)
- Intra-source unordered joint $\le 6$ (full enumeration, no
  split-pair approximation): BCN 294, HAM 71, ISIC-archive 14, MSK 4
- Intra-source NCC $\ge 0.99$ pixel-confirmed: BCN 150, HAM 28,
  ISIC-archive 4, MSK 4 (total 186)
- Candidate cross-source pairs at joint $\le 6$: 3
  (1 HAM↔ISIC-archive at NCC=0.84 not pixel-confirmed;
  2 MSK↔ISIC-archive at NCC$\ge 0.999$ pixel-confirmed)
- Pixel-confirmed cross-source rate: **0.008%** (2/25,331). Both
  pairs trace to ISIC-archive entries with empty `lesion_id` that
  are MSK-derived images; best read as within-source MSK pairs
  with a metadata gap, not as site/instrument cross-source overlap.

For a recall-tuned audit of ISIC duplicates (multi-hash pipeline,
14,310 duplicates), see Cassidy et al. 2022.

### C. HyperKvasir × CV2024 cross-bench (`run_hyperkvasir.py`, `run_hk_nonkvasir.py`)

10,662 HyperKvasir labeled images cross-benched against all four
CV2024 source slices.

- HK × CV2024-KVASIR (38,592): 0 joint $\le 6$ pairs, 0 pHash-exact
- HK × CV2024-SEE-AI (14,285): 0 / 0
- HK × CV2024-KID (541): 0 / 0
- HK × CV2024-AIIMS (321): 0 / 0
- Intra-HyperKvasir: pHash-exact 335 extras (3.14%) across 254
  groups; intra unordered joint $\le 6$ pairs: 1923

Files: `hyperkvasir_audit.json`, `hk_nonkvasir_xbench.json`.

### D. Kvasir-SEG × CV2024 cross-bench (`run_kvasir_seg.py`)

1,000 Kvasir-SEG colonoscopy + polyp-mask images.

- Kvasir-SEG × all four CV2024 slices: 0 / 0 / 0 / 0 pairs at
  joint $\le 6$, 0 pHash-exact

File: `kvasir_seg_audit.json`.

## Cross-bench summary

Concatenating C and D against all four CV2024 source slices:
$11{,}662 \times 53{,}739 \approx 627$M ordered comparisons,
0 collisions in every combination. Under a uniform-random hash
null the expected hit count at joint $\le 6$ over 627M comparisons
is $\approx 1.0\times 10^{-20}$ (numerical zero); the all-zero
outcome is therefore a sanity check, not a power claim against
random-collision FP. See `random_collision_baseline.json`.

## Statistical contrast

At the matched NCC $\ge 0.99$ pixel-confirmed endpoint:

- CV2024-KVASIR within-split: $4.66\%$ (540/11,581)
- ISIC 2019 cross-source: $0.008\%$ (2/25,331)
- Rate-ratio range: 93× (cluster lower propagated to NCC-confirmed
  / ISIC NCC Wilson upper) to 1490× (point/point joint=0 frame
  level) — i.e., $\sim 90\text{--}1{,}500\times$ depending on
  endpoint.

The two-proportion z-test reported in earlier drafts was dropped
after adversarial review (anti-conservative under within-video
clustering).

## Per-class breakdown

CV2024 within-split rate by validation class (`cv2024_per_class_breakdown.json`):

| Class | n | k | rate |
| --- | --- | --- | --- |
| Normal | 10,302 | 1,311 | 12.73% |
| Bleeding | 134 | 20 | 14.93% |
| Angioectasia | 260 | 23 | 8.85% |
| Erythema | 48 | 4 | 8.33% |
| Foreign Body | 233 | 9 | 3.86% |
| Erosion | 152 | 5 | 3.29% |
| Ulcer | 257 | 7 | 2.72% |
| Lymphangiectasia | 178 | 2 | 1.12% |
| Polyp | 17 | 0 | 0% |

Driven by the Normal class (89% of validation rows).

## Skipped measurements (decision: SKIP)

- Cassidy 2022 dedup-list intersection: outcome distribution
  bimodal-bad, no time to spin either tail responsibly.
- DINOv2 semantic feature stage on ISIC + HyperKvasir: highest
  backfire probability × longest runtime; cited as future work.

## Reproducibility

```bash
# 1. ISIC 2019
aria2c -x 16 -s 16 \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv
unzip -q ISIC_2019_Training_Input.zip -d images/
python3 run_audit.py --metadata ISIC_2019_Training_Metadata.csv \
  --images images/ISIC_2019_Training_Input \
  --out results/isic2019_audit.json
python3 run_joint_le6.py
python3 run_ncc.py

# 2. HyperKvasir
aria2c -x 16 -s 16 --check-certificate=false \
  https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip
unzip -q hyper-kvasir-labeled-images.zip
python3 run_hyperkvasir.py --images labeled-images
python3 run_hk_nonkvasir.py

# 3. Kvasir-SEG
aria2c -x 8 -s 8 --check-certificate=false \
  https://datasets.simula.no/downloads/kvasir-seg.zip
unzip -q kvasir-seg.zip
python3 run_kvasir_seg.py

# 4. Cluster bootstrap (uses CV2024 KVASIR pHash CSV)
python3 run_cluster_bootstrap.py

# 5. Statistical contrast
python3 run_stats.py
```

Source SHA-256 hashes are in `source_zips.sha256`. We redistribute
metadata, hashes, and audit results only — no image bytes.
