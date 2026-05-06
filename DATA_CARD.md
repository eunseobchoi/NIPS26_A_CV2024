# Dataset Card — CV2024 Public-Validation Audit (Kvasir-Origin-Removed Splits)

*Follows the Croissant 1.1 / RAI schema. Image bytes are not
redistributed; this card describes derived metadata only (file paths,
labels, perceptual hashes, NCC values).*

---

## Overview

| Field                  | Value                                                                                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **Name**               | CV2024 public-validation audit dataset (canonical split: `le6`; sensitivity split: `le6_plus_internal`)                                  |
| **Contents**           | Splits, per-image perceptual hashes (pHash, dHash, PDQ), pixel NCC values, audit summaries, and reproduction code. The primary public-pool fixed-list retraining contrast is $\Delta_{\texttt{le6}}{=}{-}0.213$ at $n{=}10$ paired seeds; AIIMS-test direct-evaluation means are baseline $0.4991$, random10596\_s0 $0.5069$, and \texttt{le6} $0.5110$ combined metric over $10$ seeds. |
| **License**            | CC-BY 4.0 (metadata only)                                                                                                                  |
| **Format**             | CSV (UTF-8) + JSON                                                                                                                         |
| **Hosting**            | Anonymous mirror for review at <https://anonymous.4open.science/r/NeurIPS2026ED-CV2024-Audit/>; on acceptance, mirrored to Zenodo with DOI (OSF as secondary mirror). |
| **Original datasets**  | Kvasir-Capsule (Smedsrud et al. 2021), SEE-AI, KID, AIIMS CE24 (aggregated by Capsule Vision 2024 Challenge, Handa et al. 2024).           |
| **Not redistributed**  | Image bytes. Users must download the originals under their licenses.                                                                       |

---

## Intended use

- Reproduce the audit claims in the paper (leakage rates, fixed-list
  retraining $\Delta$, protocol-gap numbers) by joining our CSVs to the
  original image files.
- Train and evaluate on the `le6` or `le6_plus_internal` sensitivity
  splits alongside the original CV2024 `training/` and `validation/`
  CSVs to report Kvasir-origin-removed public-pool accuracy beside the
  original public-pool accuracy.
- Independently replicate the perceptual-hash and pixel-NCC analyses:
  we release the 64-bit pHash and dHash and the 256-bit PDQ
  bit-strings for every file and for every nearest-Kvasir match.

**Not intended for**: clinical decision-making; replacement of the
CV2024 official AIIMS-test leaderboard. We include a direct evaluation
on the organizer-released AIIMS test for our DINOv2 arms only; this is
a scope check, not a benchmark replacement.

## Recommended use by research question

| Question | Recommended split | Scope statement to report |
|----------|-------------------|----------------------------|
| Kvasir-origin-removed CV2024 public-pool evaluation | canonical `le6` train/val CSVs | Public-pool, fixed-list evaluation with CV2024-KVASIR rows removed under pHash+dHash ≤6. Not an official-test or source-held-out claim. |
| Sensitivity to stricter same-source removal | `le6_plus_internal` validation CSV | Same as `le6`, additionally excluding measured same-source train→val duplicates. A sensitivity check, not the primary split. |
| Cross-source low-hash-collision sensitivity | `le6_strict` CSVs | Four-row cross-source sensitivity check (rows enumerated below). Not the recommended primary split. |
| Re-scoring CV2024 challenge submissions | `results/cv2024_rescored_*.json`, `results/cv2024_m7_inference.json`, `results/cv2024_public_official_bridge.json` | Re-score public-validation prediction sheets from CV2024 challenge participants. Do not reinterpret as an official Capsule Vision ranking. |
| AIIMS-test direct evaluation for our DINOv2 arms | `results/official_test/` | AIIMS-only hidden-test direct-evaluation result for these trained arms, not a benchmark replacement. |

---

## Split contents

| Split                                           | Files   | Description                                                                                                            |
|-------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------|
| `cv2024_training_dedup_le0.csv`                 | 21,241 | original CV2024 training minus 23,444 pHash+dHash-exact matches to Kvasir-Capsule labeled frames                       |
| `cv2024_training_dedup_le2.csv`                 | 10,825 | strict near-duplicate removal (both pHash and dHash Hamming ≤ 2)                                                       |
| `cv2024_training_dedup_le6.csv` (**canonical**) | 10,596 | full KVASIR-source removal (both pHash and dHash Hamming ≤ 6); the SEE-AI/KID/AIIMS-only training pool                               |
| `cv2024_training_dedup_le6_strict.csv`              | 10,592 | `le6` minus 4 enumerated cross-source pHash+dHash ≤6 rows (SEE-AI/Erosion `image17118.jpg`, SEE-AI/Polyp `image00664.jpg`, KID/Normal `normaleso220.jpg`, AIIMS/Worms `worm1_2555.jpg`; DINOv2 cosine 0.4–0.8 — incidental matches, not content duplicates). One additional SEE-AI/Erosion low-hash collision (`image07362.jpg`) remains in both `le6` and `le6_strict` and is exposed in the annotation CSVs. Included for reproducibility. Paired contrast against `le6` (n=10 paired): \|Δ\| ≤0.022 per-source, all non-significant at α=0.05 (KVASIR 0.004, SEE-AI 0.002, KID 0.022, AIIMS 0.000). |
| `cv2024_validation_dedup_le0.csv`               | 9,054 | matching validation set at `le0` (CV2024 original val 16,132 minus 7,078 pHash-exact KVASIR-source duplicates)                                                                                       |
| `cv2024_validation_dedup_le2.csv`               | 4,655 | matching validation set at `le2` (CV2024 original val 16,132 minus 11,477 pHash$\wedge$dHash$\leq 2$ KVASIR-source matches)                                                                                       |
| `cv2024_validation_dedup_le6.csv` (canonical)   | 4,551 | matching validation set at `le6`                                                                                       |
| `cv2024_validation_le6_plus_internal.csv`       | 4,326 | canonical `le6` val, additionally excluding 171 SEE-AI filename-collision pairs and intra-source pHash-exact duplicates|
| `cv2024_training_random10596_s{0,42}.csv`       | 10,596 | size-matched random draws from the full original CV2024 pool (sample-size control)                                        |
| `cv2024_training_le6_kvfree_s1.csv`             | 10,880 | training list for the same-source/domain re-exposure probe; an identical, path-normalized copy is mirrored under `results/cv2024_training_le6_kvfree_s1.csv` for cross-reference |

**Schema (training/validation dedup CSVs in `artifacts/csvs/`)** — matches
the actual released files (verify with `head -1` on any CSV):

```
image_path, Dataset, Angioectasia, Bleeding, Erosion, Erythema,
Foreign Body, Lymphangiectasia, Normal, Polyp, Ulcer, Worms
```

- `image_path` (str): relative path under CV2024 root, e.g.
  `training\\Normal\\KVASIR\\<video_id>_<frame>.jpg`.
- `Dataset` (str): one of `{KVASIR, SEE-AI, KID, AIIMS}`.
- Remaining 10 columns: one-hot class indicators (exactly one is 1).

Per-file leakage metadata (pHash/dHash/PDQ/NCC/nearest_kvasir_file) lives in
`artifacts/annotations/` and `artifacts/ncc/` — see the next two schema
blocks below.

The exact KVASIR validation-to-training pHash-exact pair list is released as
`artifacts/annotations/cv2024_KVASIR_internal_train_val_phash_exact_pairs.csv`
($1{,}381$ validation rows plus a header). Here "pHash-exact" means a
CV2024 validation row whose 64-bit pHash exactly equals at least one
CV2024 training-row pHash; the CSV lists the first deterministic training
match, shared pHash, both dHashes, same-video-prefix flag, and the number
of training rows sharing that pHash.

**Hash annotations** (`artifacts/annotations/*.csv`):

```
source, partition, cv_dataset, cv_split, filename, path,
phash (hex, 64-bit), dhash (hex, 64-bit),
min_phash_dist_to_kvasir, min_dhash_dist_to_kvasir, nearest_kvasir_file
```

**NCC annotations** (`artifacts/ncc/cv2024_KVASIR_ncc_full.csv`) — one row
per flagged pair, all 38,592 pairs:

```
cv_file, cv_path, kvasir_file, kvasir_path,
phash_dist, dhash_dist, ncc (float in [-1, 1])
```

---

## Collection methodology

1. **Hash extraction.** For every file in CV2024 (`training` +
   `validation`, 53,739 images) and every labeled Kvasir-Capsule frame
   (47,238 images), we compute 64-bit pHash and 64-bit dHash via
   ImageHash 4.3.1 at `hash_size=8`, and 256-bit PDQ via `pdqhash`
   0.2.6. Full hash dumps (`artifacts/hashes/{hashes,pdq_hashes}_{cv2024,kvasir}.json`,
   ~105 MB) are **not included in this release** to keep the dataset small;
   regenerate with `bash scripts/00_run_full_audit.sh` (~25 min CPU after
   `$CV2024_ROOT` and `$KVASIR_ROOT` are set).

2. **Nearest-Kvasir annotation.** For each CV2024 file, we brute-force
   search its nearest Kvasir-Capsule labeled-frame hash (minimum Hamming
   distance per hash family). Per-row pHash annotations
   (`cv2024_*_phash_annotated.csv`) are included and read by the consistency check;
   the larger per-row PDQ annotations for KVASIR (25 MB) and SEE-AI (9 MB)
   are regenerated by Stage 0 alongside the hash dumps. The compressed
   summary (`artifacts/summaries/pdq_audit.json`) is included and is the
   source for every paper-cited PDQ number.

3. **Pixel-level verification.** For every flagged KVASIR-source pair
   (all 38,592), we compute Normalized Cross-Correlation (NCC) on
   grayscale-converted, resized-to-256×256 images. Stored in
   `artifacts/ncc/cv2024_KVASIR_ncc_full.csv`.

4. **Label inheritance.** We map Kvasir-Capsule's 14-class vocabulary
   to CV2024's 10-class schema using the official organizer mapping
   (`artifacts/summaries/cv_to_kvasir.json`) and verify one-hot
   consistency across the 38,584 overlapping entries.

5. **Internal audit.** Within CV2024 itself, we check for
   train→validation near-duplicates by the same three-hash pipeline,
   attribute every pHash-exact KVASIR pair to its Kvasir-Capsule
   labeled video, and record per-source, per-video, and video-prefix leakage
   statistics.

6. **Dedup split generation.** For each threshold $t \in \{0, 2, 6\}$,
   we remove every CV2024 file whose pHash and dHash distances to a
   Kvasir-Capsule labeled frame are both $\leq t$. The canonical
   `le6_plus_internal` additionally removes within-source exact
   duplicates.

---

## Composition of the canonical `le6` split

| Class            | SEE-AI train | KID train | AIIMS train | Total train | SEE-AI val | KID val | AIIMS val | Total val |
|------------------|-------------:|----------:|------------:|------------:|-----------:|--------:|----------:|----------:|
| Angioectasia     |          530 |        18 |           0 |         548 |        228 |       9 |         0 |       237 |
| Bleeding         |          519 |         3 |           0 |         522 |        223 |       2 |         0 |       225 |
| Erosion          |        2,340 |         0 |           0 |       2,340 |      1,003 |       0 |         0 |     1,003 |
| Erythema         |          580 |         0 |           0 |         580 |        249 |       0 |         0 |       249 |
| Foreign Body     |          249 |         0 |           0 |         249 |        107 |       0 |         0 |       107 |
| Lymphangiectasia |          376 |         6 |           0 |         382 |        162 |       3 |         0 |       165 |
| Normal           |        4,312 |       315 |           0 |       4,627 |      1,849 |     136 |         0 |     1,985 |
| Polyp            |        1,090 |        34 |           0 |       1,124 |        468 |      15 |         0 |       483 |
| Ulcer            |            0 |         0 |          66 |          66 |          0 |       0 |        29 |        29 |
| Worms            |            0 |         0 |         158 |         158 |          0 |       0 |        68 |        68 |
| **Total**        |    **9,996** |   **376** |     **224** |  **10,596** |  **4,289** | **165** |    **97** | **4,551** |

Note: `le6` removes all KVASIR-source examples. The remaining pool is
source-skewed (SEE-AI dominates most classes; Ulcer and Worms are
AIIMS-exclusive). This is not a source-held-out evaluation — SEE-AI,
KID, and AIIMS each appear on both the training and the validation
side. Leave-one-source-out evaluation is out of scope for this dataset.

---

## Responsible AI metadata

- **Sensitive information.** The underlying corpora are de-identified
  medical images and should be treated as sensitive health data even
  without direct identifiers. This metadata release adds no patient
  identifiers. Kvasir-Capsule filenames expose video-prefix-level
  grouping prefixes; patient-level disjointness is unavailable from the
  public metadata.
- **Use cases we flag as potentially harmful**: using `le6` to claim
  cross-source or leave-one-source-out generalization (it is neither).
- **Social impact.** Correcting the CV2024 public validation score
  interpretability reduces the risk that downstream medical-AI papers
  stack inflated numbers.
- **Known biases.**
  - Source skew: `le6` is dominated by SEE-AI (94 % of training, 94 % of
    validation); Ulcer and Worms classes exist only in AIIMS.
  - The pHash and dHash Hamming thresholds (≤ 6) miss most rotations
    ≥ 5° and center-crops ≥ 10 % (see
    `artifacts/summaries/phash_geometric_robustness.json`); the reported
    near-duplicate flag rate is conservative for crop and large-rotation variants. PDQ at a
    negative-control-calibrated threshold (≤ 50) corroborates at 99.4 %.
- **Limitations in the fixed-list retraining contrast.** Removing `le6` from training
  simultaneously drops (i) Kvasir-source re-exposure, (ii) domain-matched
  KVASIR imagery, and (iii) source/class priors. An exploratory
  matched-arm decomposition (Comp-A Ulcer source swap $n{=}10$, Comp-B
  Ulcer doubled $n{=}10$) accounts for $\sim 59\%$ of
  $\Delta_{\texttt{le6}}$ as a fixed-split residual KVASIR-content
  re-exposure contrast
  (fixed-list residual $-0.126$ on shared seeds; paired-$\Delta$
  $\Delta_{\text{size}}{=}{-}0.046$, $\Delta_{\text{class}}{=}{-}0.041$),
  with Comp-B TOST \emph{EQUIVALENT} inside a documented
  $\pm 0.010$ sensitivity bound before the extension seeds after an $n{=}4$ pilot;
  Comp-C shows that AIIMS-only Ulcer doubling does not recover the
  Comp-B gain.
  A same-source/domain re-exposure probe
  (Exp.~1, $n{=}10$ paired seeds) recovers $+0.091 \pm 0.003$~SE
  by adding $284$ non-overlapping Kvasir-Capsule frames to `le6`,
  consistent with public-validation inflation reflecting same-source
  familiarity in addition to direct frame re-exposure.  Cross-model
  (DINOv2-L, DINOv2-B, DINOv2-S, ResNet-50, and ConvNeXt-Tiny, all
  $n{=}10$) residual is
  $-0.167$/$-0.156$/$-0.148$/$-0.156$/$-0.208$.
- **Per-source decomposition.** The $-0.213$ pooled drop is highly
  asymmetric: KVASIR subset $-0.473$, SEE-AI $+0.013$, KID $-0.044$,
  AIIMS $+0.021$ ($n{=}10$ paired). The magnitude therefore reflects
  in-distribution KVASIR-content scoring, not generic cross-source
	  generalization (for which see LOSO in the paper appendix).

---

## Provenance and access

| Source | Role in this dataset | Public access / license evidence | Consent / de-identification status | Bytes redistributed here? |
|--------|------------------------|----------------------------------|------------------------------------|---------------------------|
| CV2024 public training + validation | Original file paths, labels, and public split audited here | Figshare Version 3, DOI `10.6084/m9.figshare.26403469.v3`, CC BY 4.0 | Governed by CV2024 release terms; this dataset adds no patient identifiers | No |
| CV2024 official AIIMS test | Direct scope check for our trained DINOv2 arms only | Figshare Version 4, DOI `10.6084/m9.figshare.27200664.v4`, CC BY 4.0 | Hidden during challenge, later organizer-released; used only through released labels/paths | No |
| Kvasir-Capsule labeled corpus | Named public source used for hash/NCC matching | OSF DOI `10.17605/OSF.IO/DV2AG`; dataset page at `https://datasets.simula.no/kvasir-capsule/` | Public medical-image corpus; patient IDs unavailable in the metadata used here | No |
| SEE-AI, KID, AIIMS-derived public CV2024 files | Non-KVASIR controls and remaining `le6` public-pool rows | Accessed through the CV2024 public release | Sub-source consent/licensing is not independently re-verified beyond the CV2024 release; users must follow original dataset and challenge terms | No |

- **CV2024 public training + validation corpus**: downloadable from the
  Capsule Vision 2024 Challenge organizers (Handa et al. 2024); our
  CSVs reference file paths as they appear in the official distribution
  (`training/<class>/<source>/<file>.jpg` and
  `validation/<class>/<source>/<file>.jpg`). Official Figshare article:
  "Training and Validation Dataset of Capsule Vision 2024 Challenge",
  DOI `10.6084/m9.figshare.26403469.v3`, CC BY 4.0.
- **CV2024 official AIIMS test corpus**: organizer-released,
  class-separated test archive; official Figshare article "Testing Dataset
  of Capsule Vision 2024 Challenge", DOI `10.6084/m9.figshare.27200664.v4`,
  CC BY 4.0. We use it only for the direct scope check in Stage 7, not for
  model selection or leaderboard replacement.
- **Kvasir-Capsule labeled corpus**: downloadable from
  https://osf.io/dv2ag/ (authors: Smedsrud et al. 2021; DOI
  10.17605/OSF.IO/DV2AG; mirror at
  https://datasets.simula.no/kvasir-capsule/); our CSVs
  reference file paths as they appear in the official
  `labelled_images/<class>/<file>.jpg` layout.
- **AIIMS official test set**: hidden during the challenge and later
  organizer-released; used only for the DINOv2 scope check in
  `results/official_test/`, not for participant re-ranking.
- **SEE-AI / KID / AIIMS-derived public CV2024 files**: accessed only
  through the CV2024 public release. We do not redistribute image bytes
  and make no independent relicensing claim for those sources; users
  must follow the original dataset and challenge terms.

A user wishing to reproduce the audit must download the two public
corpora and set `CV2024_ROOT` and `KVASIR_ROOT` as environment variables
before running the scripts.
