# Auditing the Capsule Vision 2024 Challenge — Code and Data

This repository accompanies the paper *"Auditing Capsule Vision 2024:
Within-Split Train-to-Validation Re-Exposure and a Kvasir-Channel
Sensitivity Diagnostic"*.

**Anonymous mirror (double-blind review):**
<https://anonymous.4open.science/r/NeurIPS2026ED-CV2024-Audit/>

This repository provides:

1. The **Kvasir-origin-removed sensitivity dataset** (CSV splits with
   per-file pHash/dHash/PDQ and pixel-NCC annotations), released as
   metadata only (no image bytes), under CC-BY 4.0.
2. **Source code** reproducing every analysis in the paper:
   perceptual-hash audit (pHash, dHash, PDQ), pixel NCC verification,
   internal CV2024 train/val leak check, video-prefix attribution,
   label-inheritance audit, Kvasir-origin-removed split generation,
   fixed-list retraining contrasts, direct evaluation on the
   organizer-released AIIMS test, and auxiliary split / test-time
   adaptation stress tests.
3. **Experiment results** as JSON files, from which every number cited
   in the paper can be recomputed.

**We do not redistribute any image bytes.** Reproducing the audit
requires downloading Kvasir-Capsule (Smedsrud et al. 2021) and CV2024
(Handa et al. 2024) from the original authors under their respective
licenses; see `DATA_CARD.md`.

---

## Quickstart (≤ 30 seconds, no GPU, no data download)

To verify that the released CSVs and result JSONs are internally
consistent and reproduce the paper's headline numbers, run the
consistency check:

```bash
bash scripts/run_smoke_test.sh
```

This performs 24 read-only checks across:

1. **File integrity** — every file listed in `checksums.txt` matches its
   recorded SHA-256.
2. **Croissant manifest consistency** — `croissant.json` distribution
   entries match `checksums.txt`.
3. **CSV row counts** — `le0/le2/le6/le6_strict` train and val and
   `le6_plus_internal` validation match `DATA_CARD.md`.
4. **Headline fixed-list retraining contrast** — recomputes
   $\Delta_{\texttt{le6}}{=}{-}0.213\pm 0.005$ from the released
   fixed-list result JSONs.
5. **100% KVASIR pHash claim** — recomputes from
   `artifacts/annotations/cv2024_KVASIR_phash_annotated.csv`.
6. **Evidence-and-scope summary** — recomputes the consolidated source,
   non-Ulcer, and public re-score rows used in the paper summary table.
7. **Official-test aggregate replay** — verifies that the released
   official-test JSONs' mean AUC, balanced accuracy, and combined metric
   match the aggregate definitions in the pinned CV2024 `gen_metrics_test.py`.
8. **Kvasir 7/25 video overlap** — recomputes from
   `data/official_splits/{split_0,split_1}.csv` (skipped with `WARN` if
   those CSVs are not present locally; see
   `artifacts/csvs/SPLIT_PROVENANCE.md` for SHA-256 and a 5-line
   reproduction).

Expected output: `PASS (24/24 checks)`. Failures print the offending
file and exit non-zero, so the script can be wired into CI.

### Files included vs. files that must be regenerated

To keep this repository small (~150 MB), the following large
intermediate files are **not** included; regenerate them by running
Stage 0 or Stage 3 if you want to re-derive the audit from raw datasets:

| File / folder | Size | Regenerate with | Used by |
|---|---|---|---|
| `artifacts/hashes/{hashes,pdq_hashes}_{cv2024,kvasir}.json` | ~105 MB | `bash scripts/00_run_full_audit.sh` (~25 min CPU) | Stage 0/1 generators only — not the consistency check |
| `artifacts/annotations/cv2024_{KVASIR,SEE-AI}_pdq_annotated.csv` | ~34 MB | `bash scripts/00_run_full_audit.sh` | Per-row PDQ rows; aggregate `pdq_audit.json` is included |
| `external/cv2024_repo/Results/submitted_excel_files/` | ~55 MB | clone upstream organizer repo | Optional Stage 3 `RUN_M7=1` re-score; the included `results/cv2024_*rescored*.json` and `cv2024_m7_inference.json` already contain every paper-cited number |

The consistency check (24 checks) reads only included files and passes
without any of the above being present.

## Full Quickstart (reproduce the audit from raw datasets)

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Point to your local copies of the raw datasets
export CV2024_ROOT=/path/to/cv2024/Dataset     # or the parent containing Dataset/
export KVASIR_ROOT=/path/to/Kvasir-Capsule     # or labelled_images/

# 3. Reproduce the audit (~3 min CPU for pHash; +~15 min for full NCC on 38,592 pairs)
bash scripts/00_run_full_audit.sh
bash scripts/01_run_dedup.sh

# 4. Reproduce canonical fixed-list retraining contrasts and released-CSV controls
#    (~multi-hour GPU job)
bash scripts/02_run_counterfactual.sh

# 5. Reproduce Kvasir split / video-prefix stress experiments (~3 h on A100)
bash scripts/03_run_video_split.sh

# 6. Optionally reproduce auxiliary TTA benchmark (~2 h on A100)
bash scripts/04_run_tta.sh

# 7. Regenerate paper figures
bash scripts/05_make_figures.sh
```

All stages write JSON results under `results/<stage>/`. The included
fixed-list control JSONs preserve the exact run provenance used in the
paper. Reruns from the published CSVs write to a separate directory, so
they do not overwrite the original results. Every number cited in the
paper can be recomputed either by the consistency check (24 checks) or
by `scripts/10_verify_claim_provenance.py` (69 numeric claims).

---

## Repository layout

```
artifact-root/
├── README.md                  # this file
├── LICENSE                    # CC-BY 4.0 for metadata + MIT for code
├── DATA_CARD.md               # Croissant-style dataset metadata
├── requirements.txt           # pinned Python dependencies
├── checksums.txt              # SHA-256 manifest for released files
│
├── src/
│   ├── audit/                 # hash/NCC/label/internal-leak audit
│   ├── dedup/                 # dedup CSV generation
│   ├── counterfactual/        # fixed-list retraining contrasts
│   ├── video_split/           # 4-backbone protocol-gap + split robustness
│   ├── tta/                   # auxiliary TTA benchmark (8 methods)
│   └── utils/                 # dataset, model, figure helpers
│
├── scripts/                   # shell entry points
│   ├── 00_run_full_audit.sh
│   ├── 01_run_dedup.sh
│   ├── 02_run_counterfactual.sh
│   ├── 03_run_video_split.sh
│   ├── 04_run_tta.sh
│   ├── 05_make_figures.sh
│   ├── 06_run_m7_rescore.sh          # optional, requires organizer prediction sheets
│   ├── 06_merge_acceptance_lift.sh   # optional, merges per-seed shard JSONs
│   ├── 07_merge_strengthening_results.sh # optional, merges same-source/domain / Comp-C/Comp-D shards
│   ├── 08_run_official_test_eval.sh      # optional, GPU AIIMS-test direct evaluation
│   ├── 08_verify_official_test_metrics.py # CPU replay of official-test aggregate definitions
│   ├── 09_run_acceptance_experiments.sh  # optional CPU strengthening analyses
│   ├── 09_verify_croissant_manifest.py   # Croissant/checksum consistency check
│   ├── 10_verify_claim_provenance.py     # claim-by-claim provenance check
│   ├── 10_run_convnext_acceptance.sh     # optional GPU ConvNeXt robustness
│   ├── 11_launch_le0_le2_n10_extension.sh # GPU le0/le2 seed extension
│   ├── 11_merge_le0_le2_n10_extension.py  # strict merger for that extension
│   └── 12_make_claim_scorecard.py          # CPU evidence-and-scope summary table
│
├── artifacts/                 # metadata-only release (CC-BY 4.0)
│   ├── csvs/                  # dedup train/val splits at le0/le2/le6 + le6_plus_internal
│   ├── hashes/                # 64-bit pHash/dHash + 256-bit PDQ hashes
│   ├── annotations/           # per-file annotations + exact internal train-val pair list
│   ├── ncc/                   # pixel-NCC annotations on 38,592 flagged KVASIR pairs
│   └── summaries/             # aggregate JSON summaries used in the paper
│
├── results/                   # experiment outputs (JSON, regeneratable)
│   ├── baseline/              # baseline + le6 n=10 paired seeds
│   ├── counterfactual/        # le0/le2/le6 + random10596 paired retraining
│   ├── counterfactual_n10/    # canonical n=10 control arms (Comp-A/B/D, compmatched)
│   ├── crossmodel/            # DINOv2-B/S, ResNet-50, and ConvNeXt-Tiny robustness
│   ├── acceptance_lift/       # n=10 completion + strict cleaned-val checks
│   ├── strengthening/         # same-source/domain / Comp-C / Comp-D aggregates
│   ├── official_test/         # direct released official AIIMS-test scope check
│   ├── le0_le2_extension/     # le0/le2 n=10 extension shards/summaries
│   ├── mechanism_probes/      # NCC dose-response, label-shuffle, same-source/domain probes
│   ├── loso/                  # source-held-out (KVASIR/SEE-AI/KID) stress-test JSONs
│   ├── multibackbone/         # 4-backbone frame-vs-video protocol gap
│   ├── split_robustness/      # 60/40 / 70/30 / 80/20 / 90/10 / LOVO
│   ├── tta/                   # auxiliary 8-method TTA benchmark
│   └── auxiliary/             # LSO, per-video, scaling, strong baselines
│
├── tests/                     # small stdlib unit tests for release helpers
│
└── configs/                   # run configs (paths, seeds, etc.)
```

---

## Which script produces which paper figure or table?

| Paper/artifact element                             | Script                                                                           | Produces                                                                   |
|----------------------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Main Table 1 — multi-hash audit                    | `src/audit/01_phash_dhash_audit.py` + `src/audit/02_pdq_audit.py`                | `artifacts/summaries/phash_audit.json`, `pdq_audit.json`                   |
| Main Figure 1 — public-pool audit evidence         | `src/audit/01_phash_dhash_audit.py` + `src/audit/03_ncc_verify.py`               | `figures/fig_hamming_hist.*`, `figures/fig_ncc.*`, `artifacts/ncc/cv2024_KVASIR_ncc_full.csv` |
| Appendix — video-prefix Kvasir reuse               | `src/audit/10_per_patient_leakage.py`                                            | video-prefix attribution JSON (`artifacts/summaries/per_patient_leakage.json`) |
| Main Table 2 — internal train→val leakage          | `src/audit/07_internal_leak.py` + `src/audit/09_cross_source_internal.py`        | `cv2024_internal_*.json`; exact KVASIR rows in `artifacts/annotations/cv2024_KVASIR_internal_train_val_phash_exact_pairs.csv` |
| Main Table 3 — `le6` class × source breakdown      | `src/dedup/01_generate_dedup_splits.py`                                          | `artifacts/csvs/cv2024_{training,validation}_dedup_le6.csv`                |
| Main Figure 2 and appendix fixed-list tables       | `src/counterfactual/train_fixed_list_counterfactual.py`                           | the fixed-list result JSONs; reruns from the published CSVs write to a separate directory |
| Appendix `le0`/`le2` n=10 extension                | `scripts/11_launch_le0_le2_n10_extension.sh` + `scripts/11_merge_le0_le2_n10_extension.py` | per-seed shards and `le0_le2_n10_extension_summary.{json,md}` under `results/le0_le2_extension/` |
| Composition-matched + Comp-A/B/C/D controls        | dedup control generators + `src/counterfactual/train_fixed_list_counterfactual.py` | fixed-list controls, Comp-C/Comp-D n=10 outputs in `results/strengthening/`, and rerun outputs |
| Strict cleaned-val and matched-arm completion      | fixed-list training wrapper + `scripts/merge_acceptance_lift.py`                  | `results/acceptance_lift/*`, `acceptance_lift_summary.{json,md}` |
| Same-source/domain re-exposure and source controls | fixed-list training wrapper + `scripts/merge_strengthening_results.py`            | `results/mechanism_probes/phase5_exp1_le6_kvfree_s1_n10.json`, `results/strengthening/*`, `strengthening_summary.{json,md}` |
| Per-class decomposition                            | `src/counterfactual/08_aggregate_r4.py` + `results/r6_holm_survived_sensitivity.json` | canonical n=10 per-class decomposition and Holm-survived sensitivity checks |
| Cross-model residual robustness                    | `src/counterfactual/03_cross_model.py` + `src/counterfactual/05_consolidate_crossmodel.py` | `results/crossmodel/*_n10.json`                                            |
| Evidence-and-scope table                           | `scripts/12_make_claim_scorecard.py`                                              | `results/claim_scorecard_summary.{json,md}` and table in paper             |
| Auxiliary Kvasir split-protocol probes             | `src/video_split/*`                                                              | `results/multibackbone/*`, `results/split_robustness/*`                    |
| Auxiliary TTA stress test                          | `src/tta/tta_benchmark_full.py` + `src/tta/03_filter_pass_rate.py`                | `results/tta/tta_bench_official.json`, `artifacts/summaries/filter_pass_summary.json` |
| pHash threshold robustness                         | `src/audit/04_phash_geometric_robustness.py`                                     | `artifacts/summaries/phash_geometric_robustness.json`                      |
| Label-inheritance audit                            | `src/audit/05_label_mapping_audit.py`                                            | `artifacts/summaries/label_mapping_audit.json` + `cv_to_kvasir.json`       |
| Challenge-level public-validation re-score         | `src/m7_rescore_subset.py` → `src/m7_robustness.py` → `src/m7_inference.py`       | `results/cv2024_rescored_{orig,le6,le6_plus_internal}.json`, `cv2024_robust_null.json`, `cv2024_rescored_robust.json`, `cv2024_m7_inference.json` |
| Direct official AIIMS-test scope check             | `src/counterfactual/04_official_test_eval.py` + `scripts/merge_official_test.py` + `scripts/08_verify_official_test_metrics.py` | `results/official_test/*`, `official_test_direct_eval_summary.{json,md}`, `official_metric_replay.json` |

---

## Reproducibility checks and traceability

We provide:

1. **Pinned environment** (`requirements.txt`, `requirements-lock.txt`) —
   CUDA 12.4, PyTorch 2.6, ImageHash 4.3, `pdqhash` 0.2.6, scipy 1.15,
   pandas 2.2; exact pins are used wherever bit-identical reproduction
   was needed.
2. **Per-seed training logs** under `results/`, so that every number
   (mean ± std, per-seed) is independently verifiable without rerunning
   training.
3. **SHA-256 checksums** in `checksums.txt`; `scripts/verify_checksums.sh`
   re-verifies them (suitable for inclusion in CI).
4. **Unit tests** in `tests/` (`python3 -m unittest discover -s tests`)
   covering path normalization and aggregation edge cases.

Files in `src/provenance/` are stored as regular files rather than
symlinks, so that zip extraction and checksum verification produce
identical bytes on every system, including those that do not preserve
symlink metadata. See `src/provenance/README.md` for which exact script
matches each result file.

Hardware note: the counterfactual and optional auxiliary split/TTA stages
require a GPU (we used A100 40 GB and A6000 48 GB); the audit stages
(pHash, dHash, PDQ, NCC) are CPU-only.

---

## Licenses

- **Code** (`src/`, `scripts/`): MIT.
- **Metadata files** (`artifacts/`): CC-BY 4.0.
- **Experiment results** (`results/`): CC-BY 4.0.

No image bytes are redistributed. The underlying image datasets
(Kvasir-Capsule, SEE-AI, KID, AIIMS CE24) are governed by their
respective original licenses; see `DATA_CARD.md`.

---

## Citation

(Anonymised for double-blind review.)

```bibtex
@inproceedings{anonymous2026capsuleaudit,
  title  = {Auditing Capsule Vision 2024: Within-Split Train-to-Validation Re-Exposure and a Kvasir-Channel Sensitivity Diagnostic},
  author = {Anonymous},
  booktitle = {NeurIPS 2026 Evaluations \& Datasets Track},
  year   = {2026}
}
```
