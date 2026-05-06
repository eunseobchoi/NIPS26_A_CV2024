# Vendored upstream snapshot — Capsule Vision 2024 Challenge

This directory contains a partial mirror of the Capsule Vision 2024 Challenge
organizer repository at upstream commit:

    de38787e8267f763a5f59695c67bb6d28ece70bf

Original repository:
    https://github.com/misahub2023/Capsule-Vision-2024-Challenge

## Purpose

The mirrored files (`Results/gen_metrics_*.py`, the per-team validation/test
prediction `*.xlsx` and metric `*.json` files, and the upstream `README.md`)
are required as inputs to the public-validation re-scoring pipeline in
`scripts/06_run_m7_rescore.sh` and the AIIMS-test direct-evaluation replay
in `scripts/08_verify_official_test_metrics.py`.

## License status

The upstream repository at the mirrored commit ships **no LICENSE file**.
The challenge website and the released `*.xlsx` / `*.json` per-team
prediction sheets are public artifacts of the Capsule Vision 2024
Challenge; we redistribute the minimum subset required for the public
re-scoring claim and treat them as research-fair-use academic
reproducibility material.

If upstream subsequently publishes an explicit license, that license
governs these files. Reviewers who want to verify against the latest
upstream state should `git checkout` the SHA above directly from the
upstream repository.

## What is vendored

- `README.md` — upstream challenge description (verbatim).
- `COMMIT.txt` — upstream commit SHA used for this snapshot.
- `Results/gen_metrics_report_val_train.py` — official validation-set
  metric script.
- `Results/gen_metrics_test.py` — official test-set metric script.
- `Results/metrics_reports/metrics_reports_val/*.json` — per-team
  validation metric JSONs that the M7 re-score consumes.
- `Results/metrics_reports/metrics_reports_test/*.json` — per-team
  official-test metric JSONs that `scripts/08_verify_official_test_metrics.py`
  replays.
- `Results/submitted_excel_files/validation/*.xlsx` — per-team
  validation prediction sheets that the M7 re-score consumes when
  `RUN_M7=1` is set.

## What is NOT vendored

Image bytes, training-set Excel sheets that contain raw frame paths,
and any upstream artifact not required by the released re-scoring or
direct-evaluation pipelines.
