# Transferability probes

Three external benchmarks, holding all detector parameters fixed
(`imagehash 4.3.2` pHash + dHash, `hash_size=8`, joint $\le 6$, NCC
$\ge 0.99$ pixel-confirmation), to test the framework outside of
Capsule Vision 2024.

## Probe 1 — ISIC 2019 multi-source dermoscopy (`run_audit.py`, `run_joint_le6.py`, `run_ncc.py`)

25,331 dermoscopy images, 4 declared sources via `lesion_id` prefix
(BCN 12,413 / HAM 10,015 / ISIC-archive 2,084 / MSK 819).

- Intra-source pHash-exact: BCN 0.63%, HAM 0.19%, ISIC-archive 0.14%, MSK 0.12%
- Cross-source joint $\le 6$: 3 pairs (HAM↔ISIC-archive 1, MSK↔ISIC-archive 2)
- Cross-source NCC $\ge 0.99$ pixel-confirmed: 2 pairs (both MSK↔ISIC-archive)
- Pixel-confirmed cross-source rate: 0.008% (2/25,331)

Both confirmed pairs trace to ISIC-archive entries with empty
`lesion_id` metadata that are MSK-derived images; not a release
defect.

## Probe 2 — HyperKvasir × CV2024-KVASIR cross-bench (`run_hyperkvasir.py`)

10,662 HyperKvasir labeled images cross-benchmarked against the
38,592 CV2024-KVASIR rows (Kvasir-Capsule slice). Same lab (Simula),
same brand, different procedures (colon/upper-GI still images vs.
small-bowel video capsule frames).

- Intra-HyperKvasir pHash-exact: 335 extra rows / 254 groups (3.14%)
- HyperKvasir × CV2024-KVASIR joint $\le 6$: **0 pairs**
- HyperKvasir × CV2024-KVASIR pHash-exact: **0 HyperKvasir rows**

Framework correctly registers the two benchmarks as image-disjoint
despite the shared lab and brand.

## Probe 3 — Statistical contrast (`run_stats.py`)

Wilson 95% CIs and two-proportion z-tests against CV2024:

| Benchmark | Rate | Wilson 95% CI |
| --- | --- | --- |
| CV2024 KVASIR within-split (joint=0) | 11.92% (1,381/11,581) | [11.35%, 12.53%] |
| ISIC 2019 cross-source (NCC≥0.99) | 0.008% (2/25,331) | [0.002%, 0.029%] |
| HyperKvasir × Kvasir-Capsule (joint≤6) | 0% (0/10,662) | [0%, 0.036%] |

Two-proportion $z = 55.9$, $p < 10^{-300}$. Bootstrap rate ratio
(CV2024 / ISIC pixel-confirmed) median $\approx 1{,}500\times$ with
97.5% lower bound $\approx 590\times$.

## Reproduce

```bash
# 1. ISIC 2019 — fetch from challenge mirror
aria2c -x 16 -s 16 \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv
unzip -q ISIC_2019_Training_Input.zip -d images/

python3 run_audit.py \
  --metadata ISIC_2019_Training_Metadata.csv \
  --images images/ISIC_2019_Training_Input \
  --out results/isic2019_audit.json

python3 run_joint_le6.py
python3 run_ncc.py

# 2. HyperKvasir — fetch from Simula
aria2c -x 16 -s 16 --check-certificate=false \
  https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip
unzip -q hyper-kvasir-labeled-images.zip
python3 run_hyperkvasir.py --images labeled-images

# 3. Stats
python3 run_stats.py
```

Total runtime on a CPU box: ISIC pipeline ~5 min, HyperKvasir audit
~3 min (download excluded).

Source zip SHA-256 hashes are in `source_zips.sha256`. We
redistribute metadata, hashes, and audit results only — no image
bytes.
