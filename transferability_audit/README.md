# Transferability probe — ISIC 2019

Demonstrates portability of the perceptual-hash component
(`imagehash 4.3.2` pHash + dHash, hash_size=8) of the CV2024 audit
framework to a second multi-source medical benchmark in a different
modality (dermoscopy).

## Reproduce

```bash
# 1. Fetch ISIC 2019 training set from the official challenge mirror
aria2c -x 16 -s 16 \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv

# 2. Verify zip integrity (SHA-256)
sha256sum -c ISIC_2019_Training_Input.zip.sha256

# 3. Extract + run audit
unzip -q ISIC_2019_Training_Input.zip -d images/
python3 run_audit.py \
  --metadata ISIC_2019_Training_Metadata.csv \
  --images images/ISIC_2019_Training_Input \
  --out results/isic2019_audit.json
```

Runtime: ~2 minutes on a CPU (25,331 images).

## Result summary

- Source distribution (4 declared, from `lesion_id` prefix):
  BCN 12,413 / HAM 10,015 / ISIC-archive 2,084 / MSK 819
- Intra-source pHash-exact rates: BCN 0.63%, HAM 0.19%,
  ISIC-archive 0.14%, MSK 0.12%
- Cross-source pHash-exact collisions: 2 (0.008%),
  both MSK ↔ ISIC-archive (consistent with archive entries
  whose `lesion_id` is missing in the released metadata)

Compare against CV2024 within-split re-exposure: 11.9%
(1,381/11,581 KVASIR validation rows; equivalently 8.6% of
all validation rows). At the same operational detector setting
(`hash_size=8`, joint=0), the rates differ by three orders of
magnitude.

## Scope

This is a transferability probe of the **detector component only**
(perceptual hash family, joint=0 strict variant). It does **not**:

- run NCC or DINOv2/ResNet feature checks
- extend the sensitivity-split protocol to ISIC 2019
- claim release-validity for ISIC 2019 (the original challenge
  papers contain their own source-overlap discussion)

We redistribute only metadata and hashes from this probe — no
image bytes.
