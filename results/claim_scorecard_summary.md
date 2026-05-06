# Evidence-and-scope summary

| Axis | Recomputed value | Source |
| --- | --- | --- |
| External Kvasir-origin overlap | 38,592/38,592 KVASIR rows (100.0%); 60.7% exact on both hashes; NCC mean 0.998 | annotations + NCC summary |
| Fixed-list sensitivity | baseline 0.825, le6 0.612, Delta -0.213 | results/baseline/phase5_v5_baseline_n10.json, results/baseline/phase5_v5_le6_n10.json |
| Per-source decomposition | KVASIR -0.473, SEE-AI +0.013, KID -0.044, AIIMS +0.021 | baseline/le6 JSON per_source fields |
| Non-Ulcer sensitivity | exclude Ulcer -0.139; exclude Ulcer/Worms/Normal -0.121 | baseline/le6 JSON per_class fields |
| Public re-score boundary | trained-team Delta combined -0.062; public-official Spearman orig/le6/le6+internal = 0.566/0.355/0.390 | results/cv2024_m7_inference.json, results/cv2024_public_official_bridge.json |
