# Public-validation to official-test bridge

This analysis uses organizer-released official-test metric JSONs and
public-validation rescoring outputs. It does not evaluate new models.
Direct model evaluation is only possible when the separate
organizer-released class-separated CV2024 test archive is available.

- Common public/test teams: 25
- Official-test-only teams: Machine Minds, Pioneers
- Public-only teams: none

## Proxy quality against official-test combined score

| Variant | Group | n | Pearson | Spearman | Kendall tau-b | rank MAE | score MAE | bias | top-5 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig_public_val | all_common_25 | 25 | 0.704 | 0.566 | 0.453 | 4.40 | 0.342 | +0.326 | 3/5 |
| orig_public_val | trained_19 | 19 | 0.039 | 0.244 | 0.216 | 4.53 | 0.413 | +0.413 | 3/5 |
| orig_public_val | trained_no_stem_18 | 18 | 0.381 | 0.463 | 0.359 | 3.78 | 0.401 | +0.401 | 3/5 |
| le6_public_val | all_common_25 | 25 | 0.548 | 0.355 | 0.267 | 5.84 | 0.285 | +0.253 | 2/5 |
| le6_public_val | trained_19 | 19 | -0.081 | 0.068 | 0.076 | 5.58 | 0.351 | +0.351 | 2/5 |
| le6_public_val | trained_no_stem_18 | 18 | 0.225 | 0.257 | 0.203 | 4.67 | 0.335 | +0.335 | 3/5 |
| le6_plus_internal_public_val | all_common_25 | 25 | 0.556 | 0.390 | 0.307 | 5.60 | 0.295 | +0.263 | 2/5 |
| le6_plus_internal_public_val | trained_19 | 19 | -0.038 | 0.139 | 0.146 | 5.16 | 0.364 | +0.364 | 2/5 |
| le6_plus_internal_public_val | trained_no_stem_18 | 18 | 0.248 | 0.340 | 0.281 | 4.44 | 0.349 | +0.349 | 2/5 |

## Main readout

The bridge analysis does not show that le6 is a uniformly better proxy for the official-test ranking on trained teams (Spearman 0.068 vs original 0.244; rank MAE 5.58 vs original 4.53). It should be used as a scope-clarifying limitation rather than a positive proxy claim.

