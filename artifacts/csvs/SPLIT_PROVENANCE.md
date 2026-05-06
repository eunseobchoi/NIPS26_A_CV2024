# Provenance for Kvasir-Capsule official 2-fold split CSVs

The two CSV files used in this paper for any reference to "Kvasir-Capsule
official split_0 / split_1" come directly from the Kvasir-Capsule OSF
deposit released alongside Smedsrud et al. (2021, *Scientific Data*),
and are pinned by SHA-256 to enable independent verification of the
7/25 video-overlap finding reported in the appendix
(`app:kvasir_split_overlap`).

## Source

- **OSF repository**: <https://osf.io/dv2ag/>
- **Persistent DOI**: 10.17605/OSF.IO/DV2AG
- **Original paper**: Smedsrud, P. H., Thambawita, V., Hicks, S. A., et al.
  "Kvasir-Capsule, a video capsule endoscopy dataset." *Scientific Data*
  8(1), 142 (2021). <https://www.nature.com/articles/s41597-021-00920-z>

## Mirrored files in this dataset

| Local path                                  | SHA-256                                                            | Frame rows (excluding header) | Unique videos |
|---------------------------------------------|--------------------------------------------------------------------|----------------------|---------------|
| `data/official_splits/split_0.csv`          | `fc7ed45cb061a378403c1b31375345462c90408960ee555727f7c00f8c3c0b3d` | 23,061               | 25            |
| `data/official_splits/split_1.csv`          | `5757b5ac99bdba4642edc2fde7eb2e1dbb2e0392b591b6866ec837720508134a` | 24,100               | 25            |

Union across both folds: 43 unique videos. The 7-video frame-level
overlap is the difference: 25 + 25 − 43 = 7.

## Reproducibility

```python
import pandas as pd
SPLIT_DIR = "data/official_splits"
s0 = set(
    pd.read_csv(f"{SPLIT_DIR}/split_0.csv")["filename"].str.split("_").str[0]
)
s1 = set(
    pd.read_csv(f"{SPLIT_DIR}/split_1.csv")["filename"].str.split("_").str[0]
)
shared = sorted(s0 & s1)
assert len(shared) == 7, f"expected 7, got {len(shared)}"
```

Expected `shared`:

```
['64440803f87b4843', '7a47e8eacea04e64', '7ad22d50ebaf4596',
 '8885668afb844852', '8ebf0e483cac48d6', 'ad91cf7ca91440aa',
 'bca26705313a4644']
```

## Note on the contradiction with paper text

The Kvasir-Capsule release paper recommends that "splits should be
completely different, probably even at the level of patients."
Direct inspection of the released CSVs shows the 2-fold split is
*frame-disjoint* but not *video-disjoint*. We do not interpret this
as a labeling error — the OSF metadata does not promise zero overlap
beyond the recommendation text — but we record it because downstream
evaluators using these CSVs as a video-disjoint split must account for
the 7-video overlap.
