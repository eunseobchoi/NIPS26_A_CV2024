"""Named constants for CV2024 split-generation invariants.

These values are intentionally defensive assertions, not tunable
hyperparameters.  If one changes, the generated CSV provenance and paper
numbers must be re-audited together.
"""

LE6_TRAIN_TOTAL = 10_596
KVASIR_FLAGGED_TOTAL = 38_592

COMP_SOURCE_ULCER_COUNT = 66
COMP_DOUBLED_ULCER_COUNT = 2 * COMP_SOURCE_ULCER_COUNT

COMP_MATCHED_KVASIR_NORMAL_TOTAL = 4_627
COMP_DISPLACED_KVASIR_NORMAL_TOTAL = (
    COMP_MATCHED_KVASIR_NORMAL_TOTAL - COMP_SOURCE_ULCER_COUNT
)

COMP_NORMAL_DROP_SEED = 42
COMP_EXTRA_KVASIR_ULCER_SEED = 42
