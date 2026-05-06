# Trivial Leakage Baseline

Diagnostic only: this uses training labels plus identity/hash/verified Kvasir video-prefix signals on the public validation pool.
It is not an official-test baseline or replacement leaderboard.

Source-prefix rule: Only Kvasir 16-hex filename prefixes are used; class-name tokens and unverified source filename stems are denied.

| Subset | n | mean AUC | bal. acc. | combined |
| --- | ---: | ---: | ---: | ---: |
| orig_public_val | 16132 | 0.6453 | 0.2866 | 0.4659 |
| le6 | 4551 | 0.5189 | 0.1208 | 0.3199 |
| le6_plus_internal | 4326 | 0.5126 | 0.1062 | 0.3094 |

## Coverage

| Rule | n |
| --- | ---: |
| same_nearest_kvasir_frame | 6464 |
| fallback_train_prior | 821 |
| same_source_prefix | 8333 |
| exact_phash_dhash_train_match | 340 |
| exact_filename_train_match | 174 |
