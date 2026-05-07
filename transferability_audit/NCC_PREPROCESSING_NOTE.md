# NCC preprocessing note — `cv2024_cluster_bootstrap_ncc.json`

This supplementary file reports NCC-confirmed cluster bootstrap on
the 1,381 within-CV2024 train→val pHash-exact pairs. It is NOT the
same number cited in the paper.

**Paper number (load-bearing):** 540/11,581 = 4.66% NCC-confirmed.
This is from the original CV2024 audit
(`results/cv2024_internal_ncc_verify.json`, available in the master
audit repo) using the project's primary NCC pipeline. This is the
number used for all paper claims.

**Supplementary number in this directory:** 387 NCC>=0.99 (run by
`run_cluster_bootstrap_ncc.py`, 256x256 grayscale resize, threshold
NCC>=0.99). The 387 number reflects a different preprocessing
pipeline (image resize + grayscale at audit time) than the paper's
load-bearing 540.

**Why we keep 387:** the 256x256-gray pipeline matches the protocol
we use for the external (ISIC, HyperKvasir, Kvasir-SEG) NCC
verifications, so reporting it here lets a reviewer reproduce a
fully apples-to-apples cluster-bootstrap CI under the same
preprocessing as the external probes.

**Why we cite 540 in the paper:** 540 is the canonical CV2024 audit
number, cross-referenced with PDQ corroboration, table-of-contents
NCC summary, and our internal CV2024 NCC pipeline; replacing it
mid-paper with 387 would be inconsistent with the rest of the audit
chapter.

**Net effect on the matched-NCC claim:** at 387/11,581 = 3.34% (this
file's pipeline) the gap vs. ISIC NCC 0.008% is 418x point/point;
at the canonical 540/11,581 = 4.66% the gap is 583x. Both fall
inside the paper's reported "$\sim 150{-}600\times$" matched-NCC
range.

We declare the discrepancy explicitly here rather than collapse it.
