#!/usr/bin/env python3
"""
Statistical comparison of cross-benchmark rate gaps for the
transferability appendix.

Reports:
  - Wilson 95% CI for each binomial proportion
  - Two-proportion z-test
  - Bootstrap 95% CI on the rate ratio (CV2024 / ISIC)

Note: rates are not strictly i.i.d. (frames within videos / patients are
dependent), so Wilson and z-test under-estimate variance somewhat.
We report them as conservative lower bounds on the magnitude gap.
"""
import json
import math

import numpy as np
from scipy import stats


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval — better than normal-approx for small p."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.isf(alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(center - half, 0.0), min(center + half, 1.0))


def two_proportion_z(k1, n1, k2, n2):
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se < 1e-12:
        return float("inf"), 0.0
    z = (p1 - p2) / se
    p_two_sided = 2 * stats.norm.sf(abs(z))
    return float(z), float(p_two_sided)


def bootstrap_ratio(k1, n1, k2, n2, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    # Bernoulli resampling; safe even though the underlying events are
    # dependent because we report the resulting CI as conservative.
    p1_boot = rng.binomial(n1, k1 / n1, size=n_boot) / n1
    p2_boot = rng.binomial(n2, k2 / n2, size=n_boot) / n2
    eps = 1e-9
    ratio = (p1_boot + eps) / (p2_boot + eps)
    return {
        "median_ratio": float(np.median(ratio)),
        "ci_2.5": float(np.quantile(ratio, 0.025)),
        "ci_97.5": float(np.quantile(ratio, 0.975)),
        "log10_median": float(np.log10(np.median(ratio))),
        "log10_ci_2.5": float(np.log10(max(np.quantile(ratio, 0.025), eps))),
        "log10_ci_97.5": float(np.log10(np.quantile(ratio, 0.975))),
    }


def report(label1, k1, n1, label2, k2, n2):
    p1 = k1 / n1
    p2 = k2 / n2
    ci1 = wilson_ci(k1, n1)
    ci2 = wilson_ci(k2, n2)
    z, pv = two_proportion_z(k1, n1, k2, n2)
    ratio_boot = bootstrap_ratio(k1, n1, k2, n2)
    print(f"\n=== {label1} vs {label2} ===")
    print(f"  {label1}: {k1}/{n1} = {p1*100:.4f}%  "
          f"Wilson 95% CI = [{ci1[0]*100:.4f}%, {ci1[1]*100:.4f}%]")
    print(f"  {label2}: {k2}/{n2} = {p2*100:.4f}%  "
          f"Wilson 95% CI = [{ci2[0]*100:.4f}%, {ci2[1]*100:.4f}%]")
    print(f"  Two-proportion z = {z:.2f}, p = {pv:.2e}")
    print(f"  Rate ratio ({label1}/{label2}): "
          f"median {ratio_boot['median_ratio']:.1f} "
          f"(95% CI [{ratio_boot['ci_2.5']:.1f}, "
          f"{ratio_boot['ci_97.5']:.1f}])")
    print(f"  log10 ratio: median {ratio_boot['log10_median']:.2f} "
          f"(95% CI [{ratio_boot['log10_ci_2.5']:.2f}, "
          f"{ratio_boot['log10_ci_97.5']:.2f}])")
    return {
        "label_a": label1, "k_a": k1, "n_a": n1, "p_a": p1, "ci_a": ci1,
        "label_b": label2, "k_b": k2, "n_b": n2, "p_b": p2, "ci_b": ci2,
        "z": z, "p_two_sided": pv, "ratio_bootstrap": ratio_boot,
    }


def main():
    out = {}

    # CV2024 within-split (KVASIR validation): 1,381 / 11,581 pHash-exact
    # ISIC 2019 cross-source joint <= 6: 3 / 25,331
    out["cv2024_kvasir_val_vs_isic_cross_source_le6"] = report(
        "CV2024 KVASIR within-split (joint=0)", 1381, 11581,
        "ISIC 2019 cross-source (joint<=6)", 3, 25331,
    )

    # CV2024 all-validation (1,381 / 16,132) vs ISIC 2019 cross-source
    out["cv2024_all_val_vs_isic_cross_source_le6"] = report(
        "CV2024 all-validation (joint=0)", 1381, 16132,
        "ISIC 2019 cross-source (joint<=6)", 3, 25331,
    )

    # CV2024 KVASIR within-split vs ISIC 2019 cross-source NCC>=0.99 confirmed (2/25,331)
    out["cv2024_kvasir_val_vs_isic_ncc_confirmed"] = report(
        "CV2024 KVASIR within-split (joint=0)", 1381, 11581,
        "ISIC 2019 cross-source NCC>=0.99 (pixel-confirmed)", 2, 25331,
    )

    with open("results/transferability_stats.json", "w") as f:
        # Convert numpy types
        def _conv(o):
            if hasattr(o, "tolist"):
                return o.tolist()
            return o
        json.dump(out, f, indent=2, default=_conv)
    print(f"\nWrote results/transferability_stats.json")


if __name__ == "__main__":
    main()
