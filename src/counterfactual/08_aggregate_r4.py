"""Aggregate fixed-list sensitivity experiments into paper-ready tables. V2 handles
both the old schema (cv2024_pooled.json with history[-1].bal_acc) and the
new v2 schema (phase5_v2_*.json with history[-1].orig_val.bal_acc)."""
import os
import json
import statistics

ROOT = os.environ.get("CAPSULE_ROOT", ".") + "/results"

def load(fn):
    try:
        with open(fn) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {fn}") from exc

def get_orig_val(runs):
    vals = []
    for r in runs:
        if r.get("history"):
            last = r["history"][-1]
            # Try v2 schema first (history[-1].orig_val.bal_acc)
            if "orig_val" in last and isinstance(last["orig_val"], dict):
                v = last["orig_val"].get("bal_acc")
            else:
                # v1 schema: history[-1].bal_acc (for cv2024_pooled)
                v = last.get("bal_acc")
            if v is not None: vals.append(v)
    return vals

def get_per_class(runs):
    out = {}
    for r in runs:
        if r.get("history"):
            last = r["history"][-1]
            # v2: orig_val.per_class
            if "orig_val" in last and "per_class" in last["orig_val"]:
                pc = last["orig_val"]["per_class"]
                for cls, d in pc.items():
                    out.setdefault(cls, []).append(d.get("recall", 0.0))
            # v1 with per_class (cv2024_pooled had best_metrics.per_class, not history)
            elif "best_metrics" in r and "per_class" in r["best_metrics"]:
                pc = r["best_metrics"]["per_class"]
                for cls, d in pc.items():
                    out.setdefault(cls, []).append(d.get("acc", 0.0))
    return out

def summarize(vals, name):
    if not vals: return f"{name}: NO DATA"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0
    return f"{name}: {m:.4f} +- {s:.4f} (n={len(vals)})"

def combine(all_vals, keys):
    out = []
    for k in keys:
        out.extend(all_vals.get(k, []))
    return out


def main():
    groups = {
        "baseline":       ["cv2024_pooled.json"],
        "baseline_23":    ["phase5_v2_baseline_seeds23.json"],
        "random_01":      ["phase5_random10596_seeds01.json"],
        "random_23":      ["phase5_v2_random10596_seeds23.json"],
        "le0_01":         ["phase5_counterfactual_le0_seeds01.json"],
        "le0_23":         ["phase5_v2_le0_seeds23.json"],
        "le2_01":         ["phase5_counterfactual_le2_seeds01.json"],
        "le2_23":         ["phase5_v2_le2_seeds23.json"],
        "le6_strict":         ["phase5_v2_le6_perclass_seeds0123.json"],
        "le6_v1_01":      ["phase5_counterfactual_le6_seeds01.json"],
        "le6_v1_23":      ["phase5_counterfactual_le6_seeds23.json"],
        "compmatched":    ["phase5_v2_compmatched_s0_seeds0123.json"],
    }

    all_vals = {}
    all_pc = {}
    for name, files in groups.items():
        vals = []
        pcs = {}
        for f in files:
            d = load(f"{ROOT}/{f}")
            if d:
                vals.extend(get_orig_val(d.get("runs", [])))
                p = get_per_class(d.get("runs", []))
                for k, v in p.items():
                    pcs.setdefault(k, []).extend(v)
        all_vals[name] = vals
        all_pc[name] = pcs
        print(summarize(vals, name))

    print()
    print("=== Combined ===")
    bl = combine(all_vals, ["baseline", "baseline_23"])
    rn = combine(all_vals, ["random_01", "random_23"])
    le0 = combine(all_vals, ["le0_01", "le0_23"])
    le2 = combine(all_vals, ["le2_01", "le2_23"])
    # For le6, prefer v2 (with per_class); if incomplete, fall back to v1+v2.
    le6_strict = all_vals["le6_strict"]
    le6_v1 = combine(all_vals, ["le6_v1_01", "le6_v1_23"])
    le6 = le6_strict if len(le6_strict) >= 4 else le6_v1
    cm = all_vals["compmatched"]

    for name, v in [("baseline", bl), ("random", rn), ("le0", le0), ("le2", le2),
                    ("le6 v2 (per_class)", le6_strict), ("le6 v1 (merged)", le6_v1),
                    ("le6 (headline)", le6), ("comp-matched", cm)]:
        print(summarize(v, name))

    if bl and rn and le6 and cm:
        print()
        print("=== Δ analysis (orig_val bal_acc) ===")
        mb = statistics.mean(bl)
        mr = statistics.mean(rn)
        ml = statistics.mean(le6)
        mc = statistics.mean(cm)
        print(f"Baseline:      {mb:.4f}")
        print(f"Random:        {mr:.4f}  (Δ_random = {mr - mb:+.4f})")
        print(f"Comp-matched:  {mc:.4f}  (Δ_comp  = {mc - mb:+.4f})")
        print(f"le6:           {ml:.4f}  (Δ_le6   = {ml - mb:+.4f})")
        print()
        print(f"Residual (le6 - random) = {(ml - mb) - (mr - mb):+.4f}  [size-controlled]")
        print(f"Pure leakage-removal (compmatched - le6, class held fixed) = {mc - ml:+.4f}")
        print(f"Class-composition shift (random - compmatched) = {mr - mc:+.4f}")
        print(f"Size-only residual (baseline - random) = {mb - mr:+.4f}")

    print()
    print("=== Per-class decomposition (le6 vs baseline, if available) ===")
    pc_le6 = all_pc["le6_strict"]
    pc_bl = all_pc.get("baseline", {})
    if pc_le6 and pc_bl:
        for cls in sorted(pc_le6.keys()):
            vl = pc_le6.get(cls, [])
            vb = pc_bl.get(cls, [])
            if vl and vb:
                mlv = statistics.mean(vl)
                mbv = statistics.mean(vb)
                print(f"  {cls:20s}: base={mbv:.3f}  le6={mlv:.3f}  Δ={mlv-mbv:+.3f}  n_val={len(vl)}")
    else:
        if not pc_le6:
            print("  le6 per_class not yet saved (need seed 0+ complete)")
        if not pc_bl:
            print("  baseline per_class not available (cv2024_pooled has best_metrics.per_class; would need to recompute or use different schema)")


if __name__ == "__main__":
    main()
