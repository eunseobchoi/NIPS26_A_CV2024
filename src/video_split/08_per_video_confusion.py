"""Per-video confusion matrix analysis for Kvasir split_1.

Uses the saved per_video_kvasir_s1.json artifact to recompute.
Actually we need per-frame preds + labels per video to build confusion
matrices. Re-run inference directly — but that requires the model.

Alternative: the per_video JSON already has per_class_acc per video.
We can at least show:
(i) all 25 videos accuracy (not just bimodal ends)
(ii) per-class accuracy within each video (how many pathology frames
    are correctly classified inside Normal-dominant videos)
"""
import os
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))
PATH = ROOT / "results/per_video_kvasir_s1.json"
OUT_TEX = ROOT / "paper/tmlr/table_per_video.tex"


def main():
    with open(PATH) as f:
        rows = json.load(f)
    print(f"Total videos: {len(rows)}")

    # Overall stats
    accs = [r["acc"] for r in rows]
    print(f"All 25 video accuracies (sorted):")
    for r in sorted(rows, key=lambda x: -x["acc"]):
        cls_acc = r.get("per_class_acc", {})
        # Sub-class accuracy: what fraction of NON-dominant frames are correct?
        labels = r["classes_present"]
        total = sum(labels.values())
        dominant = r["dominant_class"]
        non_dom_acc = []
        for cls, n in labels.items():
            if cls != dominant:
                a = cls_acc.get(cls)
                if a is not None and n > 0:
                    non_dom_acc.append((cls, a, n))
        non_dom_total = sum(n for _, _, n in non_dom_acc)
        if non_dom_total > 0:
            non_dom_acc_avg = sum(a * n for _, a, n in non_dom_acc) / non_dom_total
        else:
            non_dom_acc_avg = None
        print(f"  {r['video_id']:<20} n={r['n_frames']:>5} "
              f"acc={r['acc']:.3f} dominant={dominant[:22]:<22} "
              f"non-dominant_acc={non_dom_acc_avg if non_dom_acc_avg is not None else '---'}")

    # Bucket counts
    high = [r for r in rows if r["acc"] >= 0.8]
    low = [r for r in rows if r["acc"] <= 0.2]
    mid = [r for r in rows if 0.2 < r["acc"] < 0.8]
    print(f"\nBuckets:")
    print(f"  High (acc ≥ 0.8):  {len(high)}")
    print(f"  Middle (0.2 < acc < 0.8): {len(mid)}")
    print(f"  Low (acc ≤ 0.2):    {len(low)}")

    # What do middle-bucket videos look like?
    print(f"\nMiddle bucket details:")
    for r in sorted(mid, key=lambda x: -x["acc"]):
        print(f"  {r['video_id']:<20} acc={r['acc']:.3f} dom={r['dominant_class']}")

    # Write LaTeX
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{Per-video accuracy on Kvasir split\_1.} "
        r"LoRA fold-0 checkpoint (final epoch).  Dominant class = class with most frames in that video.  "
        r"All 25 videos shown, sorted by accuracy.  The ``Non-dominant-frame accuracy'' column is the "
        r"weighted average accuracy across frames in that video whose label is not the dominant class — "
        r"i.e., whether the model can actually classify pathology inside a mostly-Normal video.}",
        r"\label{tab:per_video}",
        r"\small",
        r"\begin{tabular}{lrrlr}",
        r"\toprule",
        r"Video ID prefix & $n$ frames & Total acc & Dominant class & Non-dom acc \\",
        r"\midrule",
    ]
    for r in sorted(rows, key=lambda x: -x["acc"]):
        labels = r["classes_present"]
        dominant = r["dominant_class"]
        cls_acc = r.get("per_class_acc", {})
        non_dom_acc = []
        for cls, n in labels.items():
            if cls != dominant:
                a = cls_acc.get(cls)
                if a is not None:
                    non_dom_acc.append((cls, a, n))
        non_dom_total = sum(n for _, _, n in non_dom_acc)
        if non_dom_total > 0:
            non_dom_acc_avg = sum(a * n for _, a, n in non_dom_acc) / non_dom_total
            nd_str = f"{non_dom_acc_avg:.3f}"
        else:
            nd_str = "---"
        dom_escaped = dominant.replace("_", r"\_")
        vid_short = r["video_id"][:14]
        lines.append(f"  {vid_short} & {r['n_frames']} & {r['acc']:.3f} & "
                     f"{dom_escaped} & {nd_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    OUT_TEX.write_text("\n".join(lines))
    print(f"\nSaved {OUT_TEX}")


if __name__ == "__main__":
    main()
