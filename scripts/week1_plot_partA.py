from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["k"]))
    return rows


def arr(rows, key):
    return np.asarray(
        [float(row[key]) for row in rows],
        dtype=np.float64,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(args.summary)

    ks = np.asarray([int(row["k"]) for row in rows], dtype=int)

    wrong = arr(rows, "wrong_prefix_error_mean")
    wrong_low = arr(rows, "wrong_prefix_error_mean_ci_low")
    wrong_high = arr(rows, "wrong_prefix_error_mean_ci_high")

    aligned = arr(rows, "aligned_slot_error_mean")
    aligned_low = arr(rows, "aligned_slot_error_mean_ci_low")
    aligned_high = arr(rows, "aligned_slot_error_mean_ci_high")

    gain = arr(rows, "alignment_gain_mean")
    gain_low = arr(rows, "alignment_gain_mean_ci_low")
    gain_high = arr(rows, "alignment_gain_mean_ci_high")

    # A1
    plt.figure(figsize=(6.4, 4.8))
    plt.errorbar(
        ks,
        wrong,
        yerr=[wrong - wrong_low, wrong_high - wrong],
        marker="o",
        capsize=4,
        label="Wrong-prefix: C_t[0]",
    )
    plt.errorbar(
        ks,
        aligned,
        yerr=[aligned - aligned_low, aligned_high - aligned],
        marker="o",
        capsize=4,
        label="First-align: C_t[k]",
    )
    plt.xticks(ks)
    plt.xlabel("Temporal offset k")
    plt.ylabel("Normalized action RMSE")
    plt.title("A1: Cross-time Action Alignment")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    a1_png = args.out_dir / "A1_cross_time_error_vs_k.png"
    a1_pdf = args.out_dir / "A1_cross_time_error_vs_k.pdf"
    plt.savefig(a1_png, dpi=220)
    plt.savefig(a1_pdf)
    plt.close()

    # A2
    plt.figure(figsize=(6.4, 4.8))
    plt.errorbar(
        ks,
        gain,
        yerr=[gain - gain_low, gain_high - gain],
        marker="o",
        capsize=4,
    )
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(ks)
    plt.xlabel("Temporal offset k")
    plt.ylabel("Alignment gain (Wrong-prefix - First-align)")
    plt.title("A2: Temporal Alignment Gain")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    a2_png = args.out_dir / "A2_alignment_gain_vs_k.png"
    a2_pdf = args.out_dir / "A2_alignment_gain_vs_k.pdf"
    plt.savefig(a2_png, dpi=220)
    plt.savefig(a2_pdf)
    plt.close()

    print("WROTE =", a1_png)
    print("WROTE =", a1_pdf)
    print("WROTE =", a2_png)
    print("WROTE =", a2_pdf)
    print()
    print("===== VALUES USED =====")
    for i, k in enumerate(ks):
        print(
            f"k={k}: "
            f"wrong={wrong[i]:.6f} [{wrong_low[i]:.6f}, {wrong_high[i]:.6f}] "
            f"aligned={aligned[i]:.6f} [{aligned_low[i]:.6f}, {aligned_high[i]:.6f}] "
            f"gain={gain[i]:.6f} [{gain_low[i]:.6f}, {gain_high[i]:.6f}]"
        )
    print()
    print("WEEK1_PARTA_PLOTS_OK")


if __name__ == "__main__":
    main()
