from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base_dir = Path("rtc_diag_stage_base")
tt_dir = Path("rtc_diag_stage_tt_full_fixedprev")
stage_label_path = Path("rtc_stage_inputs_grasp_easy/raw/stage_labels.npy")

out_dir = Path("rtc_diag_stage_final_comparison")
fig_dir = out_dir / "figures"
metric_dir = out_dir / "metrics"

fig_dir.mkdir(parents=True, exist_ok=True)
metric_dir.mkdir(parents=True, exist_ok=True)

stage_labels = np.load(stage_label_path, allow_pickle=True)
obs_ids = [f"obs_{i:03d}" for i in range(len(stage_labels))]
stage_df = pd.DataFrame({
    "obs_id": obs_ids,
    "stage": stage_labels,
})

base = pd.read_csv(base_dir / "metrics" / "per_obs_metrics.csv")
tt = pd.read_csv(tt_dir / "metrics" / "per_obs_metrics.csv")

groups = []

x = base[base["method"] == "naive"].copy()
x["method"] = "Ordinary BC\nnaive"
groups.append(x)

x = base[base["method"] == "training_free_rtc"].copy()
x["method"] = "Ordinary BC\nruntime RTC"
groups.append(x)

x = tt[tt["method"] == "naive"].copy()
x["method"] = "TT-RTC ckpt\nnaive"
groups.append(x)

x = tt[tt["method"] == "training_free_rtc"].copy()
x["method"] = "TT-RTC ckpt\nconditioned"
groups.append(x)

combined = pd.concat(groups, ignore_index=True)
combined = combined.merge(stage_df, on="obs_id", how="left")

combined.to_csv(metric_dir / "stage_combined_per_obs_metrics.csv", index=False)

summary = combined.groupby(["stage", "method"]).mean(numeric_only=True).reset_index()
summary.to_csv(metric_dir / "stage_summary_metrics.csv", index=False)

print(summary)

metrics = [
    "mean_pairwise_distance",
    "endpoint_variance",
    "prefix_error",
    "seam_discontinuity",
    "jerk",
]

method_order = [
    "Ordinary BC\nnaive",
    "Ordinary BC\nruntime RTC",
    "TT-RTC ckpt\nnaive",
    "TT-RTC ckpt\nconditioned",
]

stage_order = ["early", "middle", "late"]

# Stage-wise bar plots.
for metric in metrics:
    for stage in stage_order:
        df = summary[summary["stage"] == stage]

        values = []
        labels = []

        for method in method_order:
            row = df[df["method"] == method]
            if len(row) == 0:
                continue
            labels.append(method)
            values.append(float(row[metric].iloc[0]))

        plt.figure(figsize=(9, 4))
        plt.bar(labels, values)
        plt.ylabel(metric)
        plt.title(f"{stage} stage: {metric}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{stage}_summary_{metric}.png", dpi=200)
        plt.close()

# One plot per metric with stages on x-axis and methods as curves.
for metric in metrics:
    plt.figure(figsize=(8, 4))

    for method in method_order:
        ys = []
        xs = []
        for stage in stage_order:
            row = summary[(summary["stage"] == stage) & (summary["method"] == method)]
            if len(row) == 0:
                continue
            xs.append(stage)
            ys.append(float(row[metric].iloc[0]))
        plt.plot(xs, ys, marker="o", label=method)

    plt.ylabel(metric)
    plt.title(f"Stage-wise trend: {metric}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"stage_trend_{metric}.png", dpi=200)
    plt.close()

print("Saved to:", out_dir)
