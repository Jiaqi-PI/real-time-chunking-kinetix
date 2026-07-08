from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base_dir = Path("rtc_diag_raw_64")
tt_dir = Path("rtc_diag_raw_64_training_time_full_fixedprev")

out_dir = Path("rtc_diag_final_comparison_correct_fixedprev")
fig_dir = out_dir / "figures"
metric_dir = out_dir / "metrics"

fig_dir.mkdir(parents=True, exist_ok=True)
metric_dir.mkdir(parents=True, exist_ok=True)


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
combined.to_csv(metric_dir / "combined_per_obs_metrics.csv", index=False)

summary = combined.groupby("method").mean(numeric_only=True).reset_index()
summary.to_csv(metric_dir / "combined_summary_metrics.csv", index=False)

print(summary)


metrics = [
    "mean_pairwise_distance",
    "endpoint_variance",
    "jerk",
    "seam_discontinuity",
    "prefix_error",
]

method_order = [
    "Ordinary BC\nnaive",
    "Ordinary BC\nruntime RTC",
    "TT-RTC ckpt\nnaive",
    "TT-RTC ckpt\nconditioned",
]


for metric in metrics:
    plt.figure(figsize=(9, 4))
    values = []
    labels = []

    for method in method_order:
        row = summary[summary["method"] == method]
        if len(row) == 0:
            continue
        labels.append(method)
        values.append(float(row[metric].iloc[0]))

    plt.bar(labels, values)
    plt.ylabel(metric)
    plt.title(f"Final comparison: {metric}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"summary_{metric}.png", dpi=200)
    plt.close()


for metric in metrics:
    pivot = combined.pivot(index="obs_id", columns="method", values=metric)

    existing = [m for m in method_order if m in pivot.columns]

    plt.figure(figsize=(10, 4))
    for obs_id, row in pivot.iterrows():
        plt.plot(existing, [row[m] for m in existing], marker="o", alpha=0.65)

    plt.ylabel(metric)
    plt.title(f"Paired comparison: {metric}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"paired_{metric}.png", dpi=200)
    plt.close()


obs_id = "obs_000"

heatmaps = {
    "Ordinary BC naive": base_dir / "metrics" / f"{obs_id}_naive_pairwise_distance.npy",
    "Ordinary BC runtime RTC": base_dir / "metrics" / f"{obs_id}_training_free_rtc_pairwise_distance.npy",
    "TT-RTC ckpt naive": tt_dir / "metrics" / f"{obs_id}_naive_pairwise_distance.npy",
    "TT-RTC ckpt conditioned": tt_dir / "metrics" / f"{obs_id}_training_free_rtc_pairwise_distance.npy",
}

loaded = {}
for name, path in heatmaps.items():
    if path.exists():
        loaded[name] = np.load(path)

vmax = max(float(x.max()) for x in loaded.values())

for name, dist in loaded.items():
    safe = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    plt.figure()
    plt.imshow(dist, aspect="auto", vmin=0.0, vmax=vmax)
    plt.colorbar(label="pairwise distance")
    plt.xlabel("sample id")
    plt.ylabel("sample id")
    plt.title(f"{obs_id} - {name} - shared scale")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{obs_id}_{safe}_shared_heatmap.png", dpi=200)
    plt.close()

print("Saved to:", out_dir)
