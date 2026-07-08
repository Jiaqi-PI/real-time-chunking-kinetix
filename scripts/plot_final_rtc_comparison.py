from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base_dir = Path("rtc_diag_raw_64")
tt_dir = Path("rtc_diag_raw_64_training_time")
out_dir = Path("rtc_diag_final_comparison")
fig_dir = out_dir / "figures"
metric_dir = out_dir / "metrics"

fig_dir.mkdir(parents=True, exist_ok=True)
metric_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------
# 1. Load metrics
# ----------------------------

base = pd.read_csv(base_dir / "metrics" / "per_obs_metrics.csv")
tt = pd.read_csv(tt_dir / "metrics" / "per_obs_metrics.csv")

# ordinary BC + naive
base_naive = base[base["method"] == "naive"].copy()
base_naive["method"] = "ordinary_naive"

# ordinary BC + training-free RTC
base_rtc = base[base["method"] == "training_free_rtc"].copy()
base_rtc["method"] = "training_free_rtc"

# training-time RTC checkpoint + naive sampling
tt_naive = tt[tt["method"] == "naive"].copy()
tt_naive["method"] = "training_time_rtc"

# optional: training-time checkpoint + runtime RTC
tt_runtime = tt[tt["method"] == "training_free_rtc"].copy()
tt_runtime["method"] = "training_time_plus_runtime_rtc"

combined = pd.concat(
    [base_naive, base_rtc, tt_naive, tt_runtime],
    ignore_index=True,
)

combined.to_csv(metric_dir / "combined_per_obs_metrics.csv", index=False)

summary = combined.groupby("method").mean(numeric_only=True).reset_index()
summary.to_csv(metric_dir / "combined_summary_metrics.csv", index=False)

print("Summary:")
print(summary)


# ----------------------------
# 2. Bar plots
# ----------------------------

metrics = [
    "mean_pairwise_distance",
    "endpoint_variance",
    "jerk",
    "seam_discontinuity",
    "prefix_error",
]

method_order = [
    "ordinary_naive",
    "training_free_rtc",
    "training_time_rtc",
    "training_time_plus_runtime_rtc",
]

for metric in metrics:
    plt.figure(figsize=(8, 4))
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
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / f"final_summary_{metric}.png", dpi=200)
    plt.close()


# ----------------------------
# 3. Paired plots for the main three methods
# ----------------------------

main_methods = [
    "ordinary_naive",
    "training_free_rtc",
    "training_time_rtc",
]

for metric in metrics:
    pivot = combined[combined["method"].isin(main_methods)].pivot(
        index="obs_id",
        columns="method",
        values=metric,
    )

    existing = [m for m in main_methods if m in pivot.columns]
    if len(existing) < 2:
        continue

    plt.figure(figsize=(7, 4))
    for obs_id, row in pivot.iterrows():
        plt.plot(
            existing,
            [row[m] for m in existing],
            marker="o",
            alpha=0.6,
        )

    plt.ylabel(metric)
    plt.title(f"Paired comparison: {metric}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / f"final_paired_{metric}.png", dpi=200)
    plt.close()


# ----------------------------
# 4. Shared-scale heatmaps for obs_000
# ----------------------------

obs_id = "obs_000"

heatmaps = {
    "ordinary_naive": base_dir / "metrics" / f"{obs_id}_naive_pairwise_distance.npy",
    "training_free_rtc": base_dir / "metrics" / f"{obs_id}_training_free_rtc_pairwise_distance.npy",
    "training_time_rtc": tt_dir / "metrics" / f"{obs_id}_naive_pairwise_distance.npy",
}

loaded = {}
for name, path in heatmaps.items():
    if path.exists():
        loaded[name] = np.load(path)

if loaded:
    vmax = max(float(x.max()) for x in loaded.values())

    for name, dist in loaded.items():
        plt.figure()
        plt.imshow(dist, aspect="auto", vmin=0.0, vmax=vmax)
        plt.colorbar(label="pairwise distance")
        plt.xlabel("sample id")
        plt.ylabel("sample id")
        plt.title(f"{obs_id} - {name} - shared scale")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{obs_id}_{name}_shared_heatmap.png", dpi=200)
        plt.close()

print("Saved final comparison to:", out_dir)
