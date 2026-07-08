from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

input_dir = Path("rtc_diag_raw_64")
df = pd.read_csv(input_dir / "metrics" / "per_obs_metrics.csv")
fig_dir = input_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

metrics = [
    "mean_pairwise_distance",
    "endpoint_variance",
    "jerk",
    "seam_discontinuity",
    "prefix_error",
]

for metric in metrics:
    pivot = df.pivot(index="obs_id", columns="method", values=metric)

    if "naive" not in pivot.columns or "training_free_rtc" not in pivot.columns:
        continue

    plt.figure()
    for obs_id, row in pivot.iterrows():
        plt.plot(
            ["naive", "training_free_rtc"],
            [row["naive"], row["training_free_rtc"]],
            marker="o",
            alpha=0.6,
        )

    plt.ylabel(metric)
    plt.title(f"Paired per-observation {metric}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"paired_{metric}.png", dpi=200)
    plt.close()

print("saved paired metric plots")
