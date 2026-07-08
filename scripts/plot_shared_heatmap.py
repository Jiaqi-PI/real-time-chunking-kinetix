from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

input_dir = Path("rtc_diag_raw_64")
fig_dir = input_dir / "figures"
metric_dir = input_dir / "metrics"
fig_dir.mkdir(parents=True, exist_ok=True)

for obs_id in ["obs_000"]:
    naive = np.load(metric_dir / f"{obs_id}_naive_pairwise_distance.npy")
    rtc = np.load(metric_dir / f"{obs_id}_training_free_rtc_pairwise_distance.npy")

    vmax = max(float(naive.max()), float(rtc.max()))

    for name, dist in [("naive", naive), ("training_free_rtc", rtc)]:
        plt.figure()
        plt.imshow(dist, aspect="auto", vmin=0.0, vmax=vmax)
        plt.colorbar(label="pairwise distance")
        plt.xlabel("sample id")
        plt.ylabel("sample id")
        plt.title(f"{obs_id} - {name} - shared scale")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{obs_id}_{name}_pairwise_heatmap_shared_scale.png", dpi=200)
        plt.close()

print("saved shared-scale heatmaps")
