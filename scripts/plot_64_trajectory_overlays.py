from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Use the stage diagnostic by default.
# obs_000 is early stage; obs_010 is middle; obs_020 is late if you used 10 per stage.
obs_id = "obs_000"

base_raw = Path("rtc_diag_stage_base/raw")
tt_raw = Path("rtc_diag_stage_tt_full_fixedprev/raw")

out_dir = Path("rtc_diag_stage_final_comparison/trajectory_overlays")
out_dir.mkdir(parents=True, exist_ok=True)

methods = {
    "Ordinary BC naive": base_raw / obs_id / "naive_actions.npy",
    "Ordinary BC runtime RTC": base_raw / obs_id / "training_free_rtc_actions.npy",
    "TT-RTC ckpt naive": tt_raw / obs_id / "naive_actions.npy",
    "TT-RTC ckpt conditioned": tt_raw / obs_id / "training_free_rtc_actions.npy",
}

actions = {}
for name, path in methods.items():
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    print(name, arr.shape)
    actions[name] = arr

# Shape check.
for name, arr in actions.items():
    assert arr.ndim == 3, (name, arr.shape)

num_samples, horizon, action_dim = next(iter(actions.values())).shape
t = np.arange(horizon)


# ------------------------------------------------------------
# 1. Per-action-dimension overlay.
# One figure per method, 6 panels.
# ------------------------------------------------------------

for method_name, arr in actions.items():
    fig, axes = plt.subplots(action_dim, 1, figsize=(8, 2.0 * action_dim), sharex=True)

    if action_dim == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        for s in range(num_samples):
            ax.plot(t, arr[s, :, d], alpha=0.25, linewidth=0.8)
        ax.set_ylabel(f"a[{d}]")

    axes[-1].set_xlabel("chunk timestep")
    fig.suptitle(f"{obs_id} - 64 action chunks - {method_name}")
    fig.tight_layout()
    safe = method_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    fig.savefig(out_dir / f"{obs_id}_{safe}_action_dim_overlay.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# 2. Shared PCA 2D action-space trajectory overlay.
# Fit PCA using all samples from all four methods together.
# ------------------------------------------------------------

all_points = []
for arr in actions.values():
    all_points.append(arr.reshape(-1, action_dim))
all_points = np.concatenate(all_points, axis=0)

mean = all_points.mean(axis=0, keepdims=True)
centered = all_points - mean

# SVD PCA.
_, _, vh = np.linalg.svd(centered, full_matrices=False)
components = vh[:2].T

# Shared axis limits.
projected_all = centered @ components
x_min, x_max = projected_all[:, 0].min(), projected_all[:, 0].max()
y_min, y_max = projected_all[:, 1].min(), projected_all[:, 1].max()
x_pad = 0.05 * (x_max - x_min + 1e-8)
y_pad = 0.05 * (y_max - y_min + 1e-8)

for method_name, arr in actions.items():
    traj = (arr.reshape(-1, action_dim) - mean) @ components
    traj = traj.reshape(num_samples, horizon, 2)

    plt.figure(figsize=(6, 5))
    for s in range(num_samples):
        plt.plot(traj[s, :, 0], traj[s, :, 1], alpha=0.35, linewidth=0.9)
        plt.scatter(traj[s, 0, 0], traj[s, 0, 1], s=8, alpha=0.25)
        plt.scatter(traj[s, -1, 0], traj[s, -1, 1], s=8, alpha=0.25)

    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{obs_id} - 64 action trajectories - {method_name}")
    plt.tight_layout()
    safe = method_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    plt.savefig(out_dir / f"{obs_id}_{safe}_pca_trajectory_overlay.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# 3. Combined 2x2 PCA plot for direct visual comparison.
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
axes = axes.ravel()

for ax, (method_name, arr) in zip(axes, actions.items()):
    traj = (arr.reshape(-1, action_dim) - mean) @ components
    traj = traj.reshape(num_samples, horizon, 2)

    for s in range(num_samples):
        ax.plot(traj[s, :, 0], traj[s, :, 1], alpha=0.35, linewidth=0.9)

    ax.set_title(method_name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

fig.suptitle(f"{obs_id} - same observation, 64 sampled action trajectories")
fig.tight_layout()
fig.savefig(out_dir / f"{obs_id}_four_methods_pca_trajectory_overlay.png", dpi=200)
plt.close(fig)

print("Saved trajectory overlay plots to:", out_dir)
