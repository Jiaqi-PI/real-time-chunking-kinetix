import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pairwise_distance(chunks: np.ndarray) -> np.ndarray:
    # chunks: [N, H, D]
    diff = chunks[:, None, :, :] - chunks[None, :, :, :]
    dist = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    return dist


def jerk_metric(chunks: np.ndarray) -> float:
    # second-order difference as a simple smoothness proxy
    if chunks.shape[1] < 3:
        return float("nan")
    dd = chunks[:, 2:, :] - 2 * chunks[:, 1:-1, :] + chunks[:, :-2, :]
    return float(np.linalg.norm(dd, axis=-1).mean())


def endpoint_variance(chunks: np.ndarray) -> float:
    endpoints = chunks[:, -1, :]
    return float(np.var(endpoints, axis=0).mean())


def seam_discontinuity(chunks: np.ndarray, prev: np.ndarray, inference_delay: int) -> float:
    # Compare new chunk at delay with previous chunk just before delay.
    # If delay == 0, compare first new action with first previous action.
    if inference_delay <= 0:
        return float(np.linalg.norm(chunks[:, 0, :] - prev[0, :], axis=-1).mean())
    d = min(inference_delay, chunks.shape[1] - 1)
    return float(np.linalg.norm(chunks[:, d, :] - prev[d - 1, :], axis=-1).mean())


def prefix_error(chunks: np.ndarray, prev: np.ndarray, inference_delay: int) -> float:
    if inference_delay <= 0:
        return 0.0
    d = min(inference_delay, chunks.shape[1])
    err = chunks[:, :d, :] - prev[None, :d, :]
    return float(np.linalg.norm(err, axis=-1).mean())


def pca_2d(chunks: np.ndarray) -> np.ndarray:
    x = chunks.reshape(chunks.shape[0], -1)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def main():
    input_dir = Path("rtc_diag_raw_64_training_time")
    fig_dir = input_dir / "figures"
    metric_dir = input_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    global_meta_path = input_dir / "raw" / "metadata.json"
    with global_meta_path.open() as f:
        global_meta = json.load(f)

    inference_delay = int(global_meta["inference_delay"])

    rows = []

    for obs_dir in sorted((input_dir / "raw").glob("obs_*")):
        prev = np.load(obs_dir / "prev_action_chunk.npy")

        for action_file in [
            obs_dir / "naive_actions.npy",
            obs_dir / "training_free_rtc_actions.npy",
        ]:
            if not action_file.exists():
                continue

            method = action_file.stem.replace("_actions", "")
            chunks = np.load(action_file)  # [N, H, D]

            dist = pairwise_distance(chunks)
            upper = dist[np.triu_indices(dist.shape[0], k=1)]

            rows.append(
                {
                    "obs_id": obs_dir.name,
                    "method": method,
                    "num_samples": chunks.shape[0],
                    "action_horizon": chunks.shape[1],
                    "action_dim": chunks.shape[2],
                    "mean_pairwise_distance": float(upper.mean()),
                    "std_pairwise_distance": float(upper.std()),
                    "endpoint_variance": endpoint_variance(chunks),
                    "jerk": jerk_metric(chunks),
                    "seam_discontinuity": seam_discontinuity(chunks, prev, inference_delay),
                    "prefix_error": prefix_error(chunks, prev, inference_delay),
                }
            )

            # Save pairwise matrix.
            np.save(metric_dir / f"{obs_dir.name}_{method}_pairwise_distance.npy", dist)

            # 1. Pairwise heatmap.
            plt.figure()
            plt.imshow(dist, aspect="auto")
            plt.colorbar(label="pairwise distance")
            plt.xlabel("sample id")
            plt.ylabel("sample id")
            plt.title(f"{obs_dir.name} - {method} - pairwise distance")
            plt.tight_layout()
            plt.savefig(fig_dir / f"{obs_dir.name}_{method}_pairwise_heatmap.png", dpi=200)
            plt.close()

            # 2. Trajectory overlay for first 3 action dims.
            num_dims = min(3, chunks.shape[-1])
            for d in range(num_dims):
                plt.figure()
                for i in range(chunks.shape[0]):
                    plt.plot(chunks[i, :, d], alpha=0.25)
                plt.xlabel("timestep")
                plt.ylabel(f"action dim {d}")
                plt.title(f"{obs_dir.name} - {method} - action dim {d}")
                plt.tight_layout()
                plt.savefig(fig_dir / f"{obs_dir.name}_{method}_traj_dim{d}.png", dpi=200)
                plt.close()

            # 3. PCA scatter.
            if chunks.shape[0] >= 3:
                z = pca_2d(chunks)
                plt.figure()
                plt.scatter(z[:, 0], z[:, 1])
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.title(f"{obs_dir.name} - {method} - PCA of action chunks")
                plt.tight_layout()
                plt.savefig(fig_dir / f"{obs_dir.name}_{method}_pca.png", dpi=200)
                plt.close()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metric_dir / "per_obs_metrics.csv", index=False)

    summary = metrics.groupby("method").mean(numeric_only=True).reset_index()
    summary.to_csv(metric_dir / "summary_metrics.csv", index=False)

    for metric in [
        "mean_pairwise_distance",
        "endpoint_variance",
        "jerk",
        "seam_discontinuity",
        "prefix_error",
    ]:
        plt.figure()
        plt.bar(summary["method"], summary[metric])
        plt.ylabel(metric)
        plt.title(metric)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(fig_dir / f"summary_{metric}.png", dpi=200)
        plt.close()

    print("Saved metrics to:", metric_dir)
    print("Saved figures to:", fig_dir)
    print(summary)


if __name__ == "__main__":
    main()
