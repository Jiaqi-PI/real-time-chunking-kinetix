from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


KS = (1, 2, 4)


ACTION_STD = np.asarray(
    [
        0.6498798727989197,
        0.7588233351707458,
        0.8182611465454102,
        0.6878356337547302,
        0.28773990273475647,
        0.29553285241127014,
    ],
    dtype=np.float64,
)


def action_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != (6,) or y.shape != (6,):
        raise ValueError(
            f"Expected 6-D actions, got {x.shape} and {y.shape}"
        )

    diff_normalized = (x - y) / ACTION_STD

    return float(
        np.sqrt(
            np.mean(diff_normalized ** 2)
        )
    )


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str,
    num_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Bad bootstrap input shape={values.shape}")

    if statistic == "mean":
        fn = np.mean
    elif statistic == "median":
        fn = np.median
    else:
        raise ValueError(statistic)

    rng = np.random.default_rng(seed)
    n = values.size

    samples = rng.choice(
        values,
        size=(num_bootstrap, n),
        replace=True,
    )
    stats = fn(samples, axis=1)

    return (
        float(np.percentile(stats, 2.5)),
        float(np.percentile(stats, 97.5)),
    )


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(root.glob("shard_*"))

    if not shard_dirs:
        raise RuntimeError(f"No shard_* directories under {root}")

    timestep_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    for shard_dir in shard_dirs:
        metadata = json.loads(
            (shard_dir / "metadata.json").read_text()
        )

        shard_id = int(metadata["shard_id"])
        num_evals = int(metadata["num_evals"])

        chunks = np.load(
            shard_dir / "chunks_raw.npy",
            mmap_mode="r",
        )
        valid = np.load(
            shard_dir / "valid.npy",
        )
        lengths = np.load(
            shard_dir / "episode_lengths.npy",
        )

        if chunks.ndim != 4 or chunks.shape[-2:] != (8, 6):
            raise RuntimeError(
                f"{shard_dir}: bad chunks shape={chunks.shape}"
            )

        if valid.shape != chunks.shape[:2]:
            raise RuntimeError(
                f"{shard_dir}: valid shape mismatch "
                f"{valid.shape} vs {chunks.shape[:2]}"
            )

        if len(lengths) != num_evals:
            raise RuntimeError(
                f"{shard_dir}: lengths mismatch"
            )

        for env_id in range(num_evals):
            trajectory_id = shard_id * num_evals + env_id
            length = int(lengths[env_id])

            valid_indices = np.flatnonzero(valid[env_id])

            expected_indices = np.arange(length)

            if not np.array_equal(
                valid_indices,
                expected_indices,
            ):
                raise RuntimeError(
                    f"trajectory={trajectory_id}: "
                    "valid steps are not contiguous from zero"
                )

            if length < max(KS) + 1:
                raise RuntimeError(
                    f"trajectory={trajectory_id}: "
                    f"too short length={length}"
                )

            for k in KS:
                wrong_values = []
                aligned_values = []

                for t in range(length - k):
                    c_t = np.asarray(
                        chunks[env_id, t],
                        dtype=np.float32,
                    )
                    c_future = np.asarray(
                        chunks[env_id, t + k],
                        dtype=np.float32,
                    )

                    wrong = action_rmse(
                        c_t[0],
                        c_future[0],
                    )

                    aligned = action_rmse(
                        c_t[k],
                        c_future[0],
                    )

                    gain = wrong - aligned

                    wrong_values.append(wrong)
                    aligned_values.append(aligned)

                    timestep_rows.append(
                        {
                            "trajectory_id": trajectory_id,
                            "shard_id": shard_id,
                            "env_id": env_id,
                            "t": t,
                            "k": k,
                            "wrong_prefix_error": wrong,
                            "aligned_slot_error": aligned,
                            "alignment_gain": gain,
                        }
                    )

                wrong_values_np = np.asarray(
                    wrong_values,
                    dtype=np.float64,
                )
                aligned_values_np = np.asarray(
                    aligned_values,
                    dtype=np.float64,
                )
                gain_values_np = (
                    wrong_values_np - aligned_values_np
                )

                trajectory_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "shard_id": shard_id,
                        "env_id": env_id,
                        "trajectory_length": length,
                        "k": k,
                        "num_pairs": int(length - k),
                        "wrong_prefix_error": float(
                            wrong_values_np.mean()
                        ),
                        "aligned_slot_error": float(
                            aligned_values_np.mean()
                        ),
                        "alignment_gain": float(
                            gain_values_np.mean()
                        ),
                    }
                )

    write_csv(
        out_dir / "per_timestep_metrics.csv",
        timestep_rows,
    )

    write_csv(
        out_dir / "per_trajectory_metrics.csv",
        trajectory_rows,
    )

    summary_rows: list[dict[str, object]] = []

    for k in KS:
        selected = [
            row
            for row in trajectory_rows
            if int(row["k"]) == k
        ]

        if not selected:
            raise RuntimeError(f"No trajectories for k={k}")

        row: dict[str, object] = {
            "k": k,
            "num_trajectories": len(selected),
        }

        for metric in (
            "wrong_prefix_error",
            "aligned_slot_error",
            "alignment_gain",
        ):
            values = np.asarray(
                [
                    float(item[metric])
                    for item in selected
                ],
                dtype=np.float64,
            )

            mean_low, mean_high = bootstrap_ci(
                values,
                statistic="mean",
                num_bootstrap=args.num_bootstrap,
                seed=args.bootstrap_seed + k,
            )

            median_low, median_high = bootstrap_ci(
                values,
                statistic="median",
                num_bootstrap=args.num_bootstrap,
                seed=args.bootstrap_seed + 100 + k,
            )

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(
                np.median(values)
            )
            row[f"{metric}_mean_ci_low"] = mean_low
            row[f"{metric}_mean_ci_high"] = mean_high
            row[f"{metric}_median_ci_low"] = median_low
            row[f"{metric}_median_ci_high"] = median_high

        summary_rows.append(row)

    write_csv(
        out_dir / "summary.csv",
        summary_rows,
    )

    analysis_metadata = {
        "source_root": str(root),
        "ks": list(KS),
        "metric": "sqrt(mean((x-y)^2)) over 6 action dims",
        "wrong_definition": "C_t[0] vs C_{t+k}[0]",
        "aligned_definition": "C_t[k] vs C_{t+k}[0]",
        "gain_definition": "wrong - aligned",
        "aggregation": (
            "mean over timestep pairs within trajectory, "
            "then aggregate across trajectories"
        ),
        "num_bootstrap": args.num_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "normalization": "training_action_std",
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(
            analysis_metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("===== PART A CROSS-TIME SUMMARY =====")

    for row in summary_rows:
        print(
            f"k={row['k']} "
            f"n={row['num_trajectories']} "
            f"wrong={row['wrong_prefix_error_mean']:.6f} "
            f"aligned={row['aligned_slot_error_mean']:.6f} "
            f"gain={row['alignment_gain_mean']:.6f}"
        )

    print(f"WROTE={out_dir / 'per_timestep_metrics.csv'}")
    print(f"WROTE={out_dir / 'per_trajectory_metrics.csv'}")
    print(f"WROTE={out_dir / 'summary.csv'}")
    print("WEEK1_PARTA_ANALYSIS_OK")


if __name__ == "__main__":
    main()
