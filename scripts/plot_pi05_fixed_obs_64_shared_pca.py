from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


STAGES = (
    "early",
    "middle",
    "late",
)

EXPECTED_ACTION_SHAPE = (
    10,
    64,
    8,
    6,
)


def fit_shared_pca(
    stage_actions: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    all_vectors = np.concatenate(
        [
            stage_actions[stage].reshape(-1, 6)
            for stage in STAGES
        ],
        axis=0,
    )

    print(
        "all_vectors.shape =",
        all_vectors.shape,
        flush=True,
    )

    expected_vectors = (
        len(STAGES)
        * 10
        * 64
        * 8
    )

    if all_vectors.shape != (
        expected_vectors,
        6,
    ):
        raise RuntimeError(
            "Unexpected PCA input shape: "
            f"{all_vectors.shape}"
        )

    mean = all_vectors.mean(axis=0)

    centered = (
        all_vectors
        - mean[None, :]
    )

    _, singular_values, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    components = vh[:2]

    explained_variance = (
        singular_values ** 2
        / (centered.shape[0] - 1)
    )

    explained_ratio = (
        explained_variance
        / explained_variance.sum()
    )

    return (
        mean,
        components,
        explained_ratio,
    )


def project_actions(
    actions: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    flat = actions.reshape(-1, 6)

    projected = (
        flat
        - mean[None, :]
    ) @ components.T

    return projected.reshape(
        10,
        64,
        8,
        2,
    )


def compute_limits(
    projected: dict[str, np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_points = np.concatenate(
        [
            projected[stage].reshape(-1, 2)
            for stage in STAGES
        ],
        axis=0,
    )

    x_min = float(all_points[:, 0].min())
    x_max = float(all_points[:, 0].max())

    y_min = float(all_points[:, 1].min())
    y_max = float(all_points[:, 1].max())

    x_pad = max(
        1e-6,
        0.05 * (x_max - x_min),
    )

    y_pad = max(
        1e-6,
        0.05 * (y_max - y_min),
    )

    return (
        (x_min - x_pad, x_max + x_pad),
        (y_min - y_pad, y_max + y_pad),
    )


def compute_statistics(
    stage: str,
    actions: np.ndarray,
    projected: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows: list[
        dict[str, float | int | str]
    ] = []

    for observation_index in range(10):
        obs_actions = actions[
            observation_index
        ]

        obs_projected = projected[
            observation_index
        ]

        mean_chunk = obs_actions.mean(
            axis=0,
            keepdims=True,
        )

        chunk_distances = np.linalg.norm(
            (
                obs_actions
                - mean_chunk
            ).reshape(64, -1),
            axis=1,
        )

        endpoints = obs_projected[:, -1, :]

        endpoint_center = endpoints.mean(
            axis=0,
            keepdims=True,
        )

        endpoint_distances = np.linalg.norm(
            endpoints
            - endpoint_center,
            axis=1,
        )

        mean_trajectory = obs_projected.mean(
            axis=0,
        )

        mean_trajectory_length = float(
            np.linalg.norm(
                np.diff(
                    mean_trajectory,
                    axis=0,
                ),
                axis=1,
            ).sum()
        )

        rows.append(
            {
                "stage": stage,
                "observation_index": (
                    observation_index
                ),
                "mean_chunk_dispersion": float(
                    chunk_distances.mean()
                ),
                "std_chunk_dispersion": float(
                    chunk_distances.std()
                ),
                "mean_endpoint_dispersion": float(
                    endpoint_distances.mean()
                ),
                "std_endpoint_dispersion": float(
                    endpoint_distances.std()
                ),
                "mean_trajectory_length": (
                    mean_trajectory_length
                ),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "rtc_diag_pi05_fixed_obs_64/"
            "pi05_fixed_obs_64.npz"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "rtc_diag_pi05_fixed_obs_64/"
            "shared_pca_3stage"
        ),
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("===== LOAD ACTIONS =====", flush=True)

    stage_actions: dict[
        str,
        np.ndarray,
    ] = {}

    with np.load(
        args.input,
        allow_pickle=False,
    ) as data:
        for stage in STAGES:
            key = f"{stage}_actions"

            if key not in data.files:
                raise RuntimeError(
                    f"Missing key: {key}"
                )

            actions = np.asarray(
                data[key],
                dtype=np.float64,
            )

            print(
                f"{stage}: {actions.shape}",
                flush=True,
            )

            if (
                actions.shape
                != EXPECTED_ACTION_SHAPE
            ):
                raise RuntimeError(
                    f"{stage}: expected "
                    f"{EXPECTED_ACTION_SHAPE}, "
                    f"got {actions.shape}"
                )

            if not np.isfinite(
                actions
            ).all():
                raise RuntimeError(
                    f"{stage}: non-finite actions"
                )

            stage_actions[stage] = actions

    print("===== FIT SHARED PCA =====", flush=True)

    (
        pca_mean,
        pca_components,
        explained_ratio,
    ) = fit_shared_pca(
        stage_actions
    )

    print(
        "PC1 explained variance ratio =",
        explained_ratio[0],
        flush=True,
    )

    print(
        "PC2 explained variance ratio =",
        explained_ratio[1],
        flush=True,
    )

    print(
        "PC1+PC2 explained variance ratio =",
        explained_ratio[:2].sum(),
        flush=True,
    )

    projected: dict[
        str,
        np.ndarray,
    ] = {}

    for stage in STAGES:
        projected[stage] = project_actions(
            stage_actions[stage],
            pca_mean,
            pca_components,
        )

        print(
            f"{stage} projected = "
            f"{projected[stage].shape}",
            flush=True,
        )

    np.savez_compressed(
        args.out_dir / "shared_pca_basis.npz",
        mean=pca_mean,
        components=pca_components,
        explained_variance_ratio=(
            explained_ratio
        ),
    )

    np.savez_compressed(
        args.out_dir / "projected_actions.npz",
        **{
            stage: projected[stage]
            for stage in STAGES
        },
    )

    (
        x_limits,
        y_limits,
    ) = compute_limits(projected)

    print(
        "x_limits =",
        x_limits,
        flush=True,
    )

    print(
        "y_limits =",
        y_limits,
        flush=True,
    )

    # ---------------------------------------------------------
    # Figure 1:
    # 10 mean trajectories per stage.
    # Each trajectory is the mean of 64 repeated inferences
    # at one fixed observation.
    # ---------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
        sharex=True,
        sharey=True,
    )

    for axis, stage in zip(
        axes,
        STAGES,
        strict=True,
    ):
        mean_trajectories = projected[
            stage
        ].mean(axis=1)

        for observation_index in range(10):
            trajectory = mean_trajectories[
                observation_index
            ]

            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                marker="o",
                markersize=3,
                linewidth=1.5,
                alpha=0.8,
                label=(
                    f"obs {observation_index}"
                ),
            )

        axis.set_title(
            f"{stage.capitalize()} stage"
        )

        axis.set_xlabel(
            "PC1 "
            f"({explained_ratio[0] * 100:.1f}%)"
        )

        axis.set_ylabel(
            "PC2 "
            f"({explained_ratio[1] * 100:.1f}%)"
        )

        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)

        axis.grid(
            True,
            alpha=0.3,
        )

        axis.legend(
            fontsize=7,
            ncol=2,
        )

    figure.suptitle(
        "pi0.5 Kinetix: "
        "Mean Action-Chunk Trajectories "
        "under Shared PCA"
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.94)
    )

    mean_png = (
        args.out_dir
        / "pi05_3stage_mean_trajectories.png"
    )

    mean_pdf = (
        args.out_dir
        / "pi05_3stage_mean_trajectories.pdf"
    )

    figure.savefig(
        mean_png,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        mean_pdf,
        bbox_inches="tight",
    )

    plt.close(figure)

    # ---------------------------------------------------------
    # Figure 2:
    # Full 64 repeated trajectories for every observation.
    # 3 rows x 10 observations.
    # ---------------------------------------------------------

    figure, axes = plt.subplots(
        3,
        10,
        figsize=(30, 9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for stage_index, stage in enumerate(
        STAGES
    ):
        for observation_index in range(10):
            axis = axes[
                stage_index,
                observation_index,
            ]

            trajectories = projected[
                stage
            ][observation_index]

            for trajectory in trajectories:
                axis.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    linewidth=0.6,
                    alpha=0.10,
                )

            mean_trajectory = trajectories.mean(
                axis=0
            )

            axis.plot(
                mean_trajectory[:, 0],
                mean_trajectory[:, 1],
                linewidth=2.5,
                marker="o",
                markersize=3,
            )

            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)

            axis.grid(
                True,
                alpha=0.2,
            )

            axis.set_title(
                f"{stage} #{observation_index}",
                fontsize=9,
            )

            if observation_index == 0:
                axis.set_ylabel(
                    "PC2"
                )

            if stage_index == (
                len(STAGES) - 1
            ):
                axis.set_xlabel(
                    "PC1"
                )

    figure.suptitle(
        "pi0.5 Kinetix: "
        "64 Repeated Action-Chunk Samples "
        "per Fixed Observation"
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    repeated_png = (
        args.out_dir
        / "pi05_3stage_64repeat_grid.png"
    )

    repeated_pdf = (
        args.out_dir
        / "pi05_3stage_64repeat_grid.pdf"
    )

    figure.savefig(
        repeated_png,
        dpi=250,
        bbox_inches="tight",
    )

    figure.savefig(
        repeated_pdf,
        bbox_inches="tight",
    )

    plt.close(figure)

    # ---------------------------------------------------------
    # Statistics.
    # ---------------------------------------------------------

    statistics: list[
        dict[str, float | int | str]
    ] = []

    for stage in STAGES:
        statistics.extend(
            compute_statistics(
                stage,
                stage_actions[stage],
                projected[stage],
            )
        )

    csv_path = (
        args.out_dir
        / "per_observation_statistics.csv"
    )

    fieldnames = [
        "stage",
        "observation_index",
        "mean_chunk_dispersion",
        "std_chunk_dispersion",
        "mean_endpoint_dispersion",
        "std_endpoint_dispersion",
        "mean_trajectory_length",
    ]

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            statistics
        )

    # ---------------------------------------------------------
    # Figure 3:
    # Stage-wise dispersion.
    # ---------------------------------------------------------

    chunk_dispersion = {
        stage: [
            float(
                row["mean_chunk_dispersion"]
            )
            for row in statistics
            if row["stage"] == stage
        ]
        for stage in STAGES
    }

    endpoint_dispersion = {
        stage: [
            float(
                row[
                    "mean_endpoint_dispersion"
                ]
            )
            for row in statistics
            if row["stage"] == stage
        ]
        for stage in STAGES
    }

    figure, axis = plt.subplots(
        figsize=(7, 5),
    )

    axis.boxplot(
        [
            chunk_dispersion[stage]
            for stage in STAGES
        ],
        labels=STAGES,
    )

    axis.set_xlabel("Stage")
    axis.set_ylabel(
        "Mean distance to "
        "per-observation mean chunk"
    )

    axis.set_title(
        "pi0.5 Kinetix: "
        "Action-Chunk Dispersion"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    chunk_png = (
        args.out_dir
        / "pi05_chunk_dispersion_by_stage.png"
    )

    figure.savefig(
        chunk_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(7, 5),
    )

    axis.boxplot(
        [
            endpoint_dispersion[stage]
            for stage in STAGES
        ],
        labels=STAGES,
    )

    axis.set_xlabel("Stage")
    axis.set_ylabel(
        "Mean endpoint dispersion in PCA space"
    )

    axis.set_title(
        "pi0.5 Kinetix: "
        "Action-Chunk Endpoint Dispersion"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    endpoint_png = (
        args.out_dir
        / "pi05_endpoint_dispersion_by_stage.png"
    )

    figure.savefig(
        endpoint_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    # ---------------------------------------------------------
    # Text summary.
    # ---------------------------------------------------------

    summary_lines = [
        f"input={args.input}",
        (
            "pca_input_shape="
            f"({len(STAGES) * 10 * 64 * 8}, 6)"
        ),
        (
            "pc1_explained_variance_ratio="
            f"{explained_ratio[0]:.8f}"
        ),
        (
            "pc2_explained_variance_ratio="
            f"{explained_ratio[1]:.8f}"
        ),
        (
            "pc1_pc2_explained_variance_ratio="
            f"{explained_ratio[:2].sum():.8f}"
        ),
        "",
    ]

    for stage in STAGES:
        stage_rows = [
            row
            for row in statistics
            if row["stage"] == stage
        ]

        mean_chunk_values = np.asarray(
            [
                row[
                    "mean_chunk_dispersion"
                ]
                for row in stage_rows
            ],
            dtype=np.float64,
        )

        endpoint_values = np.asarray(
            [
                row[
                    "mean_endpoint_dispersion"
                ]
                for row in stage_rows
            ],
            dtype=np.float64,
        )

        trajectory_lengths = np.asarray(
            [
                row[
                    "mean_trajectory_length"
                ]
                for row in stage_rows
            ],
            dtype=np.float64,
        )

        summary_lines.extend(
            [
                f"[{stage}]",
                (
                    "mean_chunk_dispersion="
                    f"{mean_chunk_values.mean():.8f}"
                ),
                (
                    "std_across_observations_chunk_dispersion="
                    f"{mean_chunk_values.std():.8f}"
                ),
                (
                    "mean_endpoint_dispersion="
                    f"{endpoint_values.mean():.8f}"
                ),
                (
                    "std_across_observations_endpoint_dispersion="
                    f"{endpoint_values.std():.8f}"
                ),
                (
                    "mean_trajectory_length="
                    f"{trajectory_lengths.mean():.8f}"
                ),
                "",
            ]
        )

    summary_path = (
        args.out_dir
        / "pca_summary.txt"
    )

    summary_path.write_text(
        "\n".join(summary_lines)
        + "\n"
    )

    print("===== OUTPUTS =====", flush=True)

    for path in sorted(
        args.out_dir.iterdir()
    ):
        if path.is_file():
            print(
                path,
                flush=True,
            )

    print(
        "MEAN_TRAJECTORY_PNG =",
        mean_png,
        flush=True,
    )

    print(
        "REPEATED_GRID_PNG =",
        repeated_png,
        flush=True,
    )

    print(
        "STATISTICS_CSV =",
        csv_path,
        flush=True,
    )

    print(
        "SUMMARY =",
        summary_path,
        flush=True,
    )

    print(
        "PI05_SHARED_PCA_3STAGE_PLOT_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
