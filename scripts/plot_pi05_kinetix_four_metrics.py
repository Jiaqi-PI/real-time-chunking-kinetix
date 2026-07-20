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

MAIN_METRICS = (
    "mean_pairwise_distance",
    "prefix_error",
    "jerk",
    "seam_discontinuity",
)

DISPLAY_NAMES = {
    "mean_pairwise_distance": (
        "Mean Pairwise Distance"
    ),
    "prefix_error": "Prefix Error",
    "jerk": "Jerk",
    "seam_discontinuity": (
        "Seam Discontinuity"
    ),
    "endpoint_variance": (
        "Endpoint Variance"
    ),
}


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pi05-metric-dir",
        type=Path,
        default=Path(
            "rtc_diag_pi05_fixed_obs_64/"
            "exact_old_metrics"
        ),
    )

    parser.add_argument(
        "--old-comparison-dir",
        type=Path,
        default=Path(
            "rtc_diag_stage_final_comparison/"
            "metrics"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "rtc_diag_pi05_fixed_obs_64/"
            "exact_old_metrics/figures"
        ),
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pi05_per_obs_path = (
        args.pi05_metric_dir
        / "per_obs_metrics.csv"
    )

    pi05_stage_summary_path = (
        args.pi05_metric_dir
        / "stage_summary_metrics.csv"
    )

    old_stage_summary_path = (
        args.old_comparison_dir
        / "stage_summary_metrics.csv"
    )

    for path in (
        pi05_per_obs_path,
        pi05_stage_summary_path,
        old_stage_summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    pi05_rows = read_csv(
        pi05_per_obs_path
    )

    pi05_stage_rows = read_csv(
        pi05_stage_summary_path
    )

    old_stage_rows = read_csv(
        old_stage_summary_path
    )

    # ------------------------------------------------------
    # Figure 1: pi0.5 only, four metrics by stage.
    # ------------------------------------------------------

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    axes = axes.reshape(-1)

    for axis, metric in zip(
        axes,
        MAIN_METRICS,
        strict=True,
    ):
        values_by_stage = []

        for stage in STAGES:
            values = [
                float(row[metric])
                for row in pi05_rows
                if row["stage"] == stage
            ]

            if len(values) != 10:
                raise RuntimeError(
                    f"{stage}/{metric}: "
                    f"expected 10 values, "
                    f"got {len(values)}"
                )

            values_by_stage.append(
                values
            )

        axis.boxplot(
            values_by_stage,
            tick_labels=STAGES,
        )

        for stage_index, values in enumerate(
            values_by_stage,
            start=1,
        ):
            axis.scatter(
                np.full(
                    len(values),
                    stage_index,
                    dtype=np.float64,
                ),
                values,
                alpha=0.7,
            )

        axis.set_title(
            DISPLAY_NAMES[metric]
        )

        axis.set_xlabel("Stage")
        axis.set_ylabel(
            DISPLAY_NAMES[metric]
        )

        axis.grid(
            True,
            alpha=0.3,
        )

    figure.suptitle(
        "pi0.5 Kinetix: "
        "Four RTC Diagnostic Metrics"
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    pi05_four_png = (
        args.out_dir
        / "pi05_four_metrics_by_stage.png"
    )

    pi05_four_pdf = (
        args.out_dir
        / "pi05_four_metrics_by_stage.pdf"
    )

    figure.savefig(
        pi05_four_png,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pi05_four_pdf,
        bbox_inches="tight",
    )

    plt.close(figure)

    # ------------------------------------------------------
    # Figure 2: old four methods + pi0.5.
    # Uses already-computed exact-old-protocol stage means.
    # ------------------------------------------------------

    all_stage_rows = []

    for row in old_stage_rows:
        all_stage_rows.append(
            dict(row)
        )

    for row in pi05_stage_rows:
        copied = dict(row)
        copied["method"] = "pi0.5 native"

        all_stage_rows.append(
            copied
        )

    methods = []

    for row in all_stage_rows:
        method = row["method"]

        if method not in methods:
            methods.append(method)

    print(
        "COMPARISON_METHODS =",
        methods,
        flush=True,
    )

    x = np.arange(
        len(methods),
        dtype=np.float64,
    )

    width = 0.24

    offsets = (
        -width,
        0.0,
        width,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(18, 10),
    )

    axes = axes.reshape(-1)

    for axis, metric in zip(
        axes,
        MAIN_METRICS,
        strict=True,
    ):
        for stage, offset in zip(
            STAGES,
            offsets,
            strict=True,
        ):
            stage_values = []

            for method in methods:
                matches = [
                    row
                    for row in all_stage_rows
                    if (
                        row["stage"] == stage
                        and row["method"] == method
                    )
                ]

                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one row for "
                        f"stage={stage!r} "
                        f"method={method!r}; "
                        f"got {len(matches)}"
                    )

                stage_values.append(
                    float(
                        matches[0][metric]
                    )
                )

            axis.bar(
                x + offset,
                stage_values,
                width=width,
                label=stage,
            )

        axis.set_title(
            DISPLAY_NAMES[metric]
        )

        axis.set_xticks(x)

        axis.set_xticklabels(
            methods,
            rotation=20,
            ha="right",
        )

        axis.set_ylabel(
            DISPLAY_NAMES[metric]
        )

        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        axis.legend()

    figure.suptitle(
        "Kinetix RTC Diagnostic: "
        "Old Four Methods vs pi0.5 Native"
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    comparison_png = (
        args.out_dir
        / "old_four_methods_vs_pi05_four_metrics.png"
    )

    comparison_pdf = (
        args.out_dir
        / "old_four_methods_vs_pi05_four_metrics.pdf"
    )

    figure.savefig(
        comparison_png,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        comparison_pdf,
        bbox_inches="tight",
    )

    plt.close(figure)

    # ------------------------------------------------------
    # Endpoint variance: auxiliary fifth metric.
    # ------------------------------------------------------

    endpoint_values = []

    for stage in STAGES:
        values = [
            float(
                row["endpoint_variance"]
            )
            for row in pi05_rows
            if row["stage"] == stage
        ]

        endpoint_values.append(
            values
        )

    figure, axis = plt.subplots(
        figsize=(7, 5),
    )

    axis.boxplot(
        endpoint_values,
        tick_labels=STAGES,
    )

    for stage_index, values in enumerate(
        endpoint_values,
        start=1,
    ):
        axis.scatter(
            np.full(
                len(values),
                stage_index,
                dtype=np.float64,
            ),
            values,
            alpha=0.7,
        )

    axis.set_title(
        "pi0.5 Kinetix: Endpoint Variance"
    )

    axis.set_xlabel("Stage")
    axis.set_ylabel("Endpoint Variance")

    axis.grid(
        True,
        alpha=0.3,
    )

    endpoint_png = (
        args.out_dir
        / "pi05_endpoint_variance_by_stage.png"
    )

    figure.savefig(
        endpoint_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("===== OUTPUTS =====", flush=True)
    print(
        "PI05_FOUR_METRICS =",
        pi05_four_png,
        flush=True,
    )
    print(
        "COMPARISON =",
        comparison_png,
        flush=True,
    )
    print(
        "ENDPOINT_VARIANCE =",
        endpoint_png,
        flush=True,
    )

    print(
        "PI05_FOUR_METRIC_PLOTS_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
