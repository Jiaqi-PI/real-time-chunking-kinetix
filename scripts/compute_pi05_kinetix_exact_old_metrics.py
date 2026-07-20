from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np


STAGES = (
    "early",
    "middle",
    "late",
)

FUNCTION_NAMES = (
    "pairwise_distance",
    "jerk_metric",
    "endpoint_variance",
    "seam_discontinuity",
    "prefix_error",
)

METRIC_NAMES = (
    "mean_pairwise_distance",
    "std_pairwise_distance",
    "endpoint_variance",
    "jerk",
    "seam_discontinuity",
    "prefix_error",
)

EXPECTED_NUM_OBS = 30
EXPECTED_SAMPLES = 64
EXPECTED_HORIZON = 8
EXPECTED_ACTION_DIM = 6
EXPECTED_OBS_DIM = 679


def load_exact_metric_functions(
    source_path: Path,
) -> dict[str, Callable]:
    source = source_path.read_text()

    tree = ast.parse(
        source,
        filename=str(source_path),
    )

    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    missing = [
        name
        for name in FUNCTION_NAMES
        if name not in function_nodes
    ]

    if missing:
        raise RuntimeError(
            f"Missing metric functions in {source_path}: "
            f"{missing}"
        )

    selected_nodes = [
        function_nodes[name]
        for name in FUNCTION_NAMES
    ]

    module = ast.Module(
        body=selected_nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    namespace = {
        "np": np,
    }

    exec(
        compile(
            module,
            filename=str(source_path),
            mode="exec",
        ),
        namespace,
    )

    functions = {
        name: namespace[name]
        for name in FUNCTION_NAMES
    }

    print(
        "===== EXACT OLD METRIC FUNCTIONS =====",
        flush=True,
    )

    for name in FUNCTION_NAMES:
        node = function_nodes[name]

        segment = ast.get_source_segment(
            source,
            node,
        )

        print("", flush=True)
        print(
            f"----- {name} -----",
            flush=True,
        )
        print(segment, flush=True)

    print(
        "EXACT_OLD_METRIC_FUNCTIONS_LOADED",
        flush=True,
    )

    return functions


def load_delay_metadata(
    root: Path,
) -> tuple[int, int, list[str]]:
    found = []

    for path in sorted(
        root.rglob("*.json")
    ):
        try:
            data = json.loads(
                path.read_text()
            )
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if (
            "inference_delay" in data
            and "execute_horizon" in data
        ):
            found.append(
                (
                    path,
                    int(data["inference_delay"]),
                    int(data["execute_horizon"]),
                )
            )

    if not found:
        raise RuntimeError(
            f"No metadata with inference_delay and "
            f"execute_horizon under {root}"
        )

    pairs = {
        (delay, horizon)
        for _, delay, horizon in found
    }

    print(
        "===== DELAY METADATA SOURCES =====",
        flush=True,
    )

    for path, delay, horizon in found:
        print(
            f"{path}: "
            f"inference_delay={delay} "
            f"execute_horizon={horizon}",
            flush=True,
        )

    if len(pairs) != 1:
        raise RuntimeError(
            f"Inconsistent delay metadata: {pairs}"
        )

    inference_delay, execute_horizon = next(
        iter(pairs)
    )

    source_paths = [
        str(path.resolve())
        for path, _, _ in found
    ]

    return (
        inference_delay,
        execute_horizon,
        source_paths,
    )


def require_array_equal(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    name: str,
) -> None:
    if lhs.shape != rhs.shape:
        raise RuntimeError(
            f"{name}: shape mismatch "
            f"{lhs.shape} vs {rhs.shape}"
        )

    if not np.array_equal(lhs, rhs):
        max_diff = float(
            np.max(
                np.abs(
                    lhs.astype(np.float64)
                    - rhs.astype(np.float64)
                )
            )
        )

        raise RuntimeError(
            f"{name}: arrays are not exactly equal; "
            f"max_abs_diff={max_diff}"
        )

    print(
        f"{name}: EXACT_EQUAL shape={lhs.shape}",
        flush=True,
    )


def reconstruct_pi05_observation_order(
    pi05_path: Path,
    source_observations: np.ndarray,
    source_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(
        pi05_path,
        allow_pickle=False,
    ) as data:
        stage_actions = {
            stage: np.asarray(
                data[f"{stage}_actions"],
                dtype=np.float32,
            )
            for stage in STAGES
        }

        stage_states = {
            stage: np.asarray(
                data[f"{stage}_states"],
                dtype=np.float32,
            )
            for stage in STAGES
        }

    for stage in STAGES:
        if stage_actions[stage].shape != (
            10,
            EXPECTED_SAMPLES,
            EXPECTED_HORIZON,
            EXPECTED_ACTION_DIM,
        ):
            raise RuntimeError(
                f"{stage}_actions unexpected shape: "
                f"{stage_actions[stage].shape}"
            )

        if stage_states[stage].shape != (
            10,
            EXPECTED_OBS_DIM,
        ):
            raise RuntimeError(
                f"{stage}_states unexpected shape: "
                f"{stage_states[stage].shape}"
            )

    counters = {
        stage: 0
        for stage in STAGES
    }

    aligned_actions = []
    aligned_states = []

    for raw_label in source_labels:
        stage = str(raw_label)

        if stage not in counters:
            raise RuntimeError(
                f"Unexpected source stage label: {stage}"
            )

        stage_index = counters[stage]

        if stage_index >= 10:
            raise RuntimeError(
                f"Too many observations for stage {stage}"
            )

        aligned_actions.append(
            stage_actions[stage][stage_index]
        )

        aligned_states.append(
            stage_states[stage][stage_index]
        )

        counters[stage] += 1

    if counters != {
        "early": 10,
        "middle": 10,
        "late": 10,
    }:
        raise RuntimeError(
            f"Unexpected final stage counters: {counters}"
        )

    aligned_actions_array = np.stack(
        aligned_actions,
        axis=0,
    )

    aligned_states_array = np.stack(
        aligned_states,
        axis=0,
    )

    if aligned_actions_array.shape != (
        EXPECTED_NUM_OBS,
        EXPECTED_SAMPLES,
        EXPECTED_HORIZON,
        EXPECTED_ACTION_DIM,
    ):
        raise RuntimeError(
            "Unexpected aligned action shape: "
            f"{aligned_actions_array.shape}"
        )

    require_array_equal(
        aligned_states_array,
        source_observations,
        name=(
            "PI05_STATES_VS_STAGE_INPUT_OBSERVATIONS"
        ),
    )

    return (
        aligned_actions_array,
        aligned_states_array,
    )


def mean_rows(
    rows: list[dict[str, object]],
    *,
    stage: str | None,
) -> dict[str, object]:
    if stage is None:
        selected = rows
    else:
        selected = [
            row
            for row in rows
            if row["stage"] == stage
        ]

    if not selected:
        raise RuntimeError(
            f"No rows selected for stage={stage}"
        )

    output: dict[str, object] = {
        "method": "pi05_native",
        "num_samples": float(EXPECTED_SAMPLES),
        "action_horizon": float(EXPECTED_HORIZON),
        "action_dim": float(EXPECTED_ACTION_DIM),
    }

    if stage is not None:
        output["stage"] = stage

    for metric in METRIC_NAMES:
        output[metric] = float(
            np.mean(
                [
                    float(row[metric])
                    for row in selected
                ]
            )
        )

    return output


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pi05-input",
        type=Path,
        default=Path(
            "rtc_diag_pi05_fixed_obs_64/"
            "pi05_fixed_obs_64.npz"
        ),
    )

    parser.add_argument(
        "--stage-input-root",
        type=Path,
        default=Path(
            "rtc_stage_inputs_grasp_easy/raw"
        ),
    )

    parser.add_argument(
        "--old-base-root",
        type=Path,
        default=Path(
            "rtc_diag_stage_base"
        ),
    )

    parser.add_argument(
        "--old-tt-fixedprev-root",
        type=Path,
        default=Path(
            "rtc_diag_stage_tt_full_fixedprev"
        ),
    )

    parser.add_argument(
        "--metric-source",
        type=Path,
        default=Path(
            "scripts/"
            "plot_rtc_diagnosis_stage_base.py"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "rtc_diag_pi05_fixed_obs_64/"
            "exact_old_metrics"
        ),
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_dir = (
        args.out_dir
        / "pairwise_distance"
    )

    metric_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("===== INPUT CHECK =====", flush=True)

    required_paths = [
        args.pi05_input,
        args.stage_input_root / "observations.npy",
        args.stage_input_root / "stage_labels.npy",
        (
            args.old_base_root
            / "raw"
            / "observations.npy"
        ),
        (
            args.old_base_root
            / "raw"
            / "prev_action_chunks.npy"
        ),
        (
            args.old_tt_fixedprev_root
            / "raw"
            / "observations.npy"
        ),
        (
            args.old_tt_fixedprev_root
            / "raw"
            / "prev_action_chunks.npy"
        ),
        args.metric_source,
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        print(
            f"OK {path}",
            flush=True,
        )

    functions = load_exact_metric_functions(
        args.metric_source
    )

    (
        inference_delay,
        execute_horizon,
        metadata_sources,
    ) = load_delay_metadata(
        args.old_base_root
    )

    print("", flush=True)
    print("===== PROTOCOL =====", flush=True)
    print(
        "inference_delay =",
        inference_delay,
        flush=True,
    )
    print(
        "execute_horizon =",
        execute_horizon,
        flush=True,
    )

    if inference_delay != 2:
        raise RuntimeError(
            f"Expected old main inference_delay=2, "
            f"got {inference_delay}"
        )

    if execute_horizon != 4:
        raise RuntimeError(
            f"Expected old main execute_horizon=4, "
            f"got {execute_horizon}"
        )

    source_observations = np.asarray(
        np.load(
            args.stage_input_root / "observations.npy",
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    source_labels = np.asarray(
        np.load(
            args.stage_input_root / "stage_labels.npy",
            allow_pickle=False,
        )
    )

    if source_observations.shape != (
        EXPECTED_NUM_OBS,
        EXPECTED_OBS_DIM,
    ):
        raise RuntimeError(
            f"Unexpected source observations shape: "
            f"{source_observations.shape}"
        )

    if source_labels.shape != (
        EXPECTED_NUM_OBS,
    ):
        raise RuntimeError(
            f"Unexpected source labels shape: "
            f"{source_labels.shape}"
        )

    print(
        "source stage labels =",
        source_labels.tolist(),
        flush=True,
    )

    base_observations = np.asarray(
        np.load(
            args.old_base_root
            / "raw"
            / "observations.npy",
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    tt_observations = np.asarray(
        np.load(
            args.old_tt_fixedprev_root
            / "raw"
            / "observations.npy",
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    require_array_equal(
        source_observations,
        base_observations,
        name=(
            "STAGE_INPUT_VS_OLD_BASE_OBSERVATIONS"
        ),
    )

    require_array_equal(
        source_observations,
        tt_observations,
        name=(
            "STAGE_INPUT_VS_OLD_TT_OBSERVATIONS"
        ),
    )

    base_prev = np.asarray(
        np.load(
            args.old_base_root
            / "raw"
            / "prev_action_chunks.npy",
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    tt_prev = np.asarray(
        np.load(
            args.old_tt_fixedprev_root
            / "raw"
            / "prev_action_chunks.npy",
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    if base_prev.shape != (
        EXPECTED_NUM_OBS,
        EXPECTED_HORIZON,
        EXPECTED_ACTION_DIM,
    ):
        raise RuntimeError(
            f"Unexpected base previous chunks shape: "
            f"{base_prev.shape}"
        )

    require_array_equal(
        base_prev,
        tt_prev,
        name=(
            "OLD_BASE_VS_TT_FIXED_PREVIOUS_CHUNKS"
        ),
    )

    (
        pi05_actions,
        pi05_states,
    ) = reconstruct_pi05_observation_order(
        args.pi05_input,
        source_observations,
        source_labels,
    )

    print(
        "pi05_actions.shape =",
        pi05_actions.shape,
        flush=True,
    )

    print(
        "pi05_states.shape =",
        pi05_states.shape,
        flush=True,
    )

    pairwise_distance = functions[
        "pairwise_distance"
    ]

    jerk_metric = functions[
        "jerk_metric"
    ]

    endpoint_variance = functions[
        "endpoint_variance"
    ]

    seam_discontinuity = functions[
        "seam_discontinuity"
    ]

    prefix_error = functions[
        "prefix_error"
    ]

    rows: list[dict[str, object]] = []

    print("", flush=True)
    print(
        "===== COMPUTE EXACT OLD-PROTOCOL METRICS =====",
        flush=True,
    )

    for obs_index in range(
        EXPECTED_NUM_OBS
    ):
        obs_id = f"obs_{obs_index:03d}"

        stage = str(
            source_labels[obs_index]
        )

        chunks = pi05_actions[
            obs_index
        ]

        prev = base_prev[
            obs_index
        ]

        dist = np.asarray(
            pairwise_distance(chunks)
        )

        if dist.shape != (
            EXPECTED_SAMPLES,
            EXPECTED_SAMPLES,
        ):
            raise RuntimeError(
                f"{obs_id}: unexpected pairwise "
                f"matrix shape {dist.shape}"
            )

        upper = dist[
            np.triu_indices(
                dist.shape[0],
                k=1,
            )
        ]

        if upper.shape != (
            EXPECTED_SAMPLES
            * (EXPECTED_SAMPLES - 1)
            // 2,
        ):
            raise RuntimeError(
                f"{obs_id}: unexpected upper-triangle "
                f"shape {upper.shape}"
            )

        row = {
            "obs_id": obs_id,
            "stage": stage,
            "method": "pi05_native",
            "num_samples": EXPECTED_SAMPLES,
            "action_horizon": EXPECTED_HORIZON,
            "action_dim": EXPECTED_ACTION_DIM,
            "mean_pairwise_distance": float(
                upper.mean()
            ),
            "std_pairwise_distance": float(
                upper.std()
            ),
            "endpoint_variance": float(
                endpoint_variance(chunks)
            ),
            "jerk": float(
                jerk_metric(chunks)
            ),
            "seam_discontinuity": float(
                seam_discontinuity(
                    chunks,
                    prev,
                    inference_delay,
                )
            ),
            "prefix_error": float(
                prefix_error(
                    chunks,
                    prev,
                    inference_delay,
                )
            ),
        }

        rows.append(row)

        np.save(
            metric_dir
            / f"{obs_id}_pi05_native_pairwise_distance.npy",
            dist,
        )

        print(
            f"{obs_id} "
            f"stage={stage} "
            f"pairwise="
            f"{row['mean_pairwise_distance']:.8f} "
            f"prefix="
            f"{row['prefix_error']:.8f} "
            f"jerk="
            f"{row['jerk']:.8f} "
            f"seam="
            f"{row['seam_discontinuity']:.8f} "
            f"endpoint_var="
            f"{row['endpoint_variance']:.8f}",
            flush=True,
        )

    per_obs_fields = [
        "obs_id",
        "stage",
        "method",
        "num_samples",
        "action_horizon",
        "action_dim",
        *METRIC_NAMES,
    ]

    per_obs_path = (
        args.out_dir
        / "per_obs_metrics.csv"
    )

    write_csv(
        per_obs_path,
        rows,
        per_obs_fields,
    )

    overall_row = mean_rows(
        rows,
        stage=None,
    )

    summary_fields = [
        "method",
        "num_samples",
        "action_horizon",
        "action_dim",
        *METRIC_NAMES,
    ]

    summary_path = (
        args.out_dir
        / "summary_metrics.csv"
    )

    write_csv(
        summary_path,
        [overall_row],
        summary_fields,
    )

    stage_rows = [
        mean_rows(
            rows,
            stage=stage,
        )
        for stage in STAGES
    ]

    stage_summary_fields = [
        "stage",
        "method",
        "num_samples",
        "action_horizon",
        "action_dim",
        *METRIC_NAMES,
    ]

    stage_summary_path = (
        args.out_dir
        / "stage_summary_metrics.csv"
    )

    write_csv(
        stage_summary_path,
        stage_rows,
        stage_summary_fields,
    )

    protocol = {
        "metric_source": str(
            args.metric_source.resolve()
        ),
        "metric_function_names": list(
            FUNCTION_NAMES
        ),
        "inference_delay": inference_delay,
        "execute_horizon": execute_horizon,
        "delay_metadata_sources": (
            metadata_sources
        ),
        "previous_action_chunk_source": str(
            (
                args.old_base_root
                / "raw"
                / "prev_action_chunks.npy"
            ).resolve()
        ),
        "fixed_previous_chunks_verified_equal_to_tt": (
            True
        ),
        "num_observations": EXPECTED_NUM_OBS,
        "num_samples_per_observation": (
            EXPECTED_SAMPLES
        ),
        "action_horizon": EXPECTED_HORIZON,
        "action_dim": EXPECTED_ACTION_DIM,
    }

    protocol_path = (
        args.out_dir
        / "metric_protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
        )
        + "\n"
    )

    text_lines = [
        "pi0.5 Kinetix exact old-protocol metrics",
        "",
        f"inference_delay={inference_delay}",
        f"execute_horizon={execute_horizon}",
        (
            "num_observations="
            f"{EXPECTED_NUM_OBS}"
        ),
        (
            "samples_per_observation="
            f"{EXPECTED_SAMPLES}"
        ),
        "",
        "[overall]",
    ]

    for metric in METRIC_NAMES:
        text_lines.append(
            f"{metric}="
            f"{float(overall_row[metric]):.8f}"
        )

    for stage_row in stage_rows:
        stage = str(stage_row["stage"])

        text_lines.extend(
            [
                "",
                f"[{stage}]",
            ]
        )

        for metric in METRIC_NAMES:
            text_lines.append(
                f"{metric}="
                f"{float(stage_row[metric]):.8f}"
            )

    text_summary_path = (
        args.out_dir
        / "metrics_summary.txt"
    )

    text_summary_path.write_text(
        "\n".join(text_lines)
        + "\n"
    )

    print("", flush=True)
    print("===== OVERALL =====", flush=True)

    for metric in METRIC_NAMES:
        print(
            f"{metric}="
            f"{float(overall_row[metric]):.8f}",
            flush=True,
        )

    print("", flush=True)
    print("===== STAGE SUMMARY =====", flush=True)

    for row in stage_rows:
        print(row, flush=True)

    print("", flush=True)
    print("PER_OBS =", per_obs_path, flush=True)
    print("SUMMARY =", summary_path, flush=True)
    print(
        "STAGE_SUMMARY =",
        stage_summary_path,
        flush=True,
    )
    print(
        "TEXT_SUMMARY =",
        text_summary_path,
        flush=True,
    )
    print(
        "PROTOCOL =",
        protocol_path,
        flush=True,
    )

    print(
        "PI05_EXACT_OLD_METRICS_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
