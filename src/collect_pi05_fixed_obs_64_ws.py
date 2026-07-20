from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi_client import websocket_client_policy


STAGES = ("early", "middle", "late")

EXPECTED_OBS_DIM = 679
EXPECTED_OBS_PER_STAGE = 10
EXPECTED_HORIZON = 8
EXPECTED_ACTION_DIM = 6

DEFAULT_STAGE_ROOT = Path(
    "/cluster/nvme6/pijiaqi/"
    "real-time-chunking-kinetix/"
    "rtc_stage_inputs_grasp_easy"
)

PROMPT = "solve the Kinetix grasp easy task"


def load_stage_inputs(
    stage_root: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    raw_dir = stage_root / "raw"

    observations_path = raw_dir / "observations.npy"
    labels_path = raw_dir / "stage_labels.npy"
    metadata_path = raw_dir / "metadata.json"

    print("===== LOAD EXACT STAGE SCHEMA =====", flush=True)
    print(
        "observations_path:",
        observations_path,
        flush=True,
    )
    print(
        "labels_path:",
        labels_path,
        flush=True,
    )
    print(
        "metadata_path:",
        metadata_path,
        flush=True,
    )

    if not observations_path.is_file():
        raise FileNotFoundError(observations_path)

    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    observations = np.asarray(
        np.load(
            observations_path,
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    labels = np.asarray(
        np.load(
            labels_path,
            allow_pickle=False,
        )
    )

    metadata = json.loads(
        metadata_path.read_text()
    )

    print(
        "observations.shape:",
        observations.shape,
        flush=True,
    )
    print(
        "observations.dtype:",
        observations.dtype,
        flush=True,
    )
    print(
        "labels.shape:",
        labels.shape,
        flush=True,
    )
    print(
        "labels.dtype:",
        labels.dtype,
        flush=True,
    )
    print(
        "unique labels:",
        np.unique(labels),
        flush=True,
    )

    if observations.shape != (
        30,
        EXPECTED_OBS_DIM,
    ):
        raise RuntimeError(
            "Expected observations shape "
            f"(30, {EXPECTED_OBS_DIM}), "
            f"got {observations.shape}"
        )

    if labels.shape != (30,):
        raise RuntimeError(
            f"Expected labels shape (30,), "
            f"got {labels.shape}"
        )

    if not np.isfinite(observations).all():
        raise RuntimeError(
            "Stage observations contain non-finite values"
        )

    actual_labels = set(
        str(label)
        for label in np.unique(labels)
    )

    expected_labels = set(STAGES)

    if actual_labels != expected_labels:
        raise RuntimeError(
            f"Expected labels {expected_labels}, "
            f"got {actual_labels}"
        )

    stage_states: dict[str, np.ndarray] = {}

    for stage in STAGES:
        mask = labels == stage

        states = np.asarray(
            observations[mask],
            dtype=np.float32,
        )

        print(
            f"stage={stage} "
            f"count={states.shape[0]} "
            f"shape={states.shape}",
            flush=True,
        )

        if states.shape != (
            EXPECTED_OBS_PER_STAGE,
            EXPECTED_OBS_DIM,
        ):
            raise RuntimeError(
                f"Stage {stage!r}: expected "
                f"({EXPECTED_OBS_PER_STAGE}, "
                f"{EXPECTED_OBS_DIM}), "
                f"got {states.shape}"
            )

        stage_states[stage] = states

    metadata_stage_counts = metadata.get(
        "stage_counts",
        {}
    )

    for stage in STAGES:
        metadata_count = int(
            metadata_stage_counts.get(
                stage,
                -1,
            )
        )

        if metadata_count != EXPECTED_OBS_PER_STAGE:
            raise RuntimeError(
                f"metadata stage count for {stage}: "
                f"expected {EXPECTED_OBS_PER_STAGE}, "
                f"got {metadata_count}"
            )

    print(
        "EXACT_STAGE_SCHEMA_LOAD_OK",
        flush=True,
    )

    return stage_states, metadata


def build_observation(
    state: np.ndarray,
) -> dict[str, Any]:
    if state.shape != (EXPECTED_OBS_DIM,):
        raise ValueError(
            f"Expected state shape "
            f"({EXPECTED_OBS_DIM},), "
            f"got {state.shape}"
        )

    return {
        "observation/state": np.asarray(
            state,
            dtype=np.float32,
        ),
        "observation/image": np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        ),
        "observation/wrist_image": np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        ),
        "prompt": PROMPT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=18031,
    )

    parser.add_argument(
        "--stage-root",
        type=Path,
        default=DEFAULT_STAGE_ROOT,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError(
            f"num_samples must be positive: "
            f"{args.num_samples}"
        )

    print("===== CONFIG =====", flush=True)
    print("host:", args.host, flush=True)
    print("port:", args.port, flush=True)
    print(
        "stage_root:",
        args.stage_root,
        flush=True,
    )
    print(
        "num_samples:",
        args.num_samples,
        flush=True,
    )
    print(
        "out_dir:",
        args.out_dir,
        flush=True,
    )

    stage_states, source_metadata = (
        load_stage_inputs(
            args.stage_root
        )
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_metadata = {
        "stage_root": str(
            args.stage_root.resolve()
        ),
        "stages": list(STAGES),
        "num_observations_per_stage": (
            EXPECTED_OBS_PER_STAGE
        ),
        "num_samples_per_observation": (
            args.num_samples
        ),
        "obs_dim": EXPECTED_OBS_DIM,
        "action_horizon": EXPECTED_HORIZON,
        "action_dim": EXPECTED_ACTION_DIM,
        "prompt": PROMPT,
        "source_metadata": source_metadata,
    }

    metadata_path = (
        args.out_dir
        / "collection_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            output_metadata,
            indent=2,
        )
        + "\n"
    )

    print(
        "collection metadata:",
        metadata_path,
        flush=True,
    )

    print("===== CONNECT SERVER =====", flush=True)

    policy = (
        websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
    )

    print(
        "server_metadata:",
        policy.get_server_metadata(),
        flush=True,
    )

    collected_actions: dict[
        str,
        np.ndarray,
    ] = {}

    for stage in STAGES:
        states = stage_states[stage]

        print("", flush=True)
        print(
            f"===== COLLECT STAGE={stage} =====",
            flush=True,
        )

        stage_chunks: list[np.ndarray] = []

        for observation_index, state in enumerate(states):
            print(
                f"--- stage={stage} "
                f"observation="
                f"{observation_index + 1}/"
                f"{EXPECTED_OBS_PER_STAGE}",
                flush=True,
            )

            observation = build_observation(
                state
            )

            sample_chunks: list[
                np.ndarray
            ] = []

            for sample_index in range(
                args.num_samples
            ):
                result = policy.infer(
                    observation
                )

                if "actions" not in result:
                    raise RuntimeError(
                        f"stage={stage} "
                        f"observation="
                        f"{observation_index} "
                        f"sample={sample_index}: "
                        "result has no actions; "
                        f"keys={list(result.keys())}"
                    )

                actions = np.asarray(
                    result["actions"],
                    dtype=np.float32,
                )

                expected_shape = (
                    EXPECTED_HORIZON,
                    EXPECTED_ACTION_DIM,
                )

                if actions.shape != expected_shape:
                    raise RuntimeError(
                        f"stage={stage} "
                        f"observation="
                        f"{observation_index} "
                        f"sample={sample_index}: "
                        f"expected actions "
                        f"{expected_shape}, "
                        f"got {actions.shape}"
                    )

                if not np.isfinite(
                    actions
                ).all():
                    raise RuntimeError(
                        f"stage={stage} "
                        f"observation="
                        f"{observation_index} "
                        f"sample={sample_index}: "
                        "non-finite actions"
                    )

                sample_chunks.append(
                    actions
                )

                if (
                    sample_index == 0
                    or (sample_index + 1) % 8 == 0
                    or (
                        sample_index + 1
                        == args.num_samples
                    )
                ):
                    print(
                        f"stage={stage} "
                        f"obs="
                        f"{observation_index + 1}/"
                        f"{EXPECTED_OBS_PER_STAGE} "
                        f"sample="
                        f"{sample_index + 1}/"
                        f"{args.num_samples}",
                        flush=True,
                    )

            observation_chunks = np.stack(
                sample_chunks,
                axis=0,
            )

            expected_observation_shape = (
                args.num_samples,
                EXPECTED_HORIZON,
                EXPECTED_ACTION_DIM,
            )

            if (
                observation_chunks.shape
                != expected_observation_shape
            ):
                raise RuntimeError(
                    f"stage={stage} "
                    f"observation="
                    f"{observation_index}: "
                    f"expected "
                    f"{expected_observation_shape}, "
                    f"got "
                    f"{observation_chunks.shape}"
                )

            stage_chunks.append(
                observation_chunks
            )

        stage_actions = np.stack(
            stage_chunks,
            axis=0,
        )

        expected_stage_shape = (
            EXPECTED_OBS_PER_STAGE,
            args.num_samples,
            EXPECTED_HORIZON,
            EXPECTED_ACTION_DIM,
        )

        if (
            stage_actions.shape
            != expected_stage_shape
        ):
            raise RuntimeError(
                f"stage={stage}: expected "
                f"{expected_stage_shape}, "
                f"got {stage_actions.shape}"
            )

        collected_actions[stage] = (
            stage_actions
        )

        stage_path = (
            args.out_dir
            / f"{stage}_actions.npz"
        )

        np.savez_compressed(
            stage_path,
            actions=stage_actions,
            observations=states,
        )

        print(
            f"SAVED stage={stage}: "
            f"{stage_path}",
            flush=True,
        )

        print(
            f"stage={stage} "
            f"actions.shape="
            f"{stage_actions.shape}",
            flush=True,
        )

    combined_path = (
        args.out_dir
        / "pi05_fixed_obs_64.npz"
    )

    combined_payload: dict[
        str,
        np.ndarray,
    ] = {}

    for stage in STAGES:
        combined_payload[
            f"{stage}_actions"
        ] = collected_actions[stage]

        combined_payload[
            f"{stage}_states"
        ] = stage_states[stage]

    np.savez_compressed(
        combined_path,
        **combined_payload,
    )

    print("", flush=True)
    print("===== FINAL SHAPES =====", flush=True)

    for stage in STAGES:
        print(
            f"{stage}: "
            f"states="
            f"{stage_states[stage].shape} "
            f"actions="
            f"{collected_actions[stage].shape}",
            flush=True,
        )

    print(
        "combined_path:",
        combined_path,
        flush=True,
    )

    print(
        "PI05_FIXED_OBS_64_COLLECTION_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
