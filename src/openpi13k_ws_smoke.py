from __future__ import annotations

import argparse

import numpy as np

from openpi_client import websocket_client_policy


DATASET = (
    "/cluster/nvme6/pijiaqi/"
    "real-time-chunking-kinetix/"
    "data/worlds_l_grasp_easy.npz"
)

PROMPT = "solve the Kinetix grasp easy task"

EXPECTED_HORIZON = 8
EXPECTED_ACTION_DIM = 6


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=18030,
    )

    args = parser.parse_args()

    print("===== LOAD DATASET OBS =====", flush=True)

    with np.load(DATASET, allow_pickle=False) as data:
        print("dataset keys:", data.files, flush=True)

        if "obs" not in data.files:
            raise RuntimeError(
                f"Dataset has no obs key: {data.files}"
            )

        obs_array = np.asarray(data["obs"])

        print(
            "obs array shape:",
            obs_array.shape,
            flush=True,
        )

        print(
            "obs array dtype:",
            obs_array.dtype,
            flush=True,
        )

        if obs_array.ndim < 1:
            raise RuntimeError(
                f"Unexpected obs ndim: {obs_array.ndim}"
            )

        obs_dim = obs_array.shape[-1]

        state = np.asarray(
            obs_array.reshape(-1, obs_dim)[0],
            dtype=np.float32,
        )

    print("state.shape:", state.shape, flush=True)
    print("state.dtype:", state.dtype, flush=True)
    print("state finite:", np.isfinite(state).all(), flush=True)

    if not np.isfinite(state).all():
        raise RuntimeError(
            "Input state contains non-finite values"
        )

    observation = {
        "observation/state": state,
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

    print("===== INFER =====", flush=True)

    result = policy.infer(observation)

    print(
        "result keys:",
        list(result.keys()),
        flush=True,
    )

    if "actions" not in result:
        raise RuntimeError(
            f"No actions in result: {list(result.keys())}"
        )

    actions = np.asarray(result["actions"])

    print("actions.shape:", actions.shape, flush=True)
    print("actions.dtype:", actions.dtype, flush=True)
    print(
        "actions finite:",
        np.isfinite(actions).all(),
        flush=True,
    )

    print("actions:", actions, flush=True)

    expected_shape = (
        EXPECTED_HORIZON,
        EXPECTED_ACTION_DIM,
    )

    if actions.shape != expected_shape:
        raise RuntimeError(
            f"Expected {expected_shape}, got {actions.shape}"
        )

    if not np.isfinite(actions).all():
        raise RuntimeError(
            "Policy actions contain non-finite values"
        )

    print(
        "OPENPI13K_WEBSOCKET_SMOKE_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
