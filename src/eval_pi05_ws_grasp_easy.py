from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import numpy as np

from openpi_client import websocket_client_policy

import train_expert
from collect_pi05_fixed_obs_64_ws import (
    EXPECTED_ACTION_DIM,
    EXPECTED_HORIZON,
    EXPECTED_OBS_DIM,
    build_observation,
)


PER_EPISODE_FIELDS = (
    "episode_index",
    "solved",
    "returned_episode_solved",
    "returned_episode_return",
    "returned_episode_length",
    "env_steps",
    "inference_calls",
)


def scalar_value(value: Any) -> float:
    array = np.asarray(
        jax.device_get(value)
    )

    if array.size == 0:
        raise RuntimeError(
            "Cannot extract scalar from empty array"
        )

    return float(
        array.reshape(-1)[0]
    )


def bool_value(value: Any) -> bool:
    return bool(
        scalar_value(value)
    )


def make_eval_bundle(
    level_path: str,
):
    print(
        "===== MAKE NATIVE KINETIX ENV =====",
        flush=True,
    )

    static_env_params = (
        kenv_state.StaticEnvParams(
            **train_expert.LARGE_ENV_PARAMS,
            frame_skip=train_expert.FRAME_SKIP,
        )
    )

    env_params = kenv_state.EnvParams()

    level_paths = [
        level_path,
    ]

    print(
        "level_paths =",
        level_paths,
        flush=True,
    )

    levels = train_expert.load_levels(
        level_paths,
        static_env_params,
        env_params,
    )

    level = jax.tree.map(
        lambda x: x[0],
        levels,
    )

    static_env_params = (
        static_env_params.replace(
            screen_dim=train_expert.SCREEN_DIM
        )
    )

    base_env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1",
        static_env_params=static_env_params,
    )

    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(
            wrappers.AutoReplayWrapper(
                train_expert.NoisyActionWrapper(
                    base_env
                )
            )
        ),
        1,
    )

    shape_key = jax.random.key(0)

    obs_shape = jax.eval_shape(
        env.reset_to_level,
        shape_key,
        level,
        env_params,
    )[0].shape

    action_dim = int(
        base_env.action_space(
            env_params
        ).shape[0]
    )

    print(
        "obs_shape =",
        obs_shape,
        flush=True,
    )

    print(
        "action_dim =",
        action_dim,
        flush=True,
    )

    print(
        "max_timesteps =",
        env_params.max_timesteps,
        flush=True,
    )

    if obs_shape != (
        1,
        EXPECTED_OBS_DIM,
    ):
        raise RuntimeError(
            "Expected batched observation shape "
            f"(1, {EXPECTED_OBS_DIM}), "
            f"got {obs_shape}"
        )

    if action_dim != EXPECTED_ACTION_DIM:
        raise RuntimeError(
            f"Expected action_dim "
            f"{EXPECTED_ACTION_DIM}, "
            f"got {action_dim}"
        )

    print(
        "NATIVE_KINETIX_ENV_OK",
        flush=True,
    )

    return (
        base_env,
        env,
        env_params,
        level,
    )


def infer_chunk(
    policy: (
        websocket_client_policy.WebsocketClientPolicy
    ),
    obs: Any,
) -> np.ndarray:
    obs_array = np.asarray(
        jax.device_get(obs),
        dtype=np.float32,
    )

    expected_obs_shape = (
        1,
        EXPECTED_OBS_DIM,
    )

    if obs_array.shape != expected_obs_shape:
        raise RuntimeError(
            f"Expected obs shape "
            f"{expected_obs_shape}, "
            f"got {obs_array.shape}"
        )

    observation = build_observation(
        obs_array[0]
    )

    result = policy.infer(
        observation
    )

    if "actions" not in result:
        raise RuntimeError(
            "Policy result has no actions; "
            f"keys={list(result.keys())}"
        )

    actions = np.asarray(
        result["actions"],
        dtype=np.float32,
    )

    expected_action_shape = (
        EXPECTED_HORIZON,
        EXPECTED_ACTION_DIM,
    )

    if actions.shape != expected_action_shape:
        raise RuntimeError(
            f"Expected action chunk "
            f"{expected_action_shape}, "
            f"got {actions.shape}"
        )

    if not np.isfinite(actions).all():
        raise RuntimeError(
            "Policy returned non-finite actions"
        )

    return actions


def read_existing_rows(
    path: Path,
) -> list[dict[str, str]]:
    if not path.is_file():
        return []

    with path.open(
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    for expected_index, row in enumerate(rows):
        actual_index = int(
            row["episode_index"]
        )

        if actual_index != expected_index:
            raise RuntimeError(
                "Existing episode CSV is not "
                "contiguously indexed: "
                f"expected {expected_index}, "
                f"got {actual_index}"
            )

    return rows


def write_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    config: dict[str, object],
) -> None:
    per_episode_path = (
        output_dir
        / "per_episode.csv"
    )

    tmp_csv_path = (
        output_dir
        / "per_episode.csv.tmp"
    )

    with tmp_csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                PER_EPISODE_FIELDS
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    tmp_csv_path.replace(
        per_episode_path
    )

    solved_values = np.asarray(
        [
            float(row["solved"])
            for row in rows
        ],
        dtype=np.float64,
    )

    length_values = np.asarray(
        [
            float(
                row[
                    "returned_episode_length"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    return_values = np.asarray(
        [
            float(
                row[
                    "returned_episode_return"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    summary = {
        **config,
        "num_evals_completed": len(rows),
        "num_solved": int(
            solved_values.sum()
        ),
        "success_rate": (
            float(solved_values.mean())
            if len(rows) > 0
            else None
        ),
        "mean_returned_episode_length": (
            float(length_values.mean())
            if len(rows) > 0
            else None
        ),
        "mean_returned_episode_return": (
            float(return_values.mean())
            if len(rows) > 0
            else None
        ),
    }

    summary_json_tmp = (
        output_dir
        / "summary.json.tmp"
    )

    summary_json_path = (
        output_dir
        / "summary.json"
    )

    summary_json_tmp.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    summary_json_tmp.replace(
        summary_json_path
    )

    summary_txt_tmp = (
        output_dir
        / "summary.txt.tmp"
    )

    summary_txt_path = (
        output_dir
        / "summary.txt"
    )

    summary_txt_tmp.write_text(
        "\n".join(
            [
                "pi0.5 Kinetix online evaluation",
                (
                    f"level_path="
                    f"{config['level_path']}"
                ),
                (
                    f"inference_delay="
                    f"{config['inference_delay']}"
                ),
                (
                    f"execute_horizon="
                    f"{config['execute_horizon']}"
                ),
                (
                    f"num_evals_requested="
                    f"{config['num_evals_requested']}"
                ),
                (
                    f"num_evals_completed="
                    f"{len(rows)}"
                ),
                (
                    f"num_solved="
                    f"{summary['num_solved']}"
                ),
                (
                    f"success_rate="
                    f"{summary['success_rate']}"
                ),
                (
                    "mean_returned_episode_length="
                    f"{summary['mean_returned_episode_length']}"
                ),
                (
                    "mean_returned_episode_return="
                    f"{summary['mean_returned_episode_return']}"
                ),
                "",
            ]
        )
    )

    summary_txt_tmp.replace(
        summary_txt_path
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--level-path",
        default="worlds/l/grasp_easy.json",
    )

    parser.add_argument(
        "--num-evals",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--inference-delay",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--execute-horizon",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.num_evals <= 0:
        raise ValueError(
            f"num_evals must be positive: "
            f"{args.num_evals}"
        )

    if not (
        0
        <= args.inference_delay
        <= args.execute_horizon
        <= EXPECTED_HORIZON
    ):
        raise ValueError(
            "Require "
            "0 <= inference_delay "
            "<= execute_horizon "
            f"<= {EXPECTED_HORIZON}; "
            f"got delay="
            f"{args.inference_delay}, "
            f"horizon="
            f"{args.execute_horizon}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "level_path": args.level_path,
        "num_evals_requested": args.num_evals,
        "inference_delay": (
            args.inference_delay
        ),
        "execute_horizon": (
            args.execute_horizon
        ),
        "seed": args.seed,
        "action_horizon": EXPECTED_HORIZON,
        "action_dim": EXPECTED_ACTION_DIM,
        "obs_dim": EXPECTED_OBS_DIM,
        "rollout_semantics": (
            "eval_flow_naive_host_side"
        ),
        "clip_actions_to_env_bounds": True,
    }

    config_path = (
        args.output_dir
        / "config.json"
    )

    if config_path.is_file():
        existing_config = json.loads(
            config_path.read_text()
        )

        if existing_config != config:
            raise RuntimeError(
                "Existing output directory has "
                "different configuration.\n"
                f"existing={existing_config}\n"
                f"current={config}"
            )
    else:
        config_path.write_text(
            json.dumps(
                config,
                indent=2,
            )
            + "\n"
        )

    print("===== CONFIG =====", flush=True)

    for key, value in config.items():
        print(
            f"{key}={value}",
            flush=True,
        )

    (
        base_env,
        env,
        env_params,
        level,
    ) = make_eval_bundle(
        args.level_path
    )

    action_space = base_env.action_space(
        env_params
    )

    if not hasattr(action_space, "low"):
        raise RuntimeError(
            "Action space has no low bound: "
            f"{action_space}"
        )

    if not hasattr(action_space, "high"):
        raise RuntimeError(
            "Action space has no high bound: "
            f"{action_space}"
        )

    action_low = np.broadcast_to(
        np.asarray(
            action_space.low,
            dtype=np.float32,
        ),
        (EXPECTED_ACTION_DIM,),
    ).copy()

    action_high = np.broadcast_to(
        np.asarray(
            action_space.high,
            dtype=np.float32,
        ),
        (EXPECTED_ACTION_DIM,),
    ).copy()

    if not np.isfinite(action_low).all():
        raise RuntimeError(
            f"Non-finite action_low: {action_low}"
        )

    if not np.isfinite(action_high).all():
        raise RuntimeError(
            f"Non-finite action_high: {action_high}"
        )

    if not np.all(
        action_low < action_high
    ):
        raise RuntimeError(
            "Invalid action bounds: "
            f"low={action_low} "
            f"high={action_high}"
        )

    print(
        "ACTION_LOW=",
        action_low,
        flush=True,
    )

    print(
        "ACTION_HIGH=",
        action_high,
        flush=True,
    )

    print(
        "ACTION_CLIPPING_ENABLED",
        flush=True,
    )

    print(
        "===== CONNECT POLICY SERVER =====",
        flush=True,
    )

    policy = (
        websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
    )

    print(
        "server_metadata=",
        policy.get_server_metadata(),
        flush=True,
    )

    rows_path = (
        args.output_dir
        / "per_episode.csv"
    )

    existing_rows = read_existing_rows(
        rows_path
    )

    rows: list[dict[str, object]] = [
        dict(row)
        for row in existing_rows
    ]

    start_episode = len(rows)

    print(
        "resume_start_episode=",
        start_episode,
        flush=True,
    )

    if start_episode > args.num_evals:
        raise RuntimeError(
            "Existing completed episode count "
            f"{start_episode} exceeds requested "
            f"{args.num_evals}"
        )

    if start_episode == args.num_evals:
        write_outputs(
            output_dir=args.output_dir,
            rows=rows,
            config=config,
        )

        print(
            "ONLINE_EVAL_ALREADY_COMPLETE",
            flush=True,
        )

        return

    max_timesteps = int(
        env_params.max_timesteps
    )

    max_cycles = math.ceil(
        max_timesteps
        / args.execute_horizon
    )

    print(
        "max_timesteps=",
        max_timesteps,
        flush=True,
    )

    print(
        "max_cycles=",
        max_cycles,
        flush=True,
    )

    print(
        "===== BEGIN ONLINE EVAL =====",
        flush=True,
    )

    for episode_index in range(
        start_episode,
        args.num_evals,
    ):
        episode_rng = jax.random.fold_in(
            jax.random.key(args.seed),
            episode_index,
        )

        episode_rng, reset_key = (
            jax.random.split(
                episode_rng
            )
        )

        obs, env_state = (
            env.reset_to_level(
                reset_key,
                level,
                env_params,
            )
        )

        action_chunk = infer_chunk(
            policy,
            obs,
        )

        inference_calls = 1
        env_steps = 0
        total_reward = 0.0

        done_reached = False
        returned_solved = 0.0
        returned_episode_return = 0.0
        returned_episode_length = 0.0

        for _ in range(max_cycles):
            next_action_chunk = infer_chunk(
                policy,
                obs,
            )

            inference_calls += 1

            action_chunk_to_execute = (
                np.concatenate(
                    [
                        action_chunk[
                            : args.inference_delay
                        ],
                        next_action_chunk[
                            args.inference_delay
                            : args.execute_horizon
                        ],
                    ],
                    axis=0,
                )
            )

            expected_execute_shape = (
                args.execute_horizon,
                EXPECTED_ACTION_DIM,
            )

            if (
                action_chunk_to_execute.shape
                != expected_execute_shape
            ):
                raise RuntimeError(
                    "Unexpected execute chunk shape: "
                    f"{action_chunk_to_execute.shape}"
                )

            shifted_next_action_chunk = (
                np.concatenate(
                    [
                        next_action_chunk[
                            args.execute_horizon :
                        ],
                        np.zeros(
                            (
                                args.execute_horizon,
                                EXPECTED_ACTION_DIM,
                            ),
                            dtype=np.float32,
                        ),
                    ],
                    axis=0,
                )
            )

            if (
                shifted_next_action_chunk.shape
                != (
                    EXPECTED_HORIZON,
                    EXPECTED_ACTION_DIM,
                )
            ):
                raise RuntimeError(
                    "Unexpected shifted chunk shape: "
                    f"{shifted_next_action_chunk.shape}"
                )

            for action in action_chunk_to_execute:
                episode_rng, step_key = (
                    jax.random.split(
                        episode_rng
                    )
                )

                clipped_action = np.clip(
                    np.asarray(
                        action,
                        dtype=np.float32,
                    ),
                    action_low,
                    action_high,
                )

                action_batch = jnp.asarray(
                    clipped_action[None, :],
                    dtype=jnp.float32,
                )

                (
                    next_obs,
                    next_env_state,
                    reward,
                    done,
                    info,
                ) = env.step(
                    step_key,
                    env_state,
                    action_batch,
                    env_params,
                )

                env_steps += 1

                total_reward += scalar_value(
                    reward
                )

                done_now = bool_value(
                    done
                )

                obs = next_obs
                env_state = next_env_state

                if done_now:
                    required_info_keys = (
                        "returned_episode_returns",
                        "returned_episode_lengths",
                        "returned_episode_solved",
                    )

                    missing_keys = [
                        key
                        for key in required_info_keys
                        if key not in info
                    ]

                    if missing_keys:
                        raise RuntimeError(
                            "Done reached but required "
                            "LogWrapper info keys are "
                            f"missing: {missing_keys}; "
                            f"available={list(info.keys())}"
                        )

                    returned_solved = scalar_value(
                        info[
                            "returned_episode_solved"
                        ]
                    )

                    returned_episode_return = (
                        scalar_value(
                            info[
                                "returned_episode_returns"
                            ]
                        )
                    )

                    returned_episode_length = (
                        scalar_value(
                            info[
                                "returned_episode_lengths"
                            ]
                        )
                    )

                    done_reached = True
                    break

            if done_reached:
                break

            action_chunk = (
                shifted_next_action_chunk
            )

        if not done_reached:
            print(
                f"WARNING episode={episode_index} "
                "did not reach done before "
                f"max_timesteps={max_timesteps}",
                flush=True,
            )

            returned_solved = 0.0
            returned_episode_return = (
                total_reward
            )
            returned_episode_length = float(
                env_steps
            )

        solved = int(
            returned_solved >= 0.5
        )

        row = {
            "episode_index": episode_index,
            "solved": solved,
            "returned_episode_solved": (
                returned_solved
            ),
            "returned_episode_return": (
                returned_episode_return
            ),
            "returned_episode_length": (
                returned_episode_length
            ),
            "env_steps": env_steps,
            "inference_calls": inference_calls,
        }

        rows.append(row)

        write_outputs(
            output_dir=args.output_dir,
            rows=rows,
            config=config,
        )

        cumulative_solved = sum(
            int(float(r["solved"]))
            for r in rows
        )

        cumulative_rate = (
            cumulative_solved
            / len(rows)
        )

        print(
            "EPISODE_RESULT "
            f"episode={episode_index} "
            f"solved={solved} "
            f"returned_solved="
            f"{returned_solved:.6f} "
            f"returned_length="
            f"{returned_episode_length:.1f} "
            f"env_steps={env_steps} "
            f"inference_calls={inference_calls} "
            f"cumulative_success_rate="
            f"{cumulative_rate:.6f}",
            flush=True,
        )

    print("", flush=True)
    print(
        "===== FINAL SUMMARY =====",
        flush=True,
    )

    print(
        (
            args.output_dir
            / "summary.txt"
        ).read_text(),
        flush=True,
    )

    print(
        "PI05_KINETIX_ONLINE_EVAL_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
