from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
from openpi_client import websocket_client_policy

import train_expert


PROMPT = "solve the Kinetix grasp easy task"
ACTION_HORIZON = 8
ACTION_DIM = 6
IMAGE_SHAPE = (128, 128, 3)
INFO_KEYS = (
    "returned_episode_returns",
    "returned_episode_lengths",
    "returned_episode_solved",
)


def log(message: str) -> None:
    print(message, flush=True)


def make_environment(
    *,
    num_evals: int,
    level_path: str,
):
    physics_static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS,
        frame_skip=train_expert.FRAME_SKIP,
    )
    env_params = kenv_state.EnvParams()

    levels = train_expert.load_levels(
        [level_path],
        physics_static_env_params,
        env_params,
    )
    level = jax.tree.map(lambda value: value[0], levels)

    base_env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1",
        static_env_params=physics_static_env_params,
    )
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(
            wrappers.AutoReplayWrapper(
                train_expert.NoisyActionWrapper(base_env)
            )
        ),
        num_evals,
    )

    render_static_env_params = physics_static_env_params.replace(
        screen_dim=train_expert.SCREEN_DIM
    )
    render_pixels = renderer_pixels.make_render_pixels(
        env_params,
        render_static_env_params,
    )
    render_batch = jax.jit(train_expert.make_render_video(render_pixels))

    return base_env, env, env_params, level, render_batch


def infer_batch(
    policy: websocket_client_policy.WebsocketClientPolicy,
    images: np.ndarray,
) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != IMAGE_SHAPE:
        raise RuntimeError(
            f"Expected batched images [N,{IMAGE_SHAPE}], got {images.shape}"
        )
    if images.dtype != np.uint8:
        raise RuntimeError(f"Expected uint8 images, got {images.dtype}")

    chunks = []
    for image in images:
        result = policy.infer(
            {
                "observation/image": image,
                "prompt": PROMPT,
            }
        )
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise RuntimeError(
                f"Expected action shape={(ACTION_HORIZON, ACTION_DIM)}, "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("Policy returned NaN or Inf")
        chunks.append(actions)

    return np.stack(chunks, axis=0)


def render_images(render_batch, env_state) -> np.ndarray:
    images = np.asarray(jax.device_get(render_batch(env_state)), dtype=np.uint8)
    if images.ndim != 4 or images.shape[1:] != IMAGE_SHAPE:
        raise RuntimeError(
            f"Renderer returned shape={images.shape}, expected [N,{IMAGE_SHAPE}]"
        )
    return images


def select_first_episode_values(
    done_history: list[np.ndarray],
    info_history: dict[str, list[np.ndarray]],
    *,
    num_evals: int,
) -> dict[str, np.ndarray]:
    dones = np.stack(done_history, axis=0).astype(bool)
    if dones.ndim != 2 or dones.shape[1] != num_evals:
        raise RuntimeError(f"Bad done history shape={dones.shape}")

    completed = dones.any(axis=0)
    if not completed.all():
        missing = np.flatnonzero(~completed).tolist()
        raise RuntimeError(
            f"Some rollouts never completed their first episode: {missing}"
        )

    first_done_idx = np.argmax(dones, axis=0)
    env_ids = np.arange(num_evals)
    selected: dict[str, np.ndarray] = {}

    for key in INFO_KEYS:
        if key not in info_history:
            raise RuntimeError(
                f"Missing info key={key}; available={sorted(info_history)}"
            )
        values = np.stack(info_history[key], axis=0)
        if values.shape[:2] != dones.shape:
            raise RuntimeError(
                f"Info shape mismatch for {key}: {values.shape} vs {dones.shape}"
            )
        selected[key] = np.asarray(values[first_done_idx, env_ids]).reshape(-1)

    return selected


def write_episodes_csv(
    path: Path,
    *,
    shard_id: int,
    seed: int,
    selected: dict[str, np.ndarray],
) -> None:
    num_evals = len(selected["returned_episode_solved"])
    fields = [
        "shard_id",
        "episode_in_shard",
        "seed",
        *INFO_KEYS,
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for episode_id in range(num_evals):
            writer.writerow(
                {
                    "shard_id": shard_id,
                    "episode_in_shard": episode_id,
                    "seed": seed,
                    "returned_episode_returns": float(
                        selected["returned_episode_returns"][episode_id]
                    ),
                    "returned_episode_lengths": float(
                        selected["returned_episode_lengths"][episode_id]
                    ),
                    "returned_episode_solved": float(
                        selected["returned_episode_solved"][episode_id]
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-evals", type=int, default=16)
    parser.add_argument("--inference-delay", type=int, required=True)
    parser.add_argument("--execute-horizon", type=int, required=True)
    parser.add_argument("--level-path", default="worlds/l/grasp_easy.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.num_evals <= 0:
        raise ValueError("num_evals must be positive")
    if not 0 <= args.inference_delay <= args.execute_horizon <= ACTION_HORIZON:
        raise ValueError(
            "Require 0 <= inference_delay <= execute_horizon <= 8"
        )

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(
            f"Shard output already exists; refusing to overwrite: {out_dir}"
        )
    out_dir.mkdir(parents=True)

    log("===== KINETIX VISUAL-DIRECT WS EVAL =====")
    log(f"mode={args.mode}")
    log(f"shard_id={args.shard_id}")
    log(f"num_evals={args.num_evals}")
    log(f"inference_delay={args.inference_delay}")
    log(f"execute_horizon={args.execute_horizon}")
    log(f"seed={args.seed}")

    base_env, env, env_params, level, render_batch = make_environment(
        num_evals=args.num_evals,
        level_path=args.level_path,
    )

    action_space = base_env.action_space(env_params)
    action_low = np.broadcast_to(
        np.asarray(action_space.low, dtype=np.float32),
        (ACTION_DIM,),
    ).copy()
    action_high = np.broadcast_to(
        np.asarray(action_space.high, dtype=np.float32),
        (ACTION_DIM,),
    ).copy()
    if not np.isfinite(action_low).all() or not np.isfinite(action_high).all():
        raise RuntimeError(
            f"Non-finite action bounds: low={action_low}, high={action_high}"
        )
    if not np.all(action_low < action_high):
        raise RuntimeError(
            f"Invalid action bounds: low={action_low}, high={action_high}"
        )
    log(f"action_low={action_low}")
    log(f"action_high={action_high}")

    reset_fn = jax.jit(
        lambda key: env.reset_to_level(key, level, env_params)
    )
    step_fn = jax.jit(
        lambda key, state, action: env.step(key, state, action, env_params)
    )

    log(f"Connecting to policy server ws://{args.host}:{args.port}")
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    log(f"server_metadata={policy.get_server_metadata()}")

    rng = jax.random.key(args.seed)
    rng, reset_key = jax.random.split(rng)
    _, env_state = reset_fn(reset_key)

    initial_images = render_images(render_batch, env_state)
    action_chunk = infer_batch(policy, initial_images)
    raw_min = float(action_chunk.min())
    raw_max = float(action_chunk.max())
    raw_oob_count = int(
        np.count_nonzero(
            (action_chunk < action_low[None, None, :])
            | (action_chunk > action_high[None, None, :])
        )
    )
    raw_value_count = int(action_chunk.size)
    action_chunk = np.clip(
        action_chunk,
        action_low[None, None, :],
        action_high[None, None, :],
    ).astype(np.float32)

    max_timesteps = int(env_params.max_timesteps)
    scan_length = int(math.ceil(max_timesteps / args.execute_horizon))
    done_history: list[np.ndarray] = []
    info_history: dict[str, list[np.ndarray]] = {key: [] for key in INFO_KEYS}
    total_model_calls = args.num_evals
    start_time = time.time()

    for chunk_index in range(scan_length):
        current_images = render_images(render_batch, env_state)
        next_action_chunk_raw = infer_batch(policy, current_images)
        raw_min = min(raw_min, float(next_action_chunk_raw.min()))
        raw_max = max(raw_max, float(next_action_chunk_raw.max()))
        raw_oob_count += int(
            np.count_nonzero(
                (next_action_chunk_raw < action_low[None, None, :])
                | (next_action_chunk_raw > action_high[None, None, :])
            )
        )
        raw_value_count += int(next_action_chunk_raw.size)
        total_model_calls += args.num_evals

        next_action_chunk = np.clip(
            next_action_chunk_raw,
            action_low[None, None, :],
            action_high[None, None, :],
        ).astype(np.float32)

        action_chunk_to_execute = np.concatenate(
            [
                action_chunk[:, : args.inference_delay],
                next_action_chunk[
                    :,
                    args.inference_delay : args.execute_horizon,
                ],
            ],
            axis=1,
        )
        if action_chunk_to_execute.shape != (
            args.num_evals,
            args.execute_horizon,
            ACTION_DIM,
        ):
            raise RuntimeError(
                f"Bad executable chunk shape={action_chunk_to_execute.shape}"
            )

        action_chunk = np.concatenate(
            [
                next_action_chunk[:, args.execute_horizon :],
                np.zeros(
                    (
                        args.num_evals,
                        args.execute_horizon,
                        ACTION_DIM,
                    ),
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )

        for within_chunk in range(args.execute_horizon):
            timestep = chunk_index * args.execute_horizon + within_chunk
            if timestep >= max_timesteps:
                break

            action = jnp.asarray(
                action_chunk_to_execute[:, within_chunk],
                dtype=jnp.float32,
            )
            rng, step_key = jax.random.split(rng)
            _, env_state, _, done, info = step_fn(
                step_key,
                env_state,
                action,
            )

            done_history.append(
                np.asarray(jax.device_get(done), dtype=bool)
            )
            for key in INFO_KEYS:
                if key not in info:
                    raise RuntimeError(
                        f"Missing {key} from environment info: {sorted(info)}"
                    )
                info_history[key].append(
                    np.asarray(jax.device_get(info[key]))
                )

        if (
            chunk_index == 0
            or (chunk_index + 1) % 4 == 0
            or chunk_index + 1 == scan_length
        ):
            done_count = int(np.stack(done_history).any(axis=0).sum())
            log(
                f"chunk={chunk_index + 1}/{scan_length} "
                f"first_episodes_completed={done_count}/{args.num_evals}"
            )

    elapsed = time.time() - start_time
    selected = select_first_episode_values(
        done_history,
        info_history,
        num_evals=args.num_evals,
    )

    solved = np.asarray(
        selected["returned_episode_solved"],
        dtype=np.float32,
    )
    returns = np.asarray(
        selected["returned_episode_returns"],
        dtype=np.float32,
    )
    lengths = np.asarray(
        selected["returned_episode_lengths"],
        dtype=np.float32,
    )

    write_episodes_csv(
        out_dir / "episodes.csv",
        shard_id=args.shard_id,
        seed=args.seed,
        selected=selected,
    )

    summary = {
        "mode": args.mode,
        "shard_id": int(args.shard_id),
        "seed": int(args.seed),
        "num_evals": int(args.num_evals),
        "num_solved": int(np.rint(solved).sum()),
        "success_rate": float(solved.mean()),
        "mean_return": float(returns.mean()),
        "mean_episode_length": float(lengths.mean()),
        "inference_delay": int(args.inference_delay),
        "execute_horizon": int(args.execute_horizon),
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "level_path": args.level_path,
        "checkpoint_dir": args.checkpoint_dir,
        "total_model_calls": int(total_model_calls),
        "elapsed_sec": float(elapsed),
        "raw_action_min": raw_min,
        "raw_action_max": raw_max,
        "raw_action_out_of_bounds_fraction": float(
            raw_oob_count / raw_value_count
        ),
        "success_selection": "first_done_index",
        "rollout_semantics": "eval_flow_naive_host_side",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    log("===== SHARD SUMMARY =====")
    log(json.dumps(summary, indent=2, sort_keys=True))
    log(f"WROTE={out_dir / 'episodes.csv'}")
    log(f"WROTE={out_dir / 'summary.json'}")
    log("KINETIX_VISUAL_DIRECT_WS_EVAL_OK")


if __name__ == "__main__":
    main()
