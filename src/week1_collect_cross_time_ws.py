from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import websocket_client_policy

from eval_visual_direct_ws import (
    ACTION_DIM,
    ACTION_HORIZON,
    IMAGE_SHAPE,
    INFO_KEYS,
    infer_batch,
    make_environment,
    render_images,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-evals", type=int, default=4)
    parser.add_argument("--level-path", default="worlds/l/grasp_easy.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.num_evals <= 0:
        raise ValueError("num_evals must be positive")

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {out_dir}"
        )
    out_dir.mkdir(parents=True)

    print("===== WEEK1 PART A CROSS-TIME COLLECTION =====", flush=True)
    print(f"num_evals={args.num_evals}", flush=True)
    print(f"seed={args.seed}", flush=True)
    print(f"shard_id={args.shard_id}", flush=True)
    print(f"checkpoint={args.checkpoint_dir}", flush=True)

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

    reset_fn = jax.jit(
        lambda key: env.reset_to_level(key, level, env_params)
    )
    step_fn = jax.jit(
        lambda key, state, action: env.step(
            key, state, action, env_params
        )
    )

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    print(
        f"server_metadata={policy.get_server_metadata()}",
        flush=True,
    )

    rng = jax.random.key(args.seed)
    rng, reset_key = jax.random.split(rng)
    _, env_state = reset_fn(reset_key)

    max_timesteps = int(env_params.max_timesteps)
    n = args.num_evals

    observations = np.zeros(
        (n, max_timesteps, *IMAGE_SHAPE),
        dtype=np.uint8,
    )
    chunks_raw = np.zeros(
        (n, max_timesteps, ACTION_HORIZON, ACTION_DIM),
        dtype=np.float32,
    )
    executed_actions = np.zeros(
        (n, max_timesteps, ACTION_DIM),
        dtype=np.float32,
    )
    valid = np.zeros(
        (n, max_timesteps),
        dtype=bool,
    )

    episode_lengths = np.zeros(n, dtype=np.int32)
    episode_solved = np.zeros(n, dtype=np.float32)
    episode_returns = np.zeros(n, dtype=np.float32)

    active = np.ones(n, dtype=bool)

    raw_min = np.inf
    raw_max = -np.inf
    raw_oob_count = 0
    raw_value_count = 0

    last_t = -1

    for t in range(max_timesteps):
        last_t = t

        # o_t: observation before executing control step t.
        images = render_images(render_batch, env_state)

        # C_t: full raw policy chunk [N, 8, 6].
        chunks = infer_batch(policy, images)

        if chunks.shape != (
            n,
            ACTION_HORIZON,
            ACTION_DIM,
        ):
            raise RuntimeError(
                f"Bad chunk shape={chunks.shape}"
            )
        if not np.isfinite(chunks).all():
            raise RuntimeError(
                f"Non-finite chunk at t={t}"
            )

        observations[active, t] = images[active]
        chunks_raw[active, t] = chunks[active]
        valid[active, t] = True

        raw_min = min(raw_min, float(chunks.min()))
        raw_max = max(raw_max, float(chunks.max()))
        raw_oob_count += int(
            np.count_nonzero(
                (chunks < action_low[None, None, :])
                | (chunks > action_high[None, None, :])
            )
        )
        raw_value_count += int(chunks.size)

        # Receding-horizon execution:
        # replan every control step, execute only C_t[0].
        actions = np.clip(
            chunks[:, 0, :],
            action_low[None, :],
            action_high[None, :],
        ).astype(np.float32)

        executed_actions[active, t] = actions[active]

        rng, step_key = jax.random.split(rng)
        _, env_state, _, done, info = step_fn(
            step_key,
            env_state,
            jnp.asarray(actions, dtype=jnp.float32),
        )

        done_np = np.asarray(
            jax.device_get(done),
            dtype=bool,
        )

        newly_done = active & done_np

        if newly_done.any():
            lengths_now = np.asarray(
                jax.device_get(
                    info["returned_episode_lengths"]
                )
            ).reshape(-1)
            solved_now = np.asarray(
                jax.device_get(
                    info["returned_episode_solved"]
                )
            ).reshape(-1)
            returns_now = np.asarray(
                jax.device_get(
                    info["returned_episode_returns"]
                )
            ).reshape(-1)

            episode_lengths[newly_done] = np.rint(
                lengths_now[newly_done]
            ).astype(np.int32)
            episode_solved[newly_done] = solved_now[
                newly_done
            ].astype(np.float32)
            episode_returns[newly_done] = returns_now[
                newly_done
            ].astype(np.float32)

        active = active & (~done_np)

        if (
            t == 0
            or (t + 1) % 25 == 0
            or not active.any()
        ):
            print(
                f"t={t} "
                f"completed={int((~active).sum())}/{n}",
                flush=True,
            )

        if not active.any():
            break

    if active.any():
        missing = np.flatnonzero(active).tolist()
        raise RuntimeError(
            "Some first episodes did not finish within "
            f"max_timesteps={max_timesteps}: {missing}"
        )

    np.save(
        out_dir / "observations.npy",
        observations,
        allow_pickle=False,
    )
    np.save(
        out_dir / "chunks_raw.npy",
        chunks_raw,
        allow_pickle=False,
    )
    np.save(
        out_dir / "executed_actions.npy",
        executed_actions,
        allow_pickle=False,
    )
    np.save(
        out_dir / "valid.npy",
        valid,
        allow_pickle=False,
    )
    np.save(
        out_dir / "episode_lengths.npy",
        episode_lengths,
        allow_pickle=False,
    )
    np.save(
        out_dir / "episode_solved.npy",
        episode_solved,
        allow_pickle=False,
    )
    np.save(
        out_dir / "episode_returns.npy",
        episode_returns,
        allow_pickle=False,
    )

    metadata = {
        "experiment": "week1_partA_cross_time_alignment",
        "rollout_semantics": (
            "closed_loop_replan_every_step_execute_slot0"
        ),
        "shard_id": int(args.shard_id),
        "seed": int(args.seed),
        "num_evals": int(args.num_evals),
        "max_timesteps": max_timesteps,
        "last_control_step": int(last_t),
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "image_shape": list(IMAGE_SHAPE),
        "level_path": args.level_path,
        "checkpoint_dir": args.checkpoint_dir,
        "raw_chunks_are_unclipped": True,
        "executed_actions_are_clipped": True,
        "raw_action_min": float(raw_min),
        "raw_action_max": float(raw_max),
        "raw_action_out_of_bounds_fraction": float(
            raw_oob_count / raw_value_count
        ),
        "trajectory_id_rule": (
            "trajectory_id = shard_id * num_evals + env_id"
        ),
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
        + "\n"
    )

    print("===== SUMMARY =====", flush=True)
    print(
        f"episode_lengths={episode_lengths.tolist()}",
        flush=True,
    )
    print(
        f"episode_solved={episode_solved.tolist()}",
        flush=True,
    )
    print(f"raw_action_min={raw_min}", flush=True)
    print(f"raw_action_max={raw_max}", flush=True)
    print(f"OUTPUT_DIR={out_dir}", flush=True)
    print("WEEK1_PARTA_COLLECTION_OK", flush=True)


if __name__ == "__main__":
    main()
