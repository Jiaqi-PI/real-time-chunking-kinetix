from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import numpy as np

import generate_data
import train_expert


LEVEL_PATH = "worlds/l/grasp_easy.json"
NUM_ENVS = 128
OBS_DIM = 679
ACTION_DIM = 6
FRAME_SHAPE = (128, 128, 3)


def unwrap_native_state(state):
    while not isinstance(state, kenv_state.EnvState):
        if not hasattr(state, "env_state"):
            raise RuntimeError(
                f"Cannot unwrap state type={type(state)!r}"
            )
        state = state.env_state
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--shard-steps",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--obs-atol",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--obs-rtol",
        type=float,
        default=2e-5,
    )
    args = parser.parse_args()

    if args.max_steps <= 0 or args.shard_steps <= 0:
        raise ValueError(
            "max_steps and shard_steps must be positive"
        )

    npz_path = args.npz_path.resolve()
    out_root = args.out_root.resolve()
    tmp_root = out_root.with_name(
        f"{out_root.name}.tmp.{os.getpid()}"
    )

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    if out_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite {out_root}"
        )

    if tmp_root.exists():
        raise FileExistsError(tmp_root)

    cfg = generate_data.Config(
        run_path="UNUSED"
    )

    if cfg.seed != 0 or cfg.num_envs != NUM_ENVS:
        raise RuntimeError(
            "Unexpected generate_data defaults: "
            f"seed={cfg.seed} "
            f"num_envs={cfg.num_envs}"
        )

    if tuple(cfg.level_paths)[0] != LEVEL_PATH:
        raise RuntimeError(
            "Unexpected first level: "
            f"{tuple(cfg.level_paths)[0]}"
        )

    print(
        "===== LOAD NPZ PREFIX =====",
        flush=True,
    )

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:
        for key in (
            "obs",
            "action",
            "done",
        ):
            if key not in data.files:
                raise KeyError(
                    f"Missing NPZ key {key!r}"
                )

        if args.max_steps > data["obs"].shape[0]:
            raise ValueError(
                f"max_steps={args.max_steps} "
                f"exceeds {data['obs'].shape[0]}"
            )

        obs_ref = np.array(
            data["obs"][: args.max_steps],
            dtype=np.float32,
            copy=True,
        )

        action_ref = np.array(
            data["action"][: args.max_steps],
            dtype=np.float32,
            copy=True,
        )

        done_ref = np.array(
            data["done"][: args.max_steps],
            dtype=bool,
            copy=True,
        )

    expected_obs = (
        args.max_steps,
        NUM_ENVS,
        OBS_DIM,
    )

    expected_action = (
        args.max_steps,
        NUM_ENVS,
        ACTION_DIM,
    )

    expected_done = (
        args.max_steps,
        NUM_ENVS,
    )

    if obs_ref.shape != expected_obs:
        raise RuntimeError(
            f"obs shape {obs_ref.shape} "
            f"!= {expected_obs}"
        )

    if action_ref.shape != expected_action:
        raise RuntimeError(
            f"action shape {action_ref.shape} "
            f"!= {expected_action}"
        )

    if done_ref.shape != expected_done:
        raise RuntimeError(
            f"done shape {done_ref.shape} "
            f"!= {expected_done}"
        )

    if (
        not np.isfinite(obs_ref).all()
        or not np.isfinite(action_ref).all()
    ):
        raise RuntimeError(
            "NPZ contains NaN/Inf"
        )

    print(
        f"NPZ_PATH={npz_path}",
        flush=True,
    )
    print(
        f"OBS_REF_SHAPE={obs_ref.shape}",
        flush=True,
    )
    print(
        f"ACTION_REF_SHAPE={action_ref.shape}",
        flush=True,
    )
    print(
        f"DONE_REF_SHAPE={done_ref.shape}",
        flush=True,
    )

    static_params = (
        kenv_state.StaticEnvParams(
            **train_expert.LARGE_ENV_PARAMS,
            frame_skip=train_expert.FRAME_SKIP,
        )
    )

    env_params = kenv_state.EnvParams()

    levels = train_expert.load_levels(
        [LEVEL_PATH],
        static_params,
        env_params,
    )

    level = jax.tree.map(
        lambda x: x[0],
        levels,
    )

    base_env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1",
        static_env_params=static_params,
    )

    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(
            wrappers.AutoReplayWrapper(
                train_expert.ActionHistoryWrapper(
                    train_expert.ObsHistoryWrapper(
                        train_expert.NoisyActionWrapper(
                            base_env
                        ),
                        4,
                    )
                )
            )
        ),
        NUM_ENVS,
    )

    render_params = static_params.replace(
        screen_dim=train_expert.SCREEN_DIM
    )

    render_pixels = (
        renderer_pixels.make_render_pixels(
            env_params,
            render_params,
        )
    )

    render_batch = jax.jit(
        jax.vmap(render_pixels)
    )

    reset_fn = jax.jit(
        lambda key: env.reset_to_level(
            key,
            level,
            env_params,
        )
    )

    step_fn = jax.jit(
        lambda key, state, action: env.step(
            key,
            state,
            action,
            env_params,
        )
    )

    root_rng = jax.random.key(cfg.seed)

    level_rng = jax.random.split(
        root_rng,
        len(cfg.level_paths),
    )[0]

    rng, reset_key = jax.random.split(
        level_rng
    )

    _, env_state = reset_fn(
        reset_key
    )

    # generate_data.init 的 policy-selection split
    rng, _ = jax.random.split(rng)

    buffer = np.empty(
        (
            args.shard_steps,
            NUM_ENVS,
            *FRAME_SHAPE,
        ),
        dtype=np.uint8,
    )

    buffer_count = 0
    shard_start = 0
    shard_records = []
    max_obs_abs_error = 0.0

    tmp_root.mkdir(
        parents=True
    )

    try:
        print(
            "===== EXACT REPLAY + PRE-STEP RENDER =====",
            flush=True,
        )

        print(
            f"JAX_DEVICES={jax.devices()}",
            flush=True,
        )

        for t in range(args.max_steps):
            replay_obs = np.asarray(
                jax.device_get(
                    train_expert
                    .ObsHistoryWrapper
                    .get_original_obs(
                        env_state
                    )
                ),
                dtype=np.float32,
            )

            abs_error = np.abs(
                replay_obs - obs_ref[t]
            )

            step_error = float(
                abs_error.max()
            )

            max_obs_abs_error = max(
                max_obs_abs_error,
                step_error,
            )

            if not np.allclose(
                replay_obs,
                obs_ref[t],
                atol=args.obs_atol,
                rtol=args.obs_rtol,
            ):
                idx = np.unravel_index(
                    int(np.argmax(abs_error)),
                    abs_error.shape,
                )

                raise RuntimeError(
                    "REPLAY_OBS_MISMATCH "
                    f"step={t} "
                    f"index={idx} "
                    f"replay={replay_obs[idx]} "
                    f"reference={obs_ref[t][idx]} "
                    f"max_abs_error={step_error}"
                )

            frames = (
                render_batch(
                    unwrap_native_state(
                        env_state
                    )
                )
                .round()
                .astype(jnp.uint8)
                .transpose(0, 2, 1, 3)[:, ::-1]
            )

            frames_np = np.asarray(
                jax.device_get(frames)
            )

            expected_frames = (
                NUM_ENVS,
                *FRAME_SHAPE,
            )

            if (
                frames_np.shape != expected_frames
                or frames_np.dtype != np.uint8
            ):
                raise RuntimeError(
                    "Bad frame batch: "
                    f"shape={frames_np.shape} "
                    f"dtype={frames_np.dtype}"
                )

            buffer[buffer_count] = frames_np
            buffer_count += 1

            # generate_data.step:
            # action key
            rng, _ = jax.random.split(rng)

            # env step key
            rng, step_key = jax.random.split(rng)

            (
                _,
                next_state,
                _,
                done,
                _,
            ) = step_fn(
                step_key,
                env_state,
                jnp.asarray(
                    action_ref[t],
                    dtype=jnp.float32,
                ),
            )

            done_np = np.asarray(
                jax.device_get(done),
                dtype=bool,
            )

            if not np.array_equal(
                done_np,
                done_ref[t],
            ):
                bad_envs = np.flatnonzero(
                    done_np != done_ref[t]
                )

                raise RuntimeError(
                    "REPLAY_DONE_MISMATCH "
                    f"step={t} "
                    f"env_ids="
                    f"{bad_envs[:20].tolist()}"
                )

            # policy-switch key
            rng, _ = jax.random.split(rng)

            env_state = next_state

            should_flush = (
                buffer_count == args.shard_steps
                or t + 1 == args.max_steps
            )

            if should_flush:
                shard_stop = t + 1

                filename = (
                    f"frames_{shard_start:05d}_"
                    f"{shard_stop:05d}.npy"
                )

                np.save(
                    tmp_root / filename,
                    buffer[:buffer_count],
                    allow_pickle=False,
                )

                shard_records.append(
                    {
                        "file": filename,
                        "start_step": shard_start,
                        "stop_step": shard_stop,
                    }
                )

                print(
                    "SHARD_OK "
                    f"start={shard_start} "
                    f"stop={shard_stop} "
                    f"max_obs_abs_error="
                    f"{max_obs_abs_error:.9g}",
                    flush=True,
                )

                shard_start = shard_stop
                buffer_count = 0

        metadata = {
            "format_version": 1,
            "source_npz": str(npz_path),
            "level_path": LEVEL_PATH,
            "generate_data_seed": int(cfg.seed),
            "generate_data_num_envs": int(
                cfg.num_envs
            ),
            "generate_data_level_count": len(
                cfg.level_paths
            ),
            "max_steps": int(args.max_steps),
            "num_envs": NUM_ENVS,
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "frame_shape": list(FRAME_SHAPE),
            "frame_dtype": "uint8",
            "shard_steps": int(args.shard_steps),
            "renderer_screen_dim": list(
                render_params.screen_dim
            ),
            "renderer_downscale": int(
                render_params.downscale
            ),
            "physics_frame_skip": int(
                static_params.frame_skip
            ),
            "obs_atol": float(args.obs_atol),
            "obs_rtol": float(args.obs_rtol),
            "max_obs_abs_error": float(
                max_obs_abs_error
            ),
            "done_exact_match": True,
            "shards": shard_records,
        }

        (
            tmp_root / "metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            )
        )

        tmp_root.rename(out_root)

    except Exception:
        shutil.rmtree(
            tmp_root,
            ignore_errors=True,
        )
        raise

    print(
        f"OUT_ROOT={out_root}",
        flush=True,
    )
    print(
        f"NUM_SHARDS={len(shard_records)}",
        flush=True,
    )
    print(
        "MAX_OBS_ABS_ERROR="
        f"{max_obs_abs_error:.9g}",
        flush=True,
    )
    print(
        "DONE_EXACT_MATCH=True",
        flush=True,
    )
    print(
        "KINETIX_VISUAL_REPLAY_SHARDS_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
