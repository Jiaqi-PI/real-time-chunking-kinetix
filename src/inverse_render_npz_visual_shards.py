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
import kinetix.render.renderer_pixels as renderer_pixels
from kinetix.render.renderer_symbolic_flat import (
    make_inverse_render_symbolic,
    make_render_symbolic,
)
import numpy as np

import train_expert


LEVEL_PATH = "worlds/l/grasp_easy.json"
NUM_ENVS = 128
OBS_DIM = 679
ACTION_DIM = 6
FRAME_SHAPE = (128, 128, 3)


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
        "--roundtrip-atol",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--roundtrip-rtol",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

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

    print(
        "===== LOAD NPZ =====",
        flush=True,
    )

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:
        required = (
            "obs",
            "action",
            "done",
        )

        for key in required:
            if key not in data.files:
                raise KeyError(
                    f"Missing key {key!r}"
                )

        if args.max_steps > data["obs"].shape[0]:
            raise ValueError(
                f"max_steps={args.max_steps} "
                f"exceeds NPZ length="
                f"{data['obs'].shape[0]}"
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

    expected_obs_shape = (
        args.max_steps,
        NUM_ENVS,
        OBS_DIM,
    )

    expected_action_shape = (
        args.max_steps,
        NUM_ENVS,
        ACTION_DIM,
    )

    expected_done_shape = (
        args.max_steps,
        NUM_ENVS,
    )

    if obs_ref.shape != expected_obs_shape:
        raise RuntimeError(
            f"obs shape={obs_ref.shape} "
            f"expected={expected_obs_shape}"
        )

    if action_ref.shape != expected_action_shape:
        raise RuntimeError(
            f"action shape={action_ref.shape} "
            f"expected={expected_action_shape}"
        )

    if done_ref.shape != expected_done_shape:
        raise RuntimeError(
            f"done shape={done_ref.shape} "
            f"expected={expected_done_shape}"
        )

    if not np.isfinite(obs_ref).all():
        raise RuntimeError(
            "obs contains NaN/Inf"
        )

    if not np.isfinite(action_ref).all():
        raise RuntimeError(
            "action contains NaN/Inf"
        )

    print(
        f"OBS_SHAPE={obs_ref.shape}",
        flush=True,
    )

    print(
        f"ACTION_SHAPE={action_ref.shape}",
        flush=True,
    )

    print(
        f"DONE_SHAPE={done_ref.shape}",
        flush=True,
    )

    print(
        "OBS_CLIPPED_VALUE_FRACTION="
        f"{float(np.mean(np.abs(obs_ref) >= 9.99999)):.10f}",
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

    dummy_obs, dummy_state = (
        base_env.reset_to_level(
            jax.random.key(0),
            level,
            env_params,
        )
    )

    if tuple(dummy_obs.shape) != (OBS_DIM,):
        raise RuntimeError(
            f"Unexpected dummy obs shape="
            f"{dummy_obs.shape}"
        )

    inverse_one = (
        make_inverse_render_symbolic(
            dummy_state,
            env_params,
            static_params,
        )
    )

    symbolic_one = make_render_symbolic(
        env_params,
        static_params,
        padded=False,
        clip=True,
    )

    render_params = static_params.replace(
        screen_dim=train_expert.SCREEN_DIM
    )

    render_one = (
        renderer_pixels.make_render_pixels(
            env_params,
            render_params,
        )
    )

    inverse_batch = jax.jit(
        jax.vmap(inverse_one)
    )

    symbolic_batch = jax.jit(
        jax.vmap(symbolic_one)
    )

    render_batch = jax.jit(
        jax.vmap(render_one)
    )

    expected_frame_batch_shape = (
        NUM_ENVS,
        *FRAME_SHAPE,
    )

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

    max_roundtrip_abs_error = 0.0

    tmp_root.mkdir(
        parents=True
    )

    try:
        print(
            "===== SYMBOLIC INVERSE RENDER =====",
            flush=True,
        )

        print(
            f"JAX_DEVICES={jax.devices()}",
            flush=True,
        )

        for step in range(args.max_steps):
            obs_device = jnp.asarray(
                obs_ref[step],
                dtype=jnp.float32,
            )

            reconstructed_states = (
                inverse_batch(obs_device)
            )

            roundtrip_obs = np.asarray(
                jax.device_get(
                    symbolic_batch(
                        reconstructed_states
                    )
                ),
                dtype=np.float32,
            )

            abs_error = np.abs(
                roundtrip_obs
                - obs_ref[step]
            )

            step_max_error = float(
                abs_error.max()
            )

            max_roundtrip_abs_error = max(
                max_roundtrip_abs_error,
                step_max_error,
            )

            if not np.allclose(
                roundtrip_obs,
                obs_ref[step],
                atol=args.roundtrip_atol,
                rtol=args.roundtrip_rtol,
            ):
                index = np.unravel_index(
                    int(np.argmax(abs_error)),
                    abs_error.shape,
                )

                raise RuntimeError(
                    "SYMBOLIC_ROUNDTRIP_MISMATCH "
                    f"step={step} "
                    f"index={index} "
                    f"roundtrip="
                    f"{roundtrip_obs[index]} "
                    f"reference="
                    f"{obs_ref[step][index]} "
                    f"max_abs_error="
                    f"{step_max_error}"
                )

            frames = (
                render_batch(
                    reconstructed_states
                )
                .round()
                .astype(jnp.uint8)
                .transpose(0, 2, 1, 3)[:, ::-1]
            )

            frames_np = np.asarray(
                jax.device_get(frames)
            )

            if (
                frames_np.shape
                != expected_frame_batch_shape
            ):
                raise RuntimeError(
                    "Unexpected frame batch shape: "
                    f"{frames_np.shape}"
                )

            if frames_np.dtype != np.uint8:
                raise RuntimeError(
                    "Unexpected frame dtype: "
                    f"{frames_np.dtype}"
                )

            buffer[buffer_count] = frames_np
            buffer_count += 1

            should_flush = (
                buffer_count == args.shard_steps
                or step + 1 == args.max_steps
            )

            if should_flush:
                shard_stop = step + 1

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
                    "max_roundtrip_abs_error="
                    f"{max_roundtrip_abs_error:.9g}",
                    flush=True,
                )

                shard_start = shard_stop
                buffer_count = 0

        metadata = {
            "format_version": 2,
            "source_npz": str(npz_path),
            "source_alignment": (
                "frame[t,env] = pixel_render("
                "inverse_symbolic(obs[t,env]))"
            ),
            "symbolic_roundtrip_verified": True,
            "direct_index_alignment": True,
            "max_symbolic_roundtrip_abs_error": (
                float(max_roundtrip_abs_error)
            ),
            "roundtrip_atol": float(
                args.roundtrip_atol
            ),
            "roundtrip_rtol": float(
                args.roundtrip_rtol
            ),
            "level_path": LEVEL_PATH,
            "max_steps": int(args.max_steps),
            "num_envs": NUM_ENVS,
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "frame_shape": list(FRAME_SHAPE),
            "frame_dtype": "uint8",
            "history_frames": 3,
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
        "MAX_SYMBOLIC_ROUNDTRIP_ABS_ERROR="
        f"{max_roundtrip_abs_error:.9g}",
        flush=True,
    )

    print(
        "SYMBOLIC_ROUNDTRIP_VERIFIED=True",
        flush=True,
    )

    print(
        "DIRECT_INDEX_ALIGNMENT=True",
        flush=True,
    )

    print(
        "KINETIX_VISUAL_INVERSE_RENDER_SHARDS_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
