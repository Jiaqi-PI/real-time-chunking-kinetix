from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import struct
import flax.nnx as nnx
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import numpy as np

import train_expert


LEVEL_PATH = "worlds/l/grasp_easy.json"
LEVEL_NAME = "worlds_l_grasp_easy"
NUM_ENVS = 128
ACTION_DIM = 6
FRAME_SHAPE = (128, 128, 3)


@struct.dataclass
class StepCarry:
    rng: jax.Array
    obs: jax.Array
    env_state: object
    policy_idxs: jax.Array


@struct.dataclass
class VisualBatch:
    image: jax.Array
    action: jax.Array
    done: jax.Array
    solved: jax.Array


def unwrap_native_state(state):
    while not isinstance(state, kenv_state.EnvState):
        if not hasattr(state, "env_state"):
            raise RuntimeError(
                f"Cannot unwrap env state type={type(state)!r}"
            )
        state = state.env_state
    return state


def load_expert_policies(
    expert_root: Path,
    solve_rate_threshold: float,
    seed: int,
):
    gen = np.random.default_rng(seed)

    seed_dirs = sorted(
        (
            path
            for path in expert_root.glob("seed_*")
            if path.is_dir()
        ),
        key=lambda path: int(path.name.split("_", 1)[1]),
    )

    if len(seed_dirs) != 8:
        raise RuntimeError(
            f"Expected 8 expert seed dirs, got {len(seed_dirs)}"
        )

    selected_state_dicts = []
    selected_records = []

    for seed_dir in seed_dirs:
        log_dirs = sorted(
            (
                path
                for path in seed_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        )

        if len(log_dirs) != 50:
            raise RuntimeError(
                f"{seed_dir}: expected 50 checkpoints, "
                f"got {len(log_dirs)}"
            )

        solve_rates = []

        for log_dir in log_dirs:
            stats_path = (
                log_dir
                / "stats"
                / f"{LEVEL_NAME}.json"
            )

            policy_path = (
                log_dir
                / "policies"
                / f"{LEVEL_NAME}.pkl"
            )

            if not stats_path.is_file():
                raise FileNotFoundError(stats_path)

            if not policy_path.is_file():
                raise FileNotFoundError(policy_path)

            stats = json.loads(stats_path.read_text())

            solve_rates.append(
                float(stats["returned_episode_solved"])
            )

        solve_rates_np = np.asarray(
            solve_rates,
            dtype=np.float64,
        )

        good_idxs = np.flatnonzero(
            solve_rates_np >= solve_rate_threshold
        )

        if good_idxs.size == 0:
            raise RuntimeError(
                f"{seed_dir}: no checkpoint reaches "
                f"{solve_rate_threshold}; "
                f"best={solve_rates_np.max():.6f}"
            )

        chosen_idx = int(gen.choice(good_idxs))
        chosen_dir = log_dirs[chosen_idx]
        chosen_rate = float(solve_rates_np[chosen_idx])

        policy_path = (
            chosen_dir
            / "policies"
            / f"{LEVEL_NAME}.pkl"
        )

        with policy_path.open("rb") as handle:
            selected_state_dicts.append(
                pickle.load(handle)
            )

        selected_records.append(
            {
                "seed_dir": seed_dir.name,
                "checkpoint": int(chosen_dir.name),
                "stats_solve_rate": chosen_rate,
                "policy_path": str(policy_path),
            }
        )

        print(
            "SELECTED_EXPERT "
            f"seed={seed_dir.name} "
            f"checkpoint={chosen_dir.name} "
            f"stats_solve_rate={chosen_rate:.6f}",
            flush=True,
        )

    state_dicts = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *selected_state_dicts,
    )

    return state_dicts, selected_records


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expert-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=262_144,
    )

    parser.add_argument(
        "--batch-steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--solve-rate-threshold",
        type=float,
        default=0.65,
    )

    args = parser.parse_args()

    expert_root = args.expert_root.resolve()
    out_root = args.out_root.resolve()

    tmp_root = out_root.with_name(
        f"{out_root.name}.tmp.{os.getpid()}"
    )

    if not expert_root.is_dir():
        raise FileNotFoundError(expert_root)

    if out_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite {out_root}"
        )

    if tmp_root.exists():
        raise FileExistsError(tmp_root)

    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive")

    if args.batch_steps <= 0:
        raise ValueError("batch_steps must be positive")

    if args.num_steps % NUM_ENVS != 0:
        raise ValueError(
            f"num_steps must be divisible by {NUM_ENVS}"
        )

    steps_per_env = args.num_steps // NUM_ENVS

    if steps_per_env % args.batch_steps != 0:
        raise ValueError(
            "steps_per_env must be divisible by batch_steps"
        )

    print(
        "===== LOAD OFFICIAL EXPERT POLICIES =====",
        flush=True,
    )

    state_dicts, selected_records = load_expert_policies(
        expert_root,
        args.solve_rate_threshold,
        args.seed,
    )

    num_policies = len(selected_records)

    static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS,
        frame_skip=train_expert.FRAME_SKIP,
    )

    env_params = kenv_state.EnvParams()

    levels = train_expert.load_levels(
        [LEVEL_PATH],
        static_env_params,
        env_params,
    )

    level = jax.tree.map(
        lambda x: x[0],
        levels,
    )

    base_env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1",
        static_env_params=static_env_params,
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

    render_static_env_params = (
        static_env_params.replace(
            screen_dim=train_expert.SCREEN_DIM
        )
    )

    render_one = renderer_pixels.make_render_pixels(
        env_params,
        render_static_env_params,
    )

    render_batch = jax.vmap(render_one)

    def new_policy_idxs(
        rng: jax.Array,
    ) -> jax.Array:
        rng, key = jax.random.split(rng)

        return jax.random.randint(
            key,
            (NUM_ENVS,),
            0,
            num_policies,
        )

    @jax.jit
    def init(
        rng: jax.Array,
    ) -> StepCarry:
        rng, key = jax.random.split(rng)

        obs, env_state = env.reset_to_level(
            key,
            level,
            env_params,
        )

        rng, key = jax.random.split(rng)

        policy_idxs = new_policy_idxs(key)

        return StepCarry(
            rng,
            obs,
            env_state,
            policy_idxs,
        )

    @functools.partial(
        jax.jit,
        static_argnums=(1,),
    )
    def step_n(
        carry: StepCarry,
        n: int,
    ):
        def step(
            carry: StepCarry,
            _,
        ):
            action_dim = env.action_space(
                env_params
            ).shape[0]

            obs_dim = carry.obs.shape[1]

            @jax.vmap
            def get_action(
                key,
                obs,
                policy_idx,
            ):
                agent = train_expert.Agent(
                    obs_dim,
                    action_dim,
                    1,
                    rngs=nnx.Rngs(0),
                )

                graphdef, state = nnx.split(agent)

                state.replace_by_pure_dict(
                    jax.tree.map(
                        lambda x: x[policy_idx],
                        state_dicts,
                    )
                )

                agent = nnx.merge(
                    graphdef,
                    state,
                )

                mean, std = agent.action(obs)

                action_dist = (
                    train_expert.make_squashed_normal_diag(
                        mean,
                        std,
                        static_env_params.num_motor_bindings,
                    )
                )

                return action_dist.sample(seed=key)

            rng, action_key = jax.random.split(
                carry.rng
            )

            action = get_action(
                jax.random.split(
                    action_key,
                    NUM_ENVS,
                ),
                carry.obs,
                carry.policy_idxs,
            )

            native_state = unwrap_native_state(
                carry.env_state
            )

            image = (
                render_batch(native_state)
                .round()
                .astype(jnp.uint8)
                .transpose(0, 2, 1, 3)[:, ::-1]
            )

            rng, step_key = jax.random.split(rng)

            (
                next_obs,
                next_env_state,
                _,
                done,
                info,
            ) = env.step(
                step_key,
                carry.env_state,
                action,
                env_params,
            )

            rng, policy_key = jax.random.split(rng)

            next_policy_idxs = jnp.where(
                done,
                new_policy_idxs(policy_key),
                carry.policy_idxs,
            )

            next_carry = StepCarry(
                rng,
                next_obs,
                next_env_state,
                next_policy_idxs,
            )

            result = VisualBatch(
                image=image,
                action=action,
                done=done,
                solved=info[
                    "returned_episode_solved"
                ],
            )

            return next_carry, result

        return jax.lax.scan(
            step,
            carry,
            None,
            length=n,
        )

    print(
        "===== DIRECT EXPERT RGB ROLLOUT =====",
        flush=True,
    )

    print(
        f"JAX_DEVICES={jax.devices()}",
        flush=True,
    )

    print(
        f"NUM_STEPS={args.num_steps}",
        flush=True,
    )

    print(
        f"STEPS_PER_ENV={steps_per_env}",
        flush=True,
    )

    print(
        f"BATCH_STEPS={args.batch_steps}",
        flush=True,
    )

    tmp_root.mkdir(parents=True)

    shard_records = []
    total_done = 0
    total_solved = 0.0
    first_image = None

    try:
        carry = init(
            jax.random.key(args.seed)
        )

        for start in range(
            0,
            steps_per_env,
            args.batch_steps,
        ):
            stop = start + args.batch_steps

            carry, result = step_n(
                carry,
                args.batch_steps,
            )

            result = jax.device_get(result)

            images = np.asarray(
                result.image,
                dtype=np.uint8,
            )

            actions = np.asarray(
                result.action,
                dtype=np.float32,
            )

            done = np.asarray(
                result.done,
                dtype=bool,
            )

            solved = np.asarray(
                result.solved,
                dtype=np.float32,
            )

            expected_images = (
                args.batch_steps,
                NUM_ENVS,
                *FRAME_SHAPE,
            )

            expected_actions = (
                args.batch_steps,
                NUM_ENVS,
                ACTION_DIM,
            )

            expected_done = (
                args.batch_steps,
                NUM_ENVS,
            )

            if images.shape != expected_images:
                raise RuntimeError(
                    f"Bad image shape={images.shape}; "
                    f"expected={expected_images}"
                )

            if images.dtype != np.uint8:
                raise RuntimeError(
                    f"Bad image dtype={images.dtype}"
                )

            if actions.shape != expected_actions:
                raise RuntimeError(
                    f"Bad action shape={actions.shape}; "
                    f"expected={expected_actions}"
                )

            if done.shape != expected_done:
                raise RuntimeError(
                    f"Bad done shape={done.shape}; "
                    f"expected={expected_done}"
                )

            if not np.isfinite(actions).all():
                raise RuntimeError(
                    "Action contains NaN/Inf"
                )

            max_action_abs = float(
                np.max(np.abs(actions))
            )

            if max_action_abs > 1.0001:
                raise RuntimeError(
                    "Expert action exceeds [-1, 1]: "
                    f"max_abs={max_action_abs}"
                )

            if float(images.std()) <= 1.0:
                raise RuntimeError(
                    "Rendered image batch is effectively blank"
                )

            if first_image is None:
                first_image = np.array(
                    images[0, 0],
                    copy=True,
                )

            image_file = (
                f"images_{start:05d}_{stop:05d}.npy"
            )

            action_file = (
                f"actions_{start:05d}_{stop:05d}.npy"
            )

            done_file = (
                f"done_{start:05d}_{stop:05d}.npy"
            )

            np.save(
                tmp_root / image_file,
                images,
                allow_pickle=False,
            )

            np.save(
                tmp_root / action_file,
                actions,
                allow_pickle=False,
            )

            np.save(
                tmp_root / done_file,
                done,
                allow_pickle=False,
            )

            shard_records.append(
                {
                    "start_step": start,
                    "stop_step": stop,
                    "image_file": image_file,
                    "action_file": action_file,
                    "done_file": done_file,
                }
            )

            batch_done = int(done.sum())

            batch_solved = float(
                (solved * done).sum()
            )

            total_done += batch_done
            total_solved += batch_solved

            if (
                start == 0
                or stop % 256 == 0
                or stop == steps_per_env
            ):
                current_rate = (
                    total_solved / total_done
                    if total_done
                    else float("nan")
                )

                print(
                    "SHARD_OK "
                    f"start={start} "
                    f"stop={stop} "
                    f"episodes={total_done} "
                    f"solve_rate={current_rate:.6f}",
                    flush=True,
                )

        if total_done <= 0:
            raise RuntimeError(
                "No completed episodes were generated"
            )

        generated_solve_rate = (
            total_solved / total_done
        )

        if generated_solve_rate < args.solve_rate_threshold:
            raise RuntimeError(
                "Generated expert rollout solve rate "
                f"{generated_solve_rate:.6f} "
                "is below threshold "
                f"{args.solve_rate_threshold:.6f}"
            )

        metadata = {
            "format_version": 1,
            "level_path": LEVEL_PATH,
            "alignment": (
                "image_t is rendered from pre-step carry.env_state; "
                "action_t is sampled from carry.obs in the same scan step"
            ),
            "num_steps": int(args.num_steps),
            "steps_per_env": int(steps_per_env),
            "num_envs": NUM_ENVS,
            "action_dim": ACTION_DIM,
            "frame_shape": list(FRAME_SHAPE),
            "frame_dtype": "uint8",
            "seed": int(args.seed),
            "solve_rate_threshold": float(
                args.solve_rate_threshold
            ),
            "num_completed_episodes": int(total_done),
            "num_solved_episodes": float(total_solved),
            "generated_solve_rate": float(
                generated_solve_rate
            ),
            "selected_experts": selected_records,
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

        np.save(
            tmp_root / "preview_frame.npy",
            first_image,
            allow_pickle=False,
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
        f"NUM_COMPLETED_EPISODES={total_done}",
        flush=True,
    )

    print(
        f"GENERATED_SOLVE_RATE={generated_solve_rate:.8f}",
        flush=True,
    )

    print(
        "DIRECT_PRESTEP_IMAGE_ACTION_ALIGNMENT=True",
        flush=True,
    )

    print(
        "KINETIX_VISUAL_DIRECT_DATA_OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
