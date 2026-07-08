import dataclasses
import json
import pathlib
import pickle

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import numpy as np
import tyro

import model as _model
import train_expert


@dataclasses.dataclass
class CollectConfig:
    run_path: str
    output_dir: str = "rtc_diag_raw_64"

    fixed_observations_path: str | None = None
    fixed_prev_action_chunks_path: str | None = None

    level_path: str = "worlds/l/grasp_easy.json"
    step: int = -1

    num_obs: int = 10
    num_samples: int = 64
    num_flow_steps: int = 5

    inference_delay: int = 2
    execute_horizon: int = 4

    seed: int = 0
    prefix_attention_schedule: str = "exp"
    max_guidance_weight: float = 5.0

    model: _model.ModelConfig = dataclasses.field(default_factory=_model.ModelConfig)


def load_policy(config: CollectConfig, obs_dim: int, action_dim: int, level_name: str):
    log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(config.run_path).iterdir()))
    log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
    if not log_dirs:
        raise FileNotFoundError(f"No numeric checkpoint dirs found under {config.run_path}")

    ckpt_dir = log_dirs[config.step]
    policy_path = ckpt_dir / "policies" / f"{level_name}.pkl"
    print(f"Loading policy from: {policy_path}")

    with policy_path.open("rb") as f:
        state_dict = pickle.load(f)

    state_dict = jax.tree.map(lambda x: jnp.asarray(x), state_dict)

    policy = _model.FlowPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config.model,
        rngs=nnx.Rngs(jax.random.key(config.seed)),
    )

    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    policy = nnx.merge(graphdef, state)

    return policy, str(policy_path)


def main(config: CollectConfig):
    output_dir = pathlib.Path(config.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS,
        frame_skip=train_expert.FRAME_SKIP,
    )
    env_params = kenv_state.EnvParams()

    level_paths = [config.level_path]
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    level = jax.tree.map(lambda x: x[0], levels)

    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    base_env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1",
        static_env_params=static_env_params,
    )

    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(
            wrappers.AutoReplayWrapper(
                train_expert.NoisyActionWrapper(base_env)
            )
        ),
        config.num_obs,
    )

    rng = jax.random.key(config.seed)
    rng, shape_key = jax.random.split(rng)

    obs_shape = jax.eval_shape(env.reset_to_level, shape_key, level, env_params)[0].shape
    obs_dim = obs_shape[-1]
    action_dim = base_env.action_space(env_params).shape[0]

    level_name = config.level_path.replace("/", "_").replace(".json", "")
    policy, policy_path = load_policy(config, obs_dim, action_dim, level_name)

    assert config.execute_horizon >= config.inference_delay
    prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
    assert 0 <= config.inference_delay <= policy.action_chunk_size
    assert 0 <= prefix_attention_horizon <= policy.action_chunk_size

    print("Configuration:")
    print(f"  level_path = {config.level_path}")
    print(f"  num_obs = {config.num_obs}")
    print(f"  num_samples = {config.num_samples}")
    print(f"  inference_delay = {config.inference_delay}")
    print(f"  execute_horizon = {config.execute_horizon}")
    print(f"  prefix_attention_horizon = {prefix_attention_horizon}")
    print(f"  action_chunk_size = {policy.action_chunk_size}")
    print(f"  action_dim = {policy.action_dim}")

    rng, reset_key, prev_key = jax.random.split(rng, 3)

    obs, env_state = env.reset_to_level(reset_key, level, env_params)
    obs_np = np.asarray(jax.device_get(obs))

    # Optional: use fixed observations from a previous collection run.
    # This is needed for fair paired comparison.
    if config.fixed_observations_path is not None:
        obs_np = np.load(config.fixed_observations_path).astype(np.float32)
        assert obs_np.shape[0] == config.num_obs, (
            obs_np.shape,
            config.num_obs,
        )
        obs = jnp.asarray(obs_np)

    # Optional: use fixed previous action chunks from a previous collection run.
    # This is essential because prefix_error, seam_discontinuity, and conditioned inference
    # all depend on prev_action_chunk.
    if config.fixed_prev_action_chunks_path is not None:
        prev_np = np.load(config.fixed_prev_action_chunks_path).astype(np.float32)
        assert prev_np.shape[0] == config.num_obs, (
            prev_np.shape,
            config.num_obs,
        )
        assert prev_np.shape[1] == policy.action_chunk_size, (
            prev_np.shape,
            policy.action_chunk_size,
        )
        prev_action_chunk = jnp.asarray(prev_np)
    else:
        prev_action_chunk = policy.action(prev_key, obs, config.num_flow_steps)
        prev_np = np.asarray(jax.device_get(prev_action_chunk))

    naive_chunks = []
    rtc_chunks = []

    for sample_id in range(config.num_samples):
        rng, naive_key, rtc_key = jax.random.split(rng, 3)

        naive_action = policy.action(
            naive_key,
            obs,
            config.num_flow_steps,
        )

        rtc_action = policy.realtime_action(
            rtc_key,
            obs,
            config.num_flow_steps,
            prev_action_chunk,
            config.inference_delay,
            prefix_attention_horizon,
            config.prefix_attention_schedule,
            config.max_guidance_weight,
        )

        naive_chunks.append(np.asarray(jax.device_get(naive_action)))
        rtc_chunks.append(np.asarray(jax.device_get(rtc_action)))

        if (sample_id + 1) % 8 == 0 or sample_id == config.num_samples - 1:
            print(f"Collected {sample_id + 1}/{config.num_samples} samples")

    naive_chunks = np.stack(naive_chunks, axis=0)
    rtc_chunks = np.stack(rtc_chunks, axis=0)

    # Shapes:
    # obs_np: [num_obs, obs_dim]
    # prev_np: [num_obs, H, action_dim]
    # naive_chunks: [num_samples, num_obs, H, action_dim]
    # rtc_chunks: [num_samples, num_obs, H, action_dim]

    np.save(raw_dir / "observations.npy", obs_np)
    np.save(raw_dir / "prev_action_chunks.npy", prev_np)
    np.save(raw_dir / "naive_actions_all.npy", naive_chunks)
    np.save(raw_dir / "training_free_rtc_actions_all.npy", rtc_chunks)

    metadata = {
        "run_path": config.run_path,
        "policy_path": policy_path,
        "level_path": config.level_path,
        "step": config.step,
        "num_obs": config.num_obs,
        "num_samples": config.num_samples,
        "num_flow_steps": config.num_flow_steps,
        "inference_delay": config.inference_delay,
        "execute_horizon": config.execute_horizon,
        "prefix_attention_horizon": int(prefix_attention_horizon),
        "prefix_attention_schedule": config.prefix_attention_schedule,
        "max_guidance_weight": config.max_guidance_weight,
        "seed": config.seed,
        "action_chunk_size": int(policy.action_chunk_size),
        "action_dim": int(policy.action_dim),
        "array_shapes": {
            "observations": list(obs_np.shape),
            "prev_action_chunks": list(prev_np.shape),
            "naive_actions_all": list(naive_chunks.shape),
            "training_free_rtc_actions_all": list(rtc_chunks.shape),
        },
    }

    with (raw_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # Also save per-observation folders for easy plotting.
    for obs_id in range(config.num_obs):
        obs_dir = raw_dir / f"obs_{obs_id:03d}"
        obs_dir.mkdir(parents=True, exist_ok=True)

        np.save(obs_dir / "observation.npy", obs_np[obs_id])
        np.save(obs_dir / "prev_action_chunk.npy", prev_np[obs_id])
        np.save(obs_dir / "naive_actions.npy", naive_chunks[:, obs_id])
        np.save(obs_dir / "training_free_rtc_actions.npy", rtc_chunks[:, obs_id])

        obs_meta = dict(metadata)
        obs_meta["obs_id"] = obs_id
        obs_meta["array_shapes"] = {
            "observation": list(obs_np[obs_id].shape),
            "prev_action_chunk": list(prev_np[obs_id].shape),
            "naive_actions": list(naive_chunks[:, obs_id].shape),
            "training_free_rtc_actions": list(rtc_chunks[:, obs_id].shape),
        }

        with (obs_dir / "metadata.json").open("w") as f:
            json.dump(obs_meta, f, indent=2)

    print(f"Saved raw data to: {raw_dir}")


if __name__ == "__main__":
    tyro.cli(main)
