import dataclasses
import json
import pathlib
import pickle
import time

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import numpy as np
import optax
import tyro

import model as _model
import train_expert


@dataclasses.dataclass
class Config:
    # Data root. This script reads: <run_path>/data/<level_name>.npz
    run_path: str = "."
    level_path: str = "worlds/l/grasp_easy.json"

    # Initialization checkpoint. This should contain policies/<level_name>.pkl
    load_dir: str = "bc/24"

    # Output root.
    output_root: str = "logs-bc"
    run_name: str = "tt_rtc_single_nojit"

    # Training.
    num_epochs: int = 1
    batch_size: int = 512
    seed: int = 0

    learning_rate: float = 3e-4
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000

    # For quick smoke test. Use e.g. 20.
    # Use None or <=0 for full epoch.
    max_batches_per_epoch: int | None = 20

    # Training-Time RTC config.
    simulated_delay: int = 5
    action_chunk_size: int = 8


def level_name_from_path(level_path: str) -> str:
    return level_path.replace("/", "_").replace(".json", "")


def main(config: Config):
    rng_np = np.random.default_rng(config.seed)
    rng = jax.random.key(config.seed)

    level_name = level_name_from_path(config.level_path)
    data_path = pathlib.Path(config.run_path) / "data" / f"{level_name}.npz"
    print(f"Loading data: {data_path}")

    raw = dict(np.load(data_path))
    print("Raw data keys:", sorted(raw.keys()))

    # Original data shape is usually [steps, envs, ...].
    # Flatten as (env, step) so consecutive indices correspond to time within one env.
    obs = einops.rearrange(raw["obs"], "s e ... -> (e s) ...")
    action = einops.rearrange(raw["action"], "s e ... -> (e s) ...")
    done = einops.rearrange(raw["done"], "s e ... -> (e s) ...")

    obs = np.asarray(obs)
    action = np.asarray(action)
    done = np.asarray(done)

    print("Flattened obs:", obs.shape)
    print("Flattened action:", action.shape)
    print("Flattened done:", done.shape)

    static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS,
        frame_skip=train_expert.FRAME_SKIP,
    )
    env_params = kenv_state.EnvParams()
    env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1",
        static_env_params=static_env_params,
    )

    obs_dim = obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    model_config = _model.ModelConfig(
        action_chunk_size=config.action_chunk_size,
        simulated_delay=config.simulated_delay,
    )

    policy = _model.FlowPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=model_config,
        rngs=nnx.Rngs(rng),
    )

    # Load ordinary BC checkpoint as initialization.
    ckpt_path = pathlib.Path(config.load_dir) / "policies" / f"{level_name}.pkl"
    print(f"Loading init checkpoint: {ckpt_path}")
    with ckpt_path.open("rb") as f:
        state_dict = pickle.load(f)

    graphdef_policy, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    policy = nnx.merge(graphdef_policy, state)

    total_params = sum(x.size for x in jax.tree.leaves(nnx.state(policy, nnx.Param)))
    print(f"Total params: {total_params:,}")

    optimizer = nnx.Optimizer(
        policy,
        optax.chain(
            optax.clip_by_global_norm(config.grad_norm_clip),
            optax.adamw(
                optax.warmup_constant_schedule(
                    0.0,
                    config.learning_rate,
                    config.lr_warmup_steps,
                ),
                weight_decay=config.weight_decay,
            ),
        ),
    )

    graphdef, train_state = nnx.split((policy, optimizer))

    @jax.jit
    def train_step(train_state, rng, obs_batch, action_chunks, done_chunks):
        policy, optimizer = nnx.merge(graphdef, train_state)

        # Zero actions after done, same logic as train_flow.py.
        done_idxs = jnp.where(
            jnp.any(done_chunks, axis=-1),
            jnp.argmax(done_chunks, axis=-1),
            config.action_chunk_size,
        )
        action_chunks = jnp.where(
            jnp.arange(config.action_chunk_size)[None, :, None] >= done_idxs[:, None, None],
            0.0,
            action_chunks,
        )

        def loss_fn(policy):
            # Make scalar loss explicitly.
            return jnp.mean(policy.loss(rng, obs_batch, action_chunks))

        loss, grads = nnx.value_and_grad(loss_fn)(policy)
        optimizer.update(grads)

        _, new_train_state = nnx.split((policy, optimizer))
        grad_norm = optax.global_norm(grads)
        return new_train_state, loss, grad_norm

    valid_steps = obs.shape[0] - config.action_chunk_size + 1
    num_full_batches = valid_steps // config.batch_size
    print(f"valid_steps={valid_steps:,}, num_full_batches={num_full_batches:,}")

    run_name = f"{config.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = pathlib.Path(config.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    print(f"Output run_dir: {run_dir}")

    for epoch in range(config.num_epochs):
        perm = rng_np.permutation(valid_steps)
        perm = perm[: num_full_batches * config.batch_size]
        batches = perm.reshape(num_full_batches, config.batch_size)

        if config.max_batches_per_epoch is not None and config.max_batches_per_epoch > 0:
            batches = batches[: config.max_batches_per_epoch]

        losses = []
        grad_norms = []

        print(f"Epoch {epoch}: {len(batches)} batches")

        for batch_idx, batch_idxs in enumerate(batches):
            rng, step_key = jax.random.split(rng)

            chunk_offsets = np.arange(config.action_chunk_size)[None, :]
            idx = batch_idxs[:, None] + chunk_offsets

            obs_batch = jnp.asarray(obs[batch_idxs])
            action_chunks = jnp.asarray(action[idx])
            done_chunks = jnp.asarray(done[idx])

            train_state, loss, grad_norm = train_step(
                train_state,
                step_key,
                obs_batch,
                action_chunks,
                done_chunks,
            )

            loss_f = float(jax.device_get(loss))
            grad_f = float(jax.device_get(grad_norm))
            losses.append(loss_f)
            grad_norms.append(grad_f)

            if batch_idx % 20 == 0 or batch_idx == len(batches) - 1:
                print(
                    f"epoch={epoch} batch={batch_idx + 1}/{len(batches)} "
                    f"loss={loss_f:.6f} grad_norm={grad_f:.6f}"
                )

        # Save checkpoint each epoch.
        policy_to_save, _ = nnx.merge(graphdef, train_state)
        state_dict_to_save = nnx.state(policy_to_save).to_pure_dict()

        policy_dir = run_dir / str(epoch) / "policies"
        policy_dir.mkdir(parents=True, exist_ok=True)

        with (policy_dir / f"{level_name}.pkl").open("wb") as f:
            pickle.dump(state_dict_to_save, f)

        metrics = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "mean_grad_norm": float(np.mean(grad_norms)),
            "num_batches": int(len(batches)),
        }
        with (run_dir / str(epoch) / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved checkpoint: {policy_dir}")
        print("Epoch metrics:", metrics)

    print(f"Done. Final run_dir: {run_dir}")


if __name__ == "__main__":
    tyro.cli(main)
