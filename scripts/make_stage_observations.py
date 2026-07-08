from pathlib import Path
import json
import numpy as np


data_path = Path("data/worlds_l_grasp_easy.npz")
out_dir = Path("rtc_stage_inputs_grasp_easy/raw")
out_dir.mkdir(parents=True, exist_ok=True)

num_per_stage = 10
seed = 0

stage_fracs = {
    "early": 0.20,
    "middle": 0.50,
    "late": 0.80,
}

rng = np.random.default_rng(seed)

data = dict(np.load(data_path))
obs = data["obs"]
done = data["done"]

print("obs shape:", obs.shape)
print("done shape:", done.shape)

# Expected shape: [steps, envs, obs_dim]
assert obs.ndim == 3, obs.shape
assert done.ndim == 2, done.shape

num_steps, num_envs, obs_dim = obs.shape

candidates = {stage: [] for stage in stage_fracs}

for env_id in range(num_envs):
    start = 0
    for t in range(num_steps):
        if bool(done[t, env_id]) or t == num_steps - 1:
            end = t
            length = end - start + 1

            # Skip extremely short episodes.
            if length >= 16:
                for stage, frac in stage_fracs.items():
                    idx = start + int(frac * (length - 1))
                    candidates[stage].append((idx, env_id, length))

            start = t + 1

for stage in stage_fracs:
    print(stage, "num candidates:", len(candidates[stage]))
    if len(candidates[stage]) < num_per_stage:
        raise RuntimeError(f"Not enough candidates for {stage}")

selected_obs = []
stage_labels = []
selected_indices = []

for stage in ["early", "middle", "late"]:
    chosen = rng.choice(len(candidates[stage]), size=num_per_stage, replace=False)

    for j in chosen:
        t, env_id, length = candidates[stage][j]
        selected_obs.append(obs[t, env_id])
        stage_labels.append(stage)
        selected_indices.append({
            "stage": stage,
            "t": int(t),
            "env_id": int(env_id),
            "episode_length": int(length),
        })

selected_obs = np.asarray(selected_obs, dtype=np.float32)
stage_labels = np.asarray(stage_labels)

np.save(out_dir / "observations.npy", selected_obs)
np.save(out_dir / "stage_labels.npy", stage_labels)

with (out_dir / "metadata.json").open("w") as f:
    json.dump({
        "data_path": str(data_path),
        "num_per_stage": num_per_stage,
        "seed": seed,
        "observations_shape": list(selected_obs.shape),
        "stage_counts": {
            stage: int(np.sum(stage_labels == stage))
            for stage in ["early", "middle", "late"]
        },
        "selected_indices": selected_indices,
    }, f, indent=2)

print("Saved:", out_dir)
print("selected_obs:", selected_obs.shape)
print("stage_labels:", stage_labels.shape)
