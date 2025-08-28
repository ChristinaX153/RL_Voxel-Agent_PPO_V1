import os
import time
import argparse
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
# from sb3_contrib.common.maskable.evaluation import evaluate_policy  # optional

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import wrappers

from voxel_env import VoxelEnv, export_voxel_grid

# ---------------------------
# CLI args
# ---------------------------
parser = argparse.ArgumentParser(description='Train MaskablePPO agent for voxel environment')
parser.add_argument('--port', type=int, default=81, help='Base port number for VoxelEnv (default: 81)')
parser.add_argument('--new', action='store_true', default=False, help='Start a new model instead of loading an existing one')
parser.add_argument('--n_envs', type=int, default=1, help='Number of parallel environments (default: 1)')
parser.add_argument('--grid', type=int, default=5, help='Grid size (default: 5)')
parser.add_argument('--chk_freq', type=int, default=100, help='Frequency of saving checkpoints (default: 100)')
parser.add_argument('--paneling_weight', type=float, default=1.0, help='Weight for paneling cost reward')
args = parser.parse_args()

# ---------------------------
# Basic info
# ---------------------------
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")
print("Using CPU for PPO training (recommended for MLP policies)")

# ---------------------------
# Mask function for ActionMasker
# ---------------------------
def mask_fn(env: VoxelEnv):
    # VoxelEnv implements env.action_masks()
    return env.action_masks()

# ---------------------------
# Vec env factory
# ---------------------------
def make_single_env(rank: int = 0):
    def _thunk():
        # give each env a unique port if you scale n_envs
        e = VoxelEnv(
            port=args.port + rank,
            grid_size=args.grid,
            device='cpu',
            paneling_weight=args.paneling_weight  # Pass the new arg
        )
        # wrap with the ActionMasker so the policy only samples valid actions
        return ActionMasker(e, mask_fn)
    return _thunk

num_envs = max(1, args.n_envs)
env = DummyVecEnv([make_single_env(i) for i in range(num_envs)])

# ---------------------------
# PPO hyperparams (stable defaults)
# ---------------------------
num_steps = 256          # rollout length per env 
num_epochs = 2          # PPO epochs per update
batch_size = 64          # must divide (num_steps * num_envs); 1024*k is always divisible by 256

starting_step = 0
model_dir = os.path.join(os.getcwd(), 'models', 'PPO')
os.makedirs(model_dir, exist_ok=True)
existing_models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]

if len(existing_models) > 0 and not args.new:
    latest_model = os.path.splitext(sorted(existing_models, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))[-1])[0]
    print(f"Loading existing MaskablePPO model: {latest_model}")
    model_path = os.path.join(model_dir, latest_model)
    model = MaskablePPO.load(model_path, env=env, device='cpu')
    starting_step = getattr(model, "_total_timesteps", 0)
else:
    print("Creating a new MaskablePPO model")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=num_steps,
        batch_size=batch_size,
        n_epochs=num_epochs,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        policy_kwargs=dict(ortho_init=False),  # helps stability with sparse/heterogeneous rewards
        device='cpu',
        tensorboard_log="./ppo_voxel_tensorboard/"
    )

print(f"Starting training from {starting_step} with {num_envs} parallel env(s) on CPU...")

# Keep the same "one-update" semantics you used before, just larger rollout & batch
new_steps = num_steps * num_epochs
model_date_time = time.strftime("%Y%m%d-%H%M", time.localtime())
save_path = os.path.join(model_dir, f"maskable_ppo_voxel_{model_date_time}_{new_steps+starting_step}")

# ---------------------------
# Checkpointing
# ---------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=args.chk_freq,
    save_path=os.path.join(model_dir, 'checkpoints'),
    name_prefix=f"maskable_ppo_voxel_{model_date_time}_{new_steps+starting_step}"
)

# ---------------------------
# Train
# ---------------------------
model.learn(total_timesteps=new_steps, progress_bar=True, callback=checkpoint_callback)
print("Training completed")

# ---------------------------
# Save the trained model
# ---------------------------
model.save(save_path)
print(f"Model saved as: {save_path}.zip")

# ---------------------------
# Evaluation (mask-aware rollout)
# ---------------------------
print("Starting evaluation...")

max_eval_steps = 3 * num_steps

# Build a single eval env and keep masking on predict-time
raw_eval_env: VoxelEnv = VoxelEnv(port=args.port, grid_size=args.grid, device='cpu')
raw_eval_env = ActionMasker(raw_eval_env, mask_fn)  # ensure masks are applied at sample-time
raw_eval_env = wrappers.TimeLimit(env=raw_eval_env, max_episode_steps=max_eval_steps * 2)
raw_eval_env = Monitor(raw_eval_env, allow_early_resets=True)

output_folder = f"output_steps/model_{model_date_time}"
os.makedirs(output_folder, exist_ok=True)

obs, info = raw_eval_env.reset()
print("Exporting voxel states for visualization...")

for step in range(max_eval_steps):
    # For predict-time, be explicit: recompute mask from the underlying env
    current_mask = raw_eval_env.unwrapped.action_masks()
    action_idx, _states = model.predict(obs, deterministic=True, action_masks=current_mask)

    obs, reward, terminated, truncated, info = raw_eval_env.step(action_idx)
    done = bool(terminated or truncated)

    # decode action for logging clarity (handles NO-OP)
    action_type, (x, y, z) = raw_eval_env.unwrapped._action_to_coords(int(action_idx))
    total_actions = raw_eval_env.action_space.n
    print(f"Step {step}: Action Index: {action_idx} | Action: {action_type.upper()} "
          f"at ({x},{y},{z}) | Total Actions: {total_actions} | Reward: {reward}")

    if step % 10 == 0:
        raw_eval_env.unwrapped.render()

    filename = os.path.join(output_folder, f"step_{step:03}.json")
    export_voxel_grid(raw_eval_env.unwrapped.grid, filename)

    if done:
        obs, info = raw_eval_env.reset()

print(f"Exported voxel states to: {output_folder}")
print(f"Tensorboard logs: ./ppo_voxel_tensorboard/")
print(f"Action space (masked at runtime, includes NO-OP): {env.action_space}")
