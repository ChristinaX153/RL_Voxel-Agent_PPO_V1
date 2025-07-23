from stable_baselines3 import PPO
from voxel_env import VoxelEnv, export_voxel_grid
from logging_callback import TrainingLoggerCallback 
import os

env = VoxelEnv(grid_size=5)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    gamma=0.95,
    n_steps=512,
    batch_size=64,
    gae_lambda=0.9,
    ent_coef=0.02,
    vf_coef=0.4,
    max_grad_norm=0.5,
    clip_range=0.2,
)

print("🚀 Starting training...")
callback = TrainingLoggerCallback(log_freq=10)
model.learn(total_timesteps=50000, callback=callback)
print("✅ Training completed")

# Post-training evaluation (optional visualization export)
output_folder = "output_steps"
os.makedirs(output_folder, exist_ok=True)

obs, info = env.reset()

print("Exporting voxel states for visualization...")
for step in range(20):
    action_idx, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action_idx)
    done = terminated or truncated

    print(f"Step {step}: Action Index: {action_idx} | Available: {len(env.available_actions)} | Reward: {reward}")
    env.render()

    filename = os.path.join(output_folder, f"step_{step:03}.json")
    export_voxel_grid(env.grid, filename)

    if done:
        obs, info = env.reset()

print(f"Exported voxel states to: {output_folder}")
