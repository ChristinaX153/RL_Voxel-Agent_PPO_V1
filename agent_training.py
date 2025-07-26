from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from voxel_env import VoxelEnv, send_voxel, wait_for_reward
import torch
import os
import time
import subprocess
import requests
import sys

def start_server():
    """Start the combined server in a separate process"""
    try:
        # Start the server process
        server_process = subprocess.Popen([
            sys.executable, 'combined_server.py'
        ], cwd=os.getcwd())
        
        print("Starting combined server...")
        
        # Wait for server to be ready
        server_ready = False
        max_attempts = 30  # 30 seconds timeout
        
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://localhost:5555/emit", timeout=1)
                if response.status_code == 200:
                    server_ready = True
                    print("Combined server is ready!")
                    break
            except:
                time.sleep(1)
        
        if not server_ready:
            print("Warning: Server may not have started properly")
            
        return server_process
        
    except Exception as e:
        print(f"Error starting server: {e}")
        return None

def main():
    # Start the combined server first
    print("Starting combined server...")
    server_process = start_server()
    
    if server_process is None:
        print("Failed to start server. Exiting...")
        return
    
    try:
        # Check for GPU availability (for potential future CNN policies)
        gpu_available = torch.cuda.is_available()
        print(f"GPU available: {gpu_available}")
        print("Using CPU for PPO training (recommended for MLP policies)")

        # Create vectorized environment
        def make_env():
            return VoxelEnv(session=None, grid_size=6, device='cpu')
        # Use vectorized environment for better performance
        num_envs = 1  # Can use more envs on CPU since we're not GPU-limited
        env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv)
        
        # Send initial voxel state - properly extract and convert grid
        initial_grid = env.get_attr('grid')[0]  # Get grid from first environment
        send_voxel(initial_grid.tolist(), step=0)  # Include step number
        print("Sent initial voxel state to server")
        
        # Wait for initial reward acknowledgment
        print('waiting for initial reward for step 0')
        reward_received = wait_for_reward(0, timeout=30)
        if not reward_received:
            print("Warning: No initial reward received, continuing...")
            
        # Configure PPO with CPU (optimal for MLP policies)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            gamma=0.95,
            n_steps=2048,
            batch_size=128,
            gae_lambda=0.9,
            ent_coef=0.1,
            vf_coef=0.4,
            max_grad_norm=0.5,
            clip_range=0.2,
            device='cpu',  # Explicitly use CPU for better MLP performance
            tensorboard_log="./ppo_voxel_tensorboard/"  # Add tensorboard logging
        )

        print(f"Starting training with {num_envs} parallel environments on CPU...")
        model.learn(total_timesteps=2000, progress_bar=True)
        print("Training completed")

        # Save the trained model
        model.save("ppo_voxel_model")

        # Post-training evaluation with single environment
        print("Starting evaluation...")
        eval_env = VoxelEnv(session=None, grid_size=6, device='cpu')  # Use CPU for evaluation too
        output_folder = "output_steps_vec"
        os.makedirs(output_folder, exist_ok=True)

        obs, info = eval_env.reset()
        
        # Initialize step counter for synchronization
        global_step = 0

        for step in range(101):  # Reduced steps for faster evaluation
            action_idx, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action_idx)
            done = terminated or truncated

            print(f"Step {step}: Action Index: {action_idx} | Available: {len(eval_env.available_actions)} | Reward: {reward}")
            if step % 10 == 0:  # Print grid every 10 steps to reduce output
                eval_env.render()

            # Send voxel data to combined server with step number
            print(f'sending voxels for step {global_step}')
            send_voxel(eval_env.grid.tolist(), step=global_step)
            
            # Wait for reward from Grasshopper before continuing
            print(f'waiting for reward for step {global_step}')
            reward_received = wait_for_reward(global_step, timeout=30)
            if not reward_received:
                print(f"Warning: No reward received for step {global_step}, continuing...")
            
            global_step += 1
            time.sleep(0.5)  # Small delay for synchronization

            if done:
                obs, info = eval_env.reset()

        print(f"Exported voxel states to: {output_folder}")
        print(f"Model saved as: ppo_voxel_model.zip")
        print(f"Tensorboard logs saved to: ./ppo_voxel_tensorboard/")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Clean up: terminate the server process
        if server_process:
            print("Shutting down combined server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("Server shutdown complete")

if __name__ == "__main__":
    main()