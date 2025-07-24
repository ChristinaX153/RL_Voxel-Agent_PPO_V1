import numpy as np
import os
import json
import torch
from gymnasium import Env, spaces


class VoxelEnv(Env):
    def __init__(self, grid_size=5, device=None):
        super(VoxelEnv, self).__init__()
        self.grid_size = grid_size
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)

        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size * grid_size * grid_size,),
            dtype=np.float32  # Changed to float32 for better GPU compatibility
        )

        self.available_actions = []
        self.timeouts = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.grid = np.zeros_like(self.grid, dtype=np.int32)
        center = self.grid_size // 2
        self.grid[center, center, center] = 1
        self._update_available_actions()

        observation = self.grid.flatten().astype(np.float32)  # Changed to float32
        info = {}
        return observation, info

    def step(self, action_idx):
        x, y, z = self.available_actions[action_idx]
        reward = self._calculate_reward(x, y, z)

        self._update_available_actions()

        total_voxels = self.grid_size ** 3
        active_voxels = np.sum(self.grid)

        terminated = bool(
            active_voxels >= 0.6 * total_voxels or len(self.available_actions) == 0
        )
        truncated = False
        info = {}

        observation = self.grid.flatten().astype(np.float32)  # Changed to float32
        self.action_space = spaces.Discrete(max(1, len(self.available_actions)))

        return observation, reward, terminated, truncated, info

    def render(self):
        print(self.grid)

    def _count_neighbors(self, x, y, z):
        # Vectorized neighbor counting
        neighbor_offsets = np.array([
            [1, 0, 0], [-1, 0, 0],   # x-axis neighbors
            [0, 1, 0], [0, -1, 0],   # y-axis neighbors
            [0, 0, 1], [0, 0, -1]    # z-axis neighbors
        ])
        
        # Calculate neighbor coordinates
        neighbor_coords = np.array([x, y, z]) + neighbor_offsets
        
        # Filter coordinates within grid bounds
        valid_mask = (
            (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 0] < self.grid_size) &
            (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < self.grid_size) &
            (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < self.grid_size)
        )
        valid_neighbors = neighbor_coords[valid_mask]
        
        if len(valid_neighbors) == 0:
            return 0
        
        # Count neighbors that have value 1
        neighbor_values = self.grid[valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]]
        return int(np.sum(neighbor_values == 1))

    def _update_available_actions(self):
        # Vectorized approach using NumPy operations
        # Find all occupied voxels (value = 1)
        occupied_coords = np.argwhere(self.grid == 1)
        
        if len(occupied_coords) == 0:
            self.available_actions = []
            return
        
        # Define neighbor offsets for 6-connectivity (face neighbors only)
        neighbor_offsets = np.array([
            [1, 0, 0], [-1, 0, 0],   # x-axis neighbors
            [0, 1, 0], [0, -1, 0],   # y-axis neighbors
            [0, 0, 1], [0, 0, -1]    # z-axis neighbors
        ])
        
        # Generate all potential neighbor positions
        # Shape: (num_occupied, 6, 3) - for each occupied voxel, get 6 neighbors
        potential_neighbors = occupied_coords[:, np.newaxis, :] + neighbor_offsets[np.newaxis, :, :]
        
        # Reshape to (num_occupied * 6, 3) for easier processing
        potential_neighbors = potential_neighbors.reshape(-1, 3)
        
        # Filter out coordinates that are outside grid boundaries
        valid_mask = (
            (potential_neighbors[:, 0] >= 0) & (potential_neighbors[:, 0] < self.grid_size) &
            (potential_neighbors[:, 1] >= 0) & (potential_neighbors[:, 1] < self.grid_size) &
            (potential_neighbors[:, 2] >= 0) & (potential_neighbors[:, 2] < self.grid_size)
        )
        valid_neighbors = potential_neighbors[valid_mask]
        
        if len(valid_neighbors) == 0:
            self.available_actions = []
            return
        
        # Check which of these valid positions are empty (value = 0)
        # Use advanced indexing to check grid values at these positions
        grid_values = self.grid[valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]]
        empty_mask = (grid_values == 0)
        empty_neighbors = valid_neighbors[empty_mask]
        
        # Remove duplicates by converting to set of tuples, then back to list
        unique_positions = set(map(tuple, empty_neighbors))
        self.available_actions = list(unique_positions)

    def _calculate_reward(self, x, y, z):
        if self.grid[x, y, z] == 0:
            self.grid[x, y, z] = 1
            self._write_grid_to_temp_file()
            return self._wait_for_external_reward()
        else:
            return -0.2

    def _write_grid_to_temp_file(self):
        export_voxel_grid(self.grid, "temp_voxel_input.json")

    def _wait_for_external_reward(self, timeout=10):
        import time
        import json
        global timeouts

        reward_file = "temp_reward.json"
        start_time = time.time()

        while True:
            if os.path.exists(reward_file):
                try:
                    with open(reward_file, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            raise ValueError("Empty file")
                        data = json.loads(content)
                    
                    # ✅ Wait until file is unlocked before deleting
                    for _ in range(10):
                        try:
                            os.remove(reward_file)
                            break
                        except PermissionError:
                            time.sleep(0.03)

                    return float(data.get("reward", 0.0))

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⏳ Waiting for valid reward file... ({e})")
                    time.sleep(0.1)

            else:
                if time.time() - start_time > timeout:
                    self.timeouts += 1
                    print("❌ Timeout: No external reward received. {} times".format(self.timeouts))
                    return 0.0
                time.sleep(0.1)

def export_voxel_grid(grid, filename):
    data = {"voxels": grid.tolist()}
    with open(filename, 'w') as f:
        json.dump(data, f)

class VectorizedVoxelEnv:
    """Vectorized wrapper for multiple VoxelEnv instances"""
    def __init__(self, num_envs=8, grid_size=5, device=None):
        self.num_envs = num_envs
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.envs = [VoxelEnv(grid_size=grid_size, device=self.device) for _ in range(num_envs)]
        
        # Use the action/observation space from the first env
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        
    def reset(self, seed=None):
        observations = []
        infos = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        return np.array(observations, dtype=np.float32), infos
    
    def step(self, actions):
        observations = []
        rewards = []
        terminated_list = []
        truncated_list = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            infos.append(info)
            
        return (np.array(observations, dtype=np.float32), 
                np.array(rewards, dtype=np.float32),
                np.array(terminated_list, dtype=bool),
                np.array(truncated_list, dtype=bool),
                infos)
    
    def render(self):
        # Render only the first environment
        self.envs[0].render()
