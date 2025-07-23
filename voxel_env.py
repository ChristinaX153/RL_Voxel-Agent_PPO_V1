import numpy as np
import os
import time
import json
from gymnasium import Env, spaces


class VoxelEnv(Env):
    def __init__(self, grid_size=5):
        super(VoxelEnv, self).__init__()
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)

        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size * grid_size * grid_size,),
            dtype=np.int32
        )

        self.available_actions = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = self._np_random, None  # <- this sets up self.np_random

        self.grid = np.zeros_like(self.grid, dtype=np.int32)

        x = self.np_random.integers(0, self.grid_size)
        y = self.np_random.integers(0, self.grid_size)
        z = self.np_random.integers(0, self.grid_size)
        self.grid[x, y, z] = 1

        self._update_available_actions()
        return self.grid.flatten().astype(np.int32), {}


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

        observation = self.grid.flatten().astype(np.int32)
        self.action_space = spaces.Discrete(max(1, len(self.available_actions)))

        return observation, reward, terminated, truncated, info

    def render(self):
        print(self.grid)

    def _count_neighbors(self, x, y, z):
        count = 0
        neighbors = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        for nx, ny, nz in neighbors:
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 0 <= nz < self.grid_size:
                if self.grid[nx, ny, nz] == 1:
                    count += 1
        return count

    def _update_available_actions(self):
        possible = set()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    if self.grid[x, y, z] == 1:
                        neighbors = [
                            (x + 1, y, z), (x - 1, y, z),
                            (x, y + 1, z), (x, y - 1, z),
                            (x, y, z + 1), (x, y, z - 1)
                        ]
                        for nx, ny, nz in neighbors:
                            if (
                                0 <= nx < self.grid_size and
                                0 <= ny < self.grid_size and
                                0 <= nz < self.grid_size and
                                self.grid[nx, ny, nz] == 0
                            ):
                                possible.add((nx, ny, nz))
        self.available_actions = list(possible)

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
                            time.sleep(0.1)

                    return float(data.get("reward", 0.0))

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⏳ Waiting for valid reward file... ({e})")
                    time.sleep(0.1)

            else:
                if time.time() - start_time > timeout:
                    print("❌ Timeout: No external reward received.")
                    return 0.0
                time.sleep(0.1)




def export_voxel_grid(grid, filename):
    data = {"voxels": grid.tolist()}
    with open(filename, 'w') as f:
        json.dump(data, f)
