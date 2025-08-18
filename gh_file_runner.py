import compute_rhino3d.Grasshopper as gh
import json
import gc
import os
import platform
import subprocess

def weird_str_to_float(input_string: str) -> float:
    numeric_part = ''.join(char for char in input_string if char.isdigit() or char == '.')
    try:
        return float(numeric_part)
    except ValueError:
        return -0.2  # fallback in case of error

def get_reward_gh(grid, merged_gh_file, epw_file, sun_wt, str_wt) -> float:
    if grid is None:
        return -0.2

    # epw_file = epw_file.replace("\\", "/")
    # basename = os.path.basename(epw_file)
    # epw_file = os.path.join('epw',basename)
    epw_file = os.path.abspath(epw_file)
    
    if platform.system() =="Linux":
        result = subprocess.run(['wslpath', '-w', str(epw_file)], capture_output=True, text=True)
        epw_file = result.stdout.strip().replace("\\", "/")
    print(epw_file)
    # Prepare voxel grid as JSON
    data = json.dumps({"voxels": grid.tolist()})
    voxel_in = gh.DataTree("voxel_json")
    voxel_in.Append([0], [data])

    # Send EPW path to Hops
    epw_path = gh.DataTree("epw_path")
    epw_path.Append([0], [epw_file])

    try:
        # Evaluate Grasshopper definition
        output = gh.EvaluateDefinition(merged_gh_file, [epw_path, voxel_in])
        print(output)

        # Extract reward values
        cyclops_reward_str = output['values'][0]['InnerTree']['{0;0}'][0]['data']
        karamba_reward_str = output['values'][1]['InnerTree']['{0}'][0]['data']

        # Parse to float
        cyclops_reward = weird_str_to_float(cyclops_reward_str)
        karamba_reward = weird_str_to_float(karamba_reward_str)

        # Weighted reward
        reward = sun_wt * cyclops_reward + str_wt * karamba_reward

        # ✅ Log reward summary only
        print(f"[REWARD] Cyclops: {sun_wt * cyclops_reward:.2f}, Karamba: {str_wt * karamba_reward:.2f}, Total: {reward:.2f}")

    except Exception as e:
        print(f"[ERROR] Failed to evaluate GH definition: {e}")
        reward = -0.2

    gc.collect()
    return reward
