# import asyncio
# import aiohttp
# import json
# import threading
import time
# import rhinoinside
# rhinoinside.load(8)
# import System
# import Rhino.Geometry as rg
import rhino3dm


from flask import Flask, request, jsonify
import ghhops_server as hs

# Create Flask app that will handle both server and hops
hop_server = Flask(__name__)
hop_server.config['SECRET_KEY'] = 'group7'
hops = hs.Hops(hop_server)

# Global storage for data exchange
latest_voxel_data = None
latest_reward = None
current_step = 0  # Initialize as integer, not None
step_data = {}  # Store data by step number
emit_flag = {"voxel": False, "reward": False}

# === SERVER ROUTES (for agent communication) ===

@hop_server.route('/agent/send_voxel', methods=['POST'])
def agent_send_voxel():
    global latest_voxel_data, emit_flag, current_step, step_data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
    
    # Extract step number from the data or increment current step
    step_num = data.get('step', current_step)
    if step_num is not None:
        current_step = max(current_step, int(step_num))
    else:
        current_step += 1
        step_num = current_step
    
    # Store data with step number
    step_data[step_num] = {
        'voxel_data': data,
        'timestamp': time.time(),
        'has_reward': False
    }
    
    latest_voxel_data = data
    emit_flag["voxel"] = True
    print(f"Received voxel data for step {step_num}: {data}")
    return jsonify({"status": "success", "message": "Voxel data received", "step": step_num}), 200

# @app.route('/env/get_voxel', methods=['GET'])
# def env_get_voxel():
#     global latest_voxel_data, emit_flag
#     if latest_voxel_data is not None:
#         emit_flag["voxel"] = False
#         return jsonify({"status": "success", "voxel_data": latest_voxel_data}), 200
#     else:
#         return jsonify({"status": "error", "message": "No voxel data available"}), 400

# @app.route('/env/send_reward', methods=['POST'])
# def env_send_reward():
#     global latest_reward, emit_flag
#     if request.is_json:
#         latest_reward = request.get_json()
#     else:
#         latest_reward = request.form.to_dict()
#     emit_flag["reward"] = True
#     return jsonify({"status": "success", "message": "Reward received"}), 200

@hop_server.route('/agent/get_reward', methods=['GET'])
def agent_get_reward():
    global latest_reward, emit_flag
    if latest_reward is not None:
        emit_flag["reward"] = False
        return jsonify({"status": "success", "reward": latest_reward}), 200
    else:
        return jsonify({"status": "error", "message": "No reward available"}), 400

@hop_server.route('/emit', methods=['GET'])
def emit_status():
    return jsonify(emit_flag), 200

# === HOPS COMPONENTS (for Grasshopper communication) ===

@hops.component(
    '/get_voxels_continuous',
    name='continuous voxel monitor',
    description='continuously monitor and output voxels as they update from training',
    inputs=[
        hs.HopsBoolean('enable', 'E', 'enable continuous monitoring'),
        hs.HopsNumber('update_interval', 'I', 'update check interval in seconds (default 0.1)')
    ],
    outputs=[
        hs.HopsBrep('voxels', 'V', 'current voxels from agent training', access=hs.HopsParamAccess.LIST),
        hs.HopsInteger('current_step', 'S', 'current step number being processed'),
        hs.HopsBoolean('data_updated', 'U', 'true when new data is available'),
        hs.HopsString('status', 'ST', 'current status message')
    ]
)
def get_voxels_continuous(enable=True, update_interval=0.1):
    global latest_voxel_data, emit_flag, current_step, step_data
    
    if not enable:
        return [], 0, False, "Monitoring disabled"
    
    # Check if there's any new step data
    if not step_data:
        return [], 0, False, "Waiting for training data..."
    
    # Get the latest step
    latest_step = max(step_data.keys())
    step_info = step_data[latest_step]
    voxel_data = step_info['voxel_data'].get("data")
    
    if voxel_data is None:
        return [], latest_step, False, f"No voxel data for step {latest_step}"
    
    # Process the voxel data
    boxes = []
    size = len(voxel_data)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if voxel_data[x][y][z] == 1:
                    pt1 = rhino3dm.Point3d(x, y, z)
                    pt2 = rhino3dm.Point3d(x + 1, y + 1, z + 1)
                    bbox = rhino3dm.BoundingBox(pt1, pt2)
                    boxes.append(bbox.ToBrep())
    
    # Small delay to prevent overwhelming Grasshopper
    time.sleep(update_interval)
    
    return boxes, latest_step, True, f"Updated step {latest_step} with {len(boxes)} voxels"

@hops.component(
    '/get_voxels_by_step',
    name='get voxels by step number',
    description='retrieve voxels for a specific step number',
    inputs=[
        hs.HopsInteger('step_number', 'N', 'specific step number to retrieve'),
        hs.HopsNumber('timeout', 'T', 'timeout in seconds to wait for step (default 5)')
    ],
    outputs=[
        hs.HopsBrep('voxels', 'V', 'voxels from specified step', access=hs.HopsParamAccess.LIST),
        hs.HopsInteger('step_found', 'S', 'step number that was found'),
        hs.HopsBoolean('success', 'OK', 'true if step was found within timeout'),
        hs.HopsString('message', 'M', 'status message')
    ]
)
def get_voxels_by_step(step_number, timeout=5):
    global step_data
    
    print(f"Waiting for voxel data for step {step_number}...")
    
    start_time = time.time()
    
    # Wait for data for this specific step
    while step_number not in step_data and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    if step_number not in step_data:
        message = f"Timeout: No voxel data available for step {step_number}"
        print(message)
        return [], step_number, False, message
    
    step_info = step_data[step_number]
    voxel_data = step_info['voxel_data'].get("data")
    
    if voxel_data is None:
        message = f'No voxel data in received payload for step {step_number}'
        print(message)
        return [], step_number, False, message
    
    print(f"Processing voxel data for step {step_number}")
    
    # Process the voxel data
    boxes = []
    size = len(voxel_data)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if voxel_data[x][y][z] == 1:
                    pt1 = rhino3dm.Point3d(x, y, z)
                    pt2 = rhino3dm.Point3d(x + 1, y + 1, z + 1)
                    bbox = rhino3dm.BoundingBox(pt1, pt2)
                    boxes.append(bbox.ToBrep())
    
    message = f"Successfully processed {len(boxes)} voxels for step {step_number}"
    return boxes, step_number, True, message

# Replace the old get_voxels component with get_voxels_by_step for backward compatibility
@hops.component(
    '/get_voxels',
    name='get them crooked voxels',
    description='retrieve the voxels from the training agent (legacy - use get_voxels_by_step instead)',
    inputs=[
        hs.HopsInteger('num_steps', 'N', 'number of steps in the iteration (should match agent training)'),
    ],
    outputs=[
        hs.HopsBrep('voxels', 'V', 'voxels from agent training', access=hs.HopsParamAccess.LIST),
        hs.HopsInteger('current_step', 'S', 'current step number being processed'),
    ]
)
def get_voxels(num_steps):
    # Redirect to the new component
    voxels, step_found, success, message = get_voxels_by_step(num_steps, timeout=30)
    return voxels, step_found

@hops.component(
    '/send_reward',
    name='send them rewards',
    description='send the reward back to the environment',
    inputs=[
        hs.HopsNumber('reward', 'R', 'reward for the voxels to send back'),
        hs.HopsInteger('step_number', 'S', 'step number to associate with this reward')
    ],
    outputs=[
        hs.HopsInteger('next_step', 'N', 'next step number to process')
    ]
)
def hops_send_reward(reward, step_number):
    global latest_reward, emit_flag, step_data
    
    # Store the reward for this specific step
    if step_number in step_data:
        step_data[step_number]['has_reward'] = True
        step_data[step_number]['reward'] = reward
    
    latest_reward = {"reward": reward, "step": step_number}
    emit_flag["reward"] = True
    
    print(f"Reward {reward} sent for step {step_number}")
    
    # Clean up old step data (keep only last 10 steps)
    steps_to_keep = sorted(step_data.keys())[-10:]
    step_data = {k: v for k, v in step_data.items() if k in steps_to_keep}
    
    return step_number + 1

@hops.component(
    '/wait_for_step',
    name='wait for training step',
    description='wait for a specific training step with countdown',
    inputs=[
        hs.HopsInteger('target_step', 'T', 'target step number to wait for'),
        hs.HopsNumber('wait_interval', 'I', 'wait interval in seconds (default 0.5)')
    ],
    outputs=[
        hs.HopsBoolean('step_ready', 'R', 'true when step data is ready'),
        hs.HopsInteger('current_step', 'C', 'current available step number')
    ]
)
def wait_for_step(target_step, wait_interval=0.5):
    global step_data, current_step
    
    # Wait for the specified interval
    time.sleep(wait_interval)
    
    # Check if target step is available
    step_ready = target_step in step_data
    
    return step_ready, current_step

if __name__ == "__main__":
    hop_server.run(host='localhost', port=5555, debug=True)
