from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'group7'

# Global storage for data exchange
latest_voxel_data = None
latest_reward = None
emit_flag = {"voxel": False, "reward": False}

@app.route('/agent/send_voxel', methods=['POST'])
def agent_send_voxel():
    global latest_voxel_data, emit_flag
    if request.is_json:
        latest_voxel_data = request.get_json()
    else:
        latest_voxel_data = request.form.to_dict()
    emit_flag["voxel"] = True
    return jsonify({"status": "success", "message": "Voxel data received"}), 200

@app.route('/env/get_voxel', methods=['GET'])
def env_get_voxel():
    global latest_voxel_data, emit_flag
    if latest_voxel_data is not None:
        emit_flag["voxel"] = False
        return jsonify({"status": "success", "voxel_data": latest_voxel_data}), 200
    else:
        return jsonify({"status": "error", "message": "No voxel data available"}), 400

@app.route('/env/send_reward', methods=['POST'])
def env_send_reward():
    global latest_reward, emit_flag
    if request.is_json:
        latest_reward = request.get_json()
    else:
        latest_reward = request.form.to_dict()
    emit_flag["reward"] = True
    return jsonify({"status": "success", "message": "Reward received"}), 200

@app.route('/agent/get_reward', methods=['GET'])
def agent_get_reward():
    global latest_reward, emit_flag
    if latest_reward is not None:
        emit_flag["reward"] = False
        return jsonify({"status": "success", "reward": latest_reward}), 200
    else:
        return jsonify({"status": "error", "message": "No reward available"}), 400

@app.route('/emit', methods=['GET'])
def emit_status():
    # Agent/env can poll this to check if new data is available
    return jsonify(emit_flag), 200


if __name__ == '__main__':
    app.run(debug=True, port=5555)


