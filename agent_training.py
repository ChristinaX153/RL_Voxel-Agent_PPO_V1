import os
import time
import argparse
import numpy as np
# import torch
import math
# import json
import psutil  # process management for locating/killing PowerShell

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
# from sb3_contrib.common.maskable.evaluation import evaluate_policy  # optional

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from gymnasium import wrappers
from gymnasium import spaces as gym_spaces

from voxel_env import VoxelEnv

from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

# ---------------------------
# Utility functions
# ---------------------------

# Parse grid argument to int or 3-tuple[int,int,int]
def parse_grid(grid_str: str):
    s = str(grid_str).strip()
    s = s.replace('[', '(').replace(']', ')')
    # pure int
    if s.isdigit():
        return int(s)
    # strip parens if present
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) == 1 and parts[0].isdigit():
        return int(parts[0])
    if len(parts) == 3 and all(p.lstrip('-').isdigit() for p in parts):
        return tuple(map(int, parts))
    raise ValueError(f"Invalid --grid value: {grid_str}. Use 5 or 5,5,10 or (5,5,10)")

# Enhanced export function with EPW and facade parameters
def export_voxel_grid_enhanced(grid, filename, epw_file=None, facade_params=None, action_info=None):
    """Export voxel grid with additional metadata"""
    def _to_serializable(o):
        import numpy as _np
        if isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, (_np.generic,)):
            return o.item()
        if isinstance(o, dict):
            return {k: _to_serializable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, set)):
            return [_to_serializable(x) for x in o]
        return o

    data = {
        "voxels": grid.tolist(),
        "epw_file": os.path.basename(epw_file) if epw_file else None,
        "facade_params": {},
    }

    if facade_params:
        for key, value in facade_params.items():
            data["facade_params"][key] = _to_serializable(value)

    if action_info:
        data["action_info"] = _to_serializable(action_info)

    with open(filename, 'w') as f:
        import json
        json.dump(_to_serializable(data), f, indent=2)



def _locate_ps_process(script_path: str):
    sp_norm = os.path.normcase(script_path)
    for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            cmd = p.info.get("cmdline") or []
            if any(os.path.normcase(script_path) in os.path.normcase(c) for c in cmd):
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def _start_rhino_new_window(script_path: str):
    if not os.path.exists(script_path):
        print(f"[RhinoCompute] Script not found: {script_path}")
        return None
    try:
        # Launch a new PowerShell window using os.startfile (opens with associated handler)
    # Launch PowerShell script (no direct PID returned)
        os.system(f"start powershell {script_path}")
        time.sleep(10.0)  # Increased wait time for Rhino to fully start (was 1.5)
        proc = _locate_ps_process(script_path)
        if proc:
            print(f"[RhinoCompute] Started PowerShell window (PID={proc.pid}) for {script_path}")
        else:
            print("[RhinoCompute] Warning: Could not locate PowerShell process after start.")
        return proc
    except Exception as e:
        print(f"[RhinoCompute] Failed to start: {e}")
        return None

def _stop_rhino(proc):
    if proc is None:
        return
    try:
        if proc.is_running():
            print(f"[RhinoCompute] Terminating PID={proc.pid} ...")
            for child in proc.children(recursive=True):
                try: child.terminate()
                except Exception: pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                print("[RhinoCompute] Forcing kill...")
                proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

class RhinoRestartCallback(BaseCallback):
    """
    Opens start_rhino_compute.ps1 in a new PowerShell window (os.startfile) and
    restarts it every RHINO_RESTART_INTERVAL timesteps.
    """
    def __init__(self, script_path: str, restart_interval: int, verbose: int = 0):
        super().__init__(verbose)
        self.script_path = script_path
        self.restart_interval = restart_interval
        self._proc = None
        self._next_restart = restart_interval
        self._cycle = 0

    def _init_callback(self):
        print(f"[RhinoCompute] Initializing with restart interval: {self.restart_interval} timesteps")
        print(f"[RhinoCompute] Next restart scheduled at: {self._next_restart} timesteps")
        self._proc = _start_rhino_new_window(self.script_path)

    def _on_step(self) -> bool:
        # Only check for restart at the end of complete rollouts
        # num_timesteps tracks total timesteps across all environments
        if self.num_timesteps >= self._next_restart:
            self._cycle += 1
            if self.verbose:
                print(f"[RhinoCompute] Restart #{self._cycle} at timesteps={self.num_timesteps} (interval={self.restart_interval})")
            _stop_rhino(self._proc)
            time.sleep(5.0)  # Give time for process to fully terminate
            self._proc = _start_rhino_new_window(self.script_path)
            self._next_restart = self.num_timesteps + self.restart_interval
        return True

    def _on_training_end(self):
        _stop_rhino(self._proc)
        self._proc = None
# ---------------------------
# CLI args
# ---------------------------
parser = argparse.ArgumentParser(description='Train MaskablePPO agent for voxel environment')
parser.add_argument('--port', type=int, default=81, help='Rhino.Compute coordinator port for all environments (default: 81)')
parser.add_argument('--new', action='store_true', default=False,
                    help='Start a brand new model (ignore checkpoints and saved models)')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Resume training from the most recent checkpoint in models/PPO/checkpoints/')
parser.add_argument('--n_envs', type=int, default=1, help='Number of parallel environments (default: 1)')
parser.add_argument('--grid', type=parse_grid, default='(5,5,5)',
                    help='Grid: int like 5 or tuple like 5,5,10 or (5,5,10)')
parser.add_argument('--chk_freq', type=int, default=500,
                    help='Frequency of saving checkpoints (default: 500)')
parser.add_argument('--timesteps', type=int, default=256,
                    help='Total environment steps to train (default: 256)')
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate')
parser.add_argument('--lr_final', type=float, default=3e-5, help='Final learning rate at end of training')
parser.add_argument('--lr_schedule', type=str, choices=['constant', 'linear', 'cosine'], default='cosine',
                    help='LR schedule over progress_remaining (1→0)')
parser.add_argument('--early-done', action='store_true', default=False,
                    help='Enable an explicit DONE action for early termination')
parser.add_argument('--early-done-bonus', type=float, default=0.0,
                    help='Bonus (or penalty if negative) when DONE is used meeting min occupancy')
parser.add_argument('--early-done-min-voxels', type=float, default=0.1,
                    help='Minimum fraction (<=1) or absolute count (>1) of active voxels required for DONE bonus')
parser.add_argument('--log-actions-every', type=int, default=100,
                    help='Log episode actions every N episodes (default: 100, set to 0 to disable)')
parser.add_argument('--load-model', type=str, default=None,
                    help='Exact path OR basename (without .zip) of a saved model in models/PPO/')
parser.add_argument('--checkpoint-file', type=str, default=None,
                    help='Exact checkpoint filename inside models/PPO/checkpoints/ to resume (overrides --checkpoint)')
args = parser.parse_args()

GRID_PARAM = parse_grid(args.grid)

# ---------------------------
# Basic info
# ---------------------------
print("Using CPU for PPO training (recommended for MLP policies)")

# ---------------------------
# Mask function for ActionMasker
# ---------------------------
def mask_fn(env):
    # return env.action_masks()
    return env.unwrapped.action_masks()

# ---------------------------
# PPO hyperparams
# ---------------------------
num_steps = 128          # rollout length per env 
num_epochs = 5           # PPO epochs per update
batch_size = 32          # must divide (num_steps * num_envs); 1024*k is always divisible by 256

RHINO_PS1 = os.path.join(os.getcwd(), "start_rhino_compute.ps1")
# Make restart interval a multiple of episode length (256 steps)
# With num_steps=128, this gives us: 1000//128 * 128 = 7 * 128 = 896 timesteps
RHINO_RESTART_INTERVAL = (1000 // num_steps) * num_steps
print(f"[RhinoCompute] Restart interval set to {RHINO_RESTART_INTERVAL} timesteps ({RHINO_RESTART_INTERVAL // 256} episodes)")

starting_step = 0
model_dir = os.path.join(os.getcwd(), 'models', 'PPO')
checkpoint_dir = os.path.join(model_dir, "checkpoints")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# ---------------------------
# Learning rate schedule
# ---------------------------
def _make_lr_schedule(kind: str, lr_start: float, lr_end: float):
    if kind == 'constant':
        return lambda progress_remaining: lr_start
    if kind == 'linear':
        # progress_remaining: 1 → 0, so lr: start → end
        return lambda progress_remaining: lr_end + (lr_start - lr_end) * progress_remaining
    if kind == 'cosine':
        # smooth decay start → end
        return lambda progress_remaining: lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
    return lambda progress_remaining: lr_start

# Build schedule once
lr_schedule = _make_lr_schedule(args.lr_schedule, args.lr, args.lr_final)
norm_path = os.path.join(model_dir, "vecnormalize.pkl")

# ---------------------------
# Model loading and name determination
# ---------------------------
model_date_time = time.strftime("%Y%m%d-%H%M", time.localtime())
model_name = f"maskable_ppo_voxel_{model_date_time}"  # Default for new models

if args.checkpoint:
    existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if not existing_checkpoints:
        raise FileNotFoundError("No checkpoints found in models/PPO/checkpoints/. Cannot resume.")
    latest_ckpt = max(existing_checkpoints,
                      key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    print(f"Resuming from checkpoint: {latest_ckpt}")
    model_name = os.path.splitext(latest_ckpt)[0]  # Use checkpoint name
    model_path = os.path.join(checkpoint_dir, latest_ckpt)
elif not args.new:
    existing_models = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if existing_models:
        latest_model = os.path.splitext(sorted(existing_models, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))[-1])[0]
        print(f"Loading existing MaskablePPO model: {latest_model}")
        model_name = latest_model  # Use existing model name
        starting_step = 0  # Will be updated after loading
    else:
        print("No saved models found. Starting fresh.")
        args.new = True

# After determining model_name / before env creation:
explicit_model_path = None
if args.checkpoint_file:
    ck_path = os.path.join(checkpoint_dir, args.checkpoint_file)
    if not os.path.isfile(ck_path):
        raise FileNotFoundError(f"--checkpoint-file not found: {ck_path}")
    print(f"Resuming from explicit checkpoint: {args.checkpoint_file}")
    model_path = ck_path
    model_name = os.path.splitext(args.checkpoint_file)[0]
    args.checkpoint = True  # enforce checkpoint load
elif args.load_model:
    # Accept absolute path or basename (with or without .zip)
    cand = args.load_model
    if not cand.endswith(".zip"):
        cand_zip = cand + ".zip"
    else:
        cand_zip = cand
    if os.path.isabs(cand_zip):
        mp = cand_zip
    else:
        mp = os.path.join(model_dir, cand_zip)
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"--load-model target not found: {mp}")
    print(f"Loading explicit saved model: {mp}")
    model_name = os.path.splitext(os.path.basename(mp))[0]
    explicit_model_path = mp
    args.new = False  # force load path

# ---------------------------
# Vec env factory (now with proper model name)
# ---------------------------
# Reward normalization: custom wrapper replaces prior implementation
class RewardNormVecEnv(VecEnvWrapper):
    """
    Reward normalization for Dict (or any) observations when VecNormalize (obs) is not usable.
    Normalizes rewards using running variance of discounted returns (shared across envs).
    """
    def __init__(self, venv, gamma: float = 0.99, eps: float = 1e-8, clip: float = 10.0):
        super().__init__(venv)
        self.gamma = gamma
        self.eps = eps
        self.clip = clip
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.count = 1e-4
        self.running_mean = 0.0
        self.running_var = 1.0  # actually sum of squared diffs (Welford accumulator)

    def reset(self):
        self.returns[:] = 0.0
        return self.venv.reset()

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # Update discounted returns per env
        self.returns = self.returns * self.gamma + rewards
        # Welford update over each return (treat each env return as a sample)
        for R in self.returns:
            self.count += 1.0
            delta = R - self.running_mean
            self.running_mean += delta / self.count
            delta2 = R - self.running_mean
            self.running_var += delta * delta2
        std = max(self.eps, np.sqrt(self.running_var / self.count))
        norm_rewards = np.clip(rewards / std, -self.clip, self.clip)

        # Reset return accumulator when an episode ends
        for i, d in enumerate(dones):
            if d:
                self.returns[i] = 0.0
        return obs, norm_rewards, dones, infos

def make_single_env():
    def _thunk():
        e = VoxelEnv(
            port=args.port,  # All environments use the same coordinator port
            grid_size=GRID_PARAM,
            device='cpu',
            step_penalty=0.01,
            str_wt=0.3, # 2.0
            sun_wt=0.15, # 0.4
            wst_wt=0.05, # 0.005
            cst_wt=0.05, # 0.05
            day_wt=0.15, # 0.4
            num_repulsors=1,
            repulsor_radius=3.0,
            repulsor_provider="random",
            height_wt=0.6,
            repulsor_clear_wt=0.5,
            repulsor_penalty_wt=0.5,
            early_done=args.early_done,
            early_done_bonus=args.early_done_bonus,
            early_done_min_voxels=args.early_done_min_voxels,
            save_actions=args.log_actions_every > 0,
            actions_output_dir=f"episode_actions/{model_name}",
            export_last_epoch_episode=True,
            model_name=model_name,
            log_actions_every=args.log_actions_every
        )
        e = wrappers.TimeLimit(env=e, max_episode_steps=256)
        e = ActionMasker(e, mask_fn)
        return e
    return _thunk

num_envs = max(1, args.n_envs)
print(f"[ENV] Creating {num_envs} environments, all connecting to Rhino.Compute coordinator on port {args.port}")
env = DummyVecEnv([make_single_env() for i in range(num_envs)])
env = VecMonitor(env)

# Decide normalization strategy
use_vecnormalize = isinstance(env.observation_space, gym_spaces.Box)
if use_vecnormalize:
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    if os.path.exists(norm_path):
        try:
            env = VecNormalize.load(norm_path, env)
            env.training = True
            env.norm_reward = True
            print("Loaded VecNormalize stats.")
        except AssertionError:
            print("VecNormalize stats incompatible with new obs shape. Starting fresh stats.")
else:
    print("Dict observation space detected -> applying RewardNormVecEnv (reward-only normalization).")
    env = RewardNormVecEnv(env)

# Now actually load the models with the environment created
if args.checkpoint:
    model = MaskablePPO.load(model_path, env=env, device="cpu")
    model.lr_schedule = lr_schedule
elif explicit_model_path:
    model = MaskablePPO.load(explicit_model_path, env=env, device="cpu")
    starting_step = getattr(model, "_total_timesteps", 0)
    model.lr_schedule = lr_schedule
elif not args.new and 'latest_model' in locals():
    model = MaskablePPO.load(os.path.join(model_dir, f"{model_name}.zip"), env=env, device='cpu')
    starting_step = getattr(model, "_total_timesteps", 0)
    model.lr_schedule = lr_schedule

# When creating a new model (policy always MultiInputPolicy for Dict)
if args.new:
    print("Creating a new MaskablePPO model")
    policy_type = "MultiInputPolicy"
    model = MaskablePPO(
        policy_type,
        env,
        verbose=1,
        learning_rate=lr_schedule,
        gamma=0.99,
        n_steps=num_steps,
        batch_size=batch_size,
        n_epochs=num_epochs,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        policy_kwargs=dict(ortho_init=False),
        device="cpu",
        tensorboard_log=f"./ppo_voxel_tensorboard/{model_name}/"
    )
    starting_step = 0

# ---------------------------
# Training setup
# ---------------------------
print(f"Starting training from {starting_step} with {num_envs} parallel env(s) "
      f"({'early-done ON' if args.early_done else 'early-done OFF'}) on CPU...")
total_steps = args.timesteps

save_path = os.path.join(model_dir,
                         f"{model_name}_{starting_step+total_steps}")

checkpoint_callback = CheckpointCallback(
    save_freq=args.chk_freq,
    save_path=checkpoint_dir,
    name_prefix=model_name
)

# Combine checkpointing with Rhino restart and custom logging callbacks
rhino_callback = RhinoRestartCallback(
    script_path=RHINO_PS1,
    restart_interval=RHINO_RESTART_INTERVAL,
    verbose=1
)

class LastEpisodeCallback(BaseCallback):
    """Callback to capture actions from the last episode of training"""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.last_episode_actions = []
        self.last_episode_rewards = []
        self.last_episode_infos = []
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_infos = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Get the action that was just taken
        if hasattr(self.locals, 'actions') and self.locals['actions'] is not None:
            action = self.locals['actions'][0] if isinstance(self.locals['actions'], (list, np.ndarray)) else self.locals['actions']
            self.current_episode_actions.append(int(action))
        
        # Get reward and info
        if hasattr(self.locals, 'rewards') and self.locals['rewards'] is not None:
            reward = self.locals['rewards'][0] if isinstance(self.locals['rewards'], (list, np.ndarray)) else self.locals['rewards']
            self.current_episode_rewards.append(float(reward))
        
        if hasattr(self.locals, 'infos') and self.locals['infos'] is not None:
            info = self.locals['infos'][0] if isinstance(self.locals['infos'], list) else self.locals['infos']
            self.current_episode_infos.append(info)

        # Check if episode ended
        if hasattr(self.locals, 'dones') and self.locals['dones'] is not None:
            done = self.locals['dones'][0] if isinstance(self.locals['dones'], (list, np.ndarray)) else self.locals['dones']
            if done:
                # Episode ended, save the actions
                self.last_episode_actions = self.current_episode_actions.copy()
                self.last_episode_rewards = self.current_episode_rewards.copy()
                self.last_episode_infos = self.current_episode_infos.copy()
                self.episode_count += 1
                
                # Reset for next episode
                self.current_episode_actions = []
                self.current_episode_rewards = []
                self.current_episode_infos = []
                
                if self.verbose > 0:
                    print(f"[LastEpisode] Captured episode #{self.episode_count} with {len(self.last_episode_actions)} actions")
        
        return True

class EpochTrackingCallback(BaseCallback):
    """Callback to track epochs and mark last episode for step export"""
    def __init__(self, vec_env, num_steps: int, num_epochs: int, model_name: str = "model", verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.current_epoch = 0
        self.steps_in_current_rollout = 0
        self.episodes_in_current_epoch = 0
        
    def _on_step(self) -> bool:
        self.steps_in_current_rollout += 1
        
        # Update epoch information in all environments continuously
        try:
            if hasattr(self.vec_env, 'venv'):
                dummy_vec = self.vec_env.venv.venv  # VecNormalize -> VecMonitor -> DummyVecEnv
                if hasattr(dummy_vec, 'envs'):
                    for env in dummy_vec.envs:
                        if hasattr(env, 'unwrapped'):
                            base_env = env.unwrapped
                            if hasattr(base_env, 'set_epoch_info'):
                                base_env.set_epoch_info(self.current_epoch, is_last_episode=False)
        except Exception as e:
            if self.verbose > 1:
                print(f"[EPOCH_TRACKER] Warning: Could not update epoch info: {e}")
        
        # Check if we completed a rollout (num_steps per env)
        if self.steps_in_current_rollout >= self.num_steps:
            self.steps_in_current_rollout = 0
            self.current_epoch += 1
            
            if self.verbose > 0:
                print(f"[EPOCH_TRACKER] Starting epoch {self.current_epoch}")
            
            # Update epoch information after rollout completion
            try:
                if hasattr(self.vec_env, 'venv'):
                    dummy_vec = self.vec_env.venv.venv  # VecNormalize -> VecMonitor -> DummyVecEnv
                    if hasattr(dummy_vec, 'envs'):
                        for env in dummy_vec.envs:
                            if hasattr(env, 'unwrapped'):
                                base_env = env.unwrapped
                                if hasattr(base_env, 'set_epoch_info'):
                                    # Mark for epoch step export if this is the last epoch of current update
                                    is_last_epoch = (self.current_epoch % self.num_epochs == 0)
                                    base_env.set_epoch_info(self.current_epoch, is_last_episode=is_last_epoch)
                                    if self.verbose > 0 and is_last_epoch:
                                        print(f"[EPOCH_TRACKER] Marked env for epoch {self.current_epoch} step export")
            except Exception as e:
                if self.verbose > 0:
                    print(f"[EPOCH_TRACKER] Warning: Could not access base environments: {e}")
        
        return True

last_episode_callback = LastEpisodeCallback(verbose=1)

epoch_tracking_callback = EpochTrackingCallback(
    vec_env=env,
    num_steps=num_steps,
    num_epochs=num_epochs,
    model_name=model_name,
    verbose=1
)

combined_callback = CallbackList([checkpoint_callback, rhino_callback, last_episode_callback, epoch_tracking_callback])
# ---------------------------
# Train
# ---------------------------
model.learn(
    total_timesteps=total_steps,
    progress_bar=True,
    callback=combined_callback,
    reset_num_timesteps=False
)

if use_vecnormalize:
    env.save(norm_path)
    print(f"Saved VecNormalize stats: {norm_path}")
else:
    print("VecNormalize not used (Dict obs); skipping stats save.")
print("Training completed")

# ---------------------------
# Save the trained model
# ---------------------------
final_step = int(getattr(model, "num_timesteps", starting_step+total_steps))
final_save_path = os.path.join(model_dir,
                               f"{model_name}_{final_step}")
model.save(final_save_path)
print(f"Model saved as: {final_save_path}.zip")

# ---------------------------
# Output actions from last training episode
# ---------------------------
print("Outputting actions from the last training episode...")

