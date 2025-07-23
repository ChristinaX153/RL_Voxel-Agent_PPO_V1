# logging_callback.py

from stable_baselines3.common.callbacks import BaseCallback

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            print(f"🧠 Training step: {self.num_timesteps}")
        return True
