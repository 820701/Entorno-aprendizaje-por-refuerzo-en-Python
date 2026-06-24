import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from crane_env_lqr_reward import CraneEnvDiscrete
import os
import shutil
import numpy as np
from collections import Counter

class LogTerminationCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.causas = []

    def on_episode_end(self, *, episode, **kwargs):
        cause = episode.get_infos()[-1].get("termination_cause", "desconocido")
        self.causas.append(cause)

    def on_sample_end(self, **kwargs):
        conteo = Counter(self.causas)
        print(f"  → Causas: objetivo={conteo.get('objetivo_alcanzado',0)} | truncated={conteo.get('truncated',0)} | fin_fisico={conteo.get('fin_fisico',0)}")
        self.causas = []

CHECKPOINT_PATH = r"C:\Users\josem\OneDrive\Escritorio\2025-2026\TFG\RL\tfg_rl_crane\checkpoints\crane_ppo_lqr_re"

if os.path.exists(CHECKPOINT_PATH):
    shutil.rmtree(CHECKPOINT_PATH)
    print("Checkpoint anterior eliminado")

ray.init()

config = (
    PPOConfig()
    .environment(env=CraneEnvDiscrete)
    .env_runners(num_env_runners=0)
    .callbacks(LogTerminationCallback)
    .training(
        lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        train_batch_size=4000,
        minibatch_size=128,
        num_epochs=10,
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.005,
    )
)

algo = config.build()

best_reward = -float("inf")
ESPERA = 50
iters_sin_mejora = 0

for i in range(300):
    result = algo.train()
    reward_mean = result["env_runners"]["episode_return_mean"]
    ep_len = result["env_runners"]["episode_len_mean"]

    print(f"Iter {i:3d} | reward: {reward_mean:8.2f} | ep.len: {ep_len:.0f}| best: {best_reward:.2f}")

    if not np.isnan(reward_mean) and reward_mean > best_reward:
        best_reward = reward_mean
        iters_sin_mejora = 0
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        algo.save(CHECKPOINT_PATH)
        print(f"→ Checkpoint guardado (mejor: {best_reward:.2f})")
    else:
        iters_sin_mejora += 1

    if iters_sin_mejora >= ESPERA:
        print(f"  → Early stopping en iter {i} — sin mejora en {ESPERA} iteraciones")
        break

ray.shutdown()