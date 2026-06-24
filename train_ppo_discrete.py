import ray 
from ray.rllib.algorithms.ppo import PPOConfig 
from ray.rllib.callbacks.callbacks import RLlibCallback 
from crane_env_discrete import CraneEnvDiscrete 
import os 
import shutil 
import numpy as np 
from collections import Counter 
 
# ── Callback ────────────────────────────────────────────────────────────────── 
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
 
# Borrar checkpoint anterior si existe 
old_checkpoint = os.path.abspath("checkpoints/crane_ppo_discrete") 
if os.path.exists(old_checkpoint): 
    shutil.rmtree(old_checkpoint) 
    print("Checkpoint anterior eliminado") 
 
ray.init() 
 
config = ( 
    PPOConfig() 
    .environment(env=CraneEnvDiscrete) 
    .env_runners(num_env_runners=0) 
    .callbacks(LogTerminationCallback) 
    .training( 
            lr=1e-4,  
            gamma=0.99, #factor de descuento 
            lambda_=0.95, #factor GAE 
            train_batch_size=8000, 
            minibatch_size=128, 
            num_epochs=10, 
            clip_param=0.2, 
            vf_loss_coeff=0.5, 
            entropy_coeff=0.005, 
    ) 
) 
 
algo = config.build() 
 
best_reward=-float("inf") 
ESPERA = 50 
iters_sin_mejora = 0 
 
for i in range(300): 
     
    result = algo.train() 
    reward_mean=result["env_runners"]["episode_return_mean"] 
    ep_len=result["env_runners"]["episode_len_mean"] 
 
    print(f"Iter {i:3d} | reward: {reward_mean:8.2f} | ep.len: {ep_len:.0f}| best: {best_reward:.2f}") 
 
    #Guardar checkpoint si mejora     
    if not np.isnan(reward_mean) and reward_mean > best_reward: 
        best_reward=reward_mean 
        iters_sin_mejora = 0 
        save_dir=os.path.abspath("checkpoints/crane_ppo_discrete") 
        os.makedirs(save_dir, exist_ok=True) 
        algo.save(save_dir) 
        print(f"→ Checkpoint guardado (mejor: {best_reward:.2f})") 
    else: 
        iters_sin_mejora +=1 
 
    if iters_sin_mejora >= ESPERA: 
        print(f"  → Early stopping en iter {i} — sin mejora en {ESPERA} iteraciones") 
        break 
 
ray.shutdown()
