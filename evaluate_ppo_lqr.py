import ray
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ray.rllib.algorithms.ppo import PPOConfig
from crane_env_lqr_reward import CraneEnvDiscrete

ray.init()

CHECKPOINT_PATH = r"C:\Users\josem\OneDrive\Escritorio\2025-2026\TFG\RL\tfg_rl_crane\checkpoints\crane_ppo_lqr_re"

config = (
    PPOConfig()
    .environment(env=CraneEnvDiscrete)
    .env_runners(num_env_runners=0)
)

algo = config.build()
algo.restore(CHECKPOINT_PATH)

rl_module = algo.get_module()

env = CraneEnvDiscrete()
obs, _ = env.reset(seed=42)

xs, dxs, phis, dphis, voltajes = [], [], [], [], []
cause = "max_steps"

for _ in range(500):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = rl_module.forward_inference({"obs": obs_tensor})

    logits = torch.tensor(output["action_dist_inputs"].numpy()[0])
    action = int(torch.argmax(F.softmax(logits, dim=-1)).item())


    obs, reward, terminated, truncated, info = env.step(action)

    xs.append(env.state[0])
    dxs.append(env.state[1])
    phis.append(env.state[2])
    dphis.append(env.state[3])
    voltajes.append(env.state[4] * env.Vmax)

    if terminated or truncated:
        cause = info.get("termination_cause", "desconocido")
        break

t = [i * env.dt for i in range(len(xs))]
print(f"Episodio terminado por: {cause} en t={t[-1]:.2f}s")

fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

axes[0].plot(t, xs, label='x (m)')
axes[0].axhline(env.xref, color='r', linestyle='--', label='xref=1')
axes[0].set_ylabel('Posición x (m)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t, [p * 180 / np.pi for p in phis], color='orange', label='phi (deg)')
axes[1].axhline(0, color='r', linestyle='--')
axes[1].axhline(np.degrees(env.phi_max), color='r', linestyle=':', label=f'phi_max=±{np.degrees(env.phi_max):.1f}°')
axes[1].axhline(-np.degrees(env.phi_max), color='r', linestyle=':')
axes[1].set_ylabel('Ángulo phi (°)')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(t, dxs, color='purple', label='dx (m/s)')
axes[2].axhline(env.dx_limit, color='r', linestyle=':', label=f'dx_limit=±{env.dx_limit}')
axes[2].axhline(-env.dx_limit, color='r', linestyle=':')
axes[2].set_ylabel('Velocidad dx (m/s)')
axes[2].legend()
axes[2].grid(True)

axes[3].plot(t, dphis, color='red', label='dphi (rad/s)')
axes[3].axhline(env.dphi_limit, color='r', linestyle=':', label=f'dphi_limit=±{env.dphi_limit}')
axes[3].axhline(-env.dphi_limit, color='r', linestyle=':')
axes[3].set_ylabel('Vel. angular dphi (rad/s)')
axes[3].legend()
axes[3].grid(True)

axes[4].step(t, voltajes, color='green', label='Voltaje V', where='post')
axes[4].axhline(0, color='r', linestyle='--')
axes[4].set_ylabel('Voltaje (V)')
axes[4].set_xlabel('Tiempo (s)')
axes[4].legend()
axes[4].grid(True)

plt.suptitle('Evaluación política PPO Discreta - CraneEnv')
plt.tight_layout()
plt.savefig('evaluacion_ppo_discrete.png', dpi=150)
plt.show()

ray.shutdown()