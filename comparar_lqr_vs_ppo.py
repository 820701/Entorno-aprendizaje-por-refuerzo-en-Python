import ray
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.linalg import solve_continuous_are
from ray.rllib.algorithms.ppo import PPOConfig
from crane_env_lqr_reward import CraneEnvDiscrete


M=0.5; m=0.1; l=0.5; bx=2.65; Kt=1.34; g=9.8
A = np.array([[0,1,0,0],[0,-bx/M,m*g/M,0],[0,0,0,1],[0,bx/(M*l),-(g/l)*((m/M)+1),0]])
B = np.array([[0],[Kt/M],[0],[-Kt/(M*l)]])
Q = np.diag([7.0, 0.1, 1.0, 9.0])
R = np.array([[1.0]])
P = solve_continuous_are(A, B, Q, R)
K_LQR = ((1/R[0,0]) * B.T @ P)[0]

CHECKPOINT_RL = r"C:\Users\josem\OneDrive\Escritorio\2025-2026\TFG\RL\tfg_rl_crane\checkpoints\crane_ppo_lqr_reward - DEFINITIVO"


def run_lqr(K):
    env = CraneEnvDiscrete()
    obs, _ = env.reset()
    xs, phis, dxs, dphis, voltajes = [], [], [], [], []
    for _ in range(500):
        x, dx, phi, dphi = obs[0], obs[1], obs[2], obs[3]
        state_error = np.array([x - env.xref, dx, phi, dphi])
        V = float(np.clip(-K @ state_error, -env.Vmax, env.Vmax))
        obs, _, terminated, truncated, _ = env.step_with_voltage(V)
        xs.append(obs[0]); dxs.append(obs[1]); phis.append(obs[2]); dphis.append(obs[3])
        voltajes.append(V)
        if terminated or truncated:
            break
    t = [i * env.dt for i in range(len(xs))]
    return t, xs, phis, dxs, dphis, voltajes


def run_rl(checkpoint_path):
    config = PPOConfig().environment(env=CraneEnvDiscrete).env_runners(num_env_runners=0)
    algo = config.build()
    algo.restore(checkpoint_path)
    rl_module = algo.get_module()
    env = CraneEnvDiscrete()
    obs, _ = env.reset()
    xs, phis, dxs, dphis, actions, v_norms = [], [], [], [], [], []
    for _ in range(500):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = rl_module.forward_inference({"obs": obs_tensor})
        action = int(np.argmax(output["action_dist_inputs"].numpy()[0]))
        obs, _, terminated, truncated, _ = env.step(action)
        v_norms.append(obs[4])
        xs.append(obs[0]); dxs.append(obs[1]); phis.append(obs[2]); dphis.append(obs[3])
        actions.append(action)
        if terminated or truncated:
            break
    algo.stop()
    t = [i * env.dt for i in range(len(xs))]
    voltajes = [v_norm * env.Vmax for v_norm in v_norms]
    return t, xs, phis, dxs, dphis, voltajes


print("Simulando LQR...")
t_lqr, x_lqr, phi_lqr, dx_lqr, dphi_lqr, v_lqr = run_lqr(K_LQR)

ray.init(ignore_reinit_error=True)
print("Simulando PPO LQR Reward...")
t_rl, x_rl, phi_rl, dx_rl, dphi_rl, v_rl = run_rl(CHECKPOINT_RL)
ray.shutdown()


LABELS = ['LQR', 'PPO LQR Reward']
COLORS = ['royalblue', 'darkorange']
LWS    = [2.2, 1.8]
ALPHAS = [1.0, 0.9]

datasets = [
    (t_lqr, x_lqr, phi_lqr, dx_lqr, dphi_lqr, v_lqr),
    (t_rl,  x_rl,  phi_rl,  dx_rl,  dphi_rl,  v_rl),
]

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')

env_ref = CraneEnvDiscrete()

for (t, x, phi, dx, dphi, v), label, color, lw, alpha in zip(datasets, LABELS, COLORS, LWS, ALPHAS):
    axes[0].plot(t, x,                             color=color, label=label, lw=lw, alpha=alpha)
    axes[1].plot(t, [p*180/np.pi for p in phi],   color=color, label=label, lw=lw, alpha=alpha)
    axes[2].plot(t, dx,                            color=color, label=label, lw=lw, alpha=alpha)
    axes[3].plot(t, dphi,                          color=color, label=label, lw=lw, alpha=alpha)
    axes[4].step(t, v, where='post',               color=color, label=label, lw=lw, alpha=alpha)

axes[0].axhline(env_ref.xref, color='gray', linestyle='--', lw=1.2, label='xref = 1 m')
for ax, lo, hi, label in [
    (axes[1], -np.degrees(env_ref.phi_max), np.degrees(env_ref.phi_max), f'±{np.degrees(env_ref.phi_max):.1f}°'),
    (axes[2], -env_ref.dx_limit, env_ref.dx_limit, f'±{env_ref.dx_limit} m/s'),
    (axes[3], -env_ref.dphi_limit, env_ref.dphi_limit, f'±{env_ref.dphi_limit} rad/s'),
]:
    ax.axhline(hi, color='gray', linestyle=':', lw=1.0, label=label)
    ax.axhline(lo, color='gray', linestyle=':', lw=1.0)
axes[4].axhline(0, color='gray', linestyle='--', lw=1.0)

ylabels = ['Posición x (m)', 'Ángulo φ (°)', 'Velocidad dx (m/s)', 'Vel. angular dφ (rad/s)', 'Voltaje V (V)']
for ax, ylabel in zip(axes, ylabels):
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, color='#e0e0e0', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)

axes[4].set_xlabel('Tiempo (s)', fontsize=11)
plt.suptitle('Comparación LQR vs PPO LQR Reward — CraneEnv', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('comparacion_lqr_vs_ppo_lqr.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Gráfica guardada: comparacion_lqr_vs_ppo_lqr.png")
plt.show()
