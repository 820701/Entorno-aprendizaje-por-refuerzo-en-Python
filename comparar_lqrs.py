import ray
import numpy as np
import matplotlib.pyplot as plt
import torch
from gymnasium import spaces
from scipy.linalg import solve_continuous_are
from ray.rllib.algorithms.ppo import PPOConfig
from crane_env_discrete import CraneEnvDiscrete


class CraneEnvDiscrete4(CraneEnvDiscrete):
    def __init__(self, config=None):
        super().__init__(config)
        obs_low  = np.array([-self.x_max, -self.dx_max, -self.phi_max, -self.dphi_max], dtype=np.float32)
        obs_high = np.array([ self.x_max,  self.dx_max,  self.phi_max,  self.dphi_max], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs[:4], info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs[:4], reward, terminated, truncated, info


M=0.5; m=0.1; l=0.5; bx=2.65; Kt=1.34; g=9.8
A = np.array([[0,1,0,0],[0,-bx/M,m*g/M,0],[0,0,0,1],[0,bx/(M*l),-(g/l)*((m/M)+1),0]])
B = np.array([[0],[Kt/M],[0],[-Kt/(M*l)]])
Q = np.diag([7.0, 0.1, 1.0, 9.0])
R = np.array([[1.0]])
P = solve_continuous_are(A, B, Q, R)
K_LQR = ((1/R[0,0]) * B.T @ P)[0]

CHECKPOINT_RL = r"C:\Users\josem\OneDrive\Escritorio\2025-2026\TFG\RL\tfg_rl_crane\checkpoints\crane_ppo_discrete - DEFINITIVO"


def run_lqr(K):
    env = CraneEnvDiscrete()
    obs, _ = env.reset()
    xs, phis, dxs, dphis, voltajes = [], [], [], [], []
    for _ in range(500):
        x, dx, phi, dphi = obs[:4]
        state_error = np.array([x - env.xref, dx, phi, dphi])
        V = float(np.clip(-K @ state_error, -env.Vmax, env.Vmax))
        F = env.Kt * V
        ddx   = (F / env.M) - (env.bx / env.M)*dx + (env.m*env.g / env.M)*phi
        ddphi = -(env.g / env.L)*phi - (1/env.L)*ddx
        dx   = dx   + env.dt * ddx
        x    = x    + env.dt * dx
        dphi = dphi + env.dt * ddphi
        phi  = phi  + env.dt * dphi
        obs = np.array([x, dx, phi, dphi], dtype=np.float32)
        xs.append(x); dxs.append(dx); phis.append(phi); dphis.append(dphi); voltajes.append(V)
        fin = bool(abs(x) > env.x_max or abs(phi) > env.phi_max or abs(dphi) > env.dphi_max)
        if fin:
            break
    t = [i * env.dt for i in range(len(xs))]
    return t, xs, phis, dxs, dphis, voltajes


def run_rl(checkpoint_path, env_class):
    config = PPOConfig().environment(env=env_class).env_runners(num_env_runners=0)
    algo = config.build()
    algo.restore(checkpoint_path)
    rl_module = algo.get_module()
    env = env_class()
    obs, _ = env.reset()
    xs, phis, dxs, dphis, actions = [], [], [], [], []
    for _ in range(500):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = rl_module.forward_inference({"obs": obs_tensor})
        action = int(np.argmax(output["action_dist_inputs"].numpy()[0]))
        obs, _, terminated, truncated, _ = env.step(action)
        xs.append(obs[0]); dxs.append(obs[1]); phis.append(obs[2]); dphis.append(obs[3])
        actions.append(action)
        if terminated or truncated:
            break
    algo.stop()
    t = [i * env.dt for i in range(len(xs))]
    voltajes = [(a / (env.n_acciones - 1) * 2 - 1) * env.Vmax for a in actions]
    return t, xs, phis, dxs, dphis, voltajes


print("Simulando LQR...")
t_lqr, x_lqr, phi_lqr, dx_lqr, dphi_lqr, v_lqr = run_lqr(K_LQR)

ray.init(ignore_reinit_error=True)
print("Simulando PPO Discreto...")
t_rl, x_rl, phi_rl, dx_rl, dphi_rl, v_rl = run_rl(CHECKPOINT_RL, CraneEnvDiscrete4)
ray.shutdown()


LABELS = ['LQR', 'PPO Discreto']
COLORS = ['royalblue', 'crimson']
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
plt.suptitle('Comparación LQR vs PPO Discreto — CraneEnv', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('comparacion_lqr_vs_ppo.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Gráfica guardada: comparacion_lqr_vs_ppo.png")
plt.show()