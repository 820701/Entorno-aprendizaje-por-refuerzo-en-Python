import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class CraneEnvDiscrete(gym.Env):
    def __init__(self, config=None):
        super().__init__()

        self.M = 0.5
        self.m = 0.1
        self.L = 0.5
        self.g = 9.8
        self.bx = 2.65
        self.Kt = 1.34
        self.dt = 0.02
        self.Vmax = 6
        self.state = None

        # Valores límite
        self.x_max = 10
        self.phi_max = 0.5  # 29º
        self.dx_max = 5
        self.dphi_max = 5

        self.xref = 1
        self.phiref = 0
        self.dxref = 0
        self.dphiref = 0

        # Pesos de la reward
        self.w_x = 10
        self.w_phi = 1
        self.w_dx = 6
        self.w_dphi = 8

        # Número de pasos
        self.max_step = 500
        self.current_step = 0

        # Valores de tolerancia límites
        self.x_limit = 0.1
        self.dx_limit = 0.5  # buscamos que el carro esté casi quieto
        self.dphi_limit = 0.2

        obs_low = np.array(
            [-self.x_max, -self.dx_max, -self.phi_max, -self.dphi_max],
            dtype=np.float32
        )
        obs_high = np.array(
            [self.x_max, self.dx_max, self.phi_max, self.dphi_max],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.n_acciones = 13
        self.action_space = spaces.Discrete(self.n_acciones)

        self.last_action = 6  # acción central, V = 0V
        self.voltaje_buffer = deque(maxlen=3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        x = 0.0
        dx = 0.0
        phi = self.np_random.uniform(-0.05, 0.05)
        dphi = 0.0

        self.last_action = 6
        self.voltaje_buffer.clear()

        self.state = np.array([x, dx, phi, dphi], dtype=np.float32)
        self.current_step = 0

        return self.state, {}

    def normalizar_error_x(self):
        return max(0.0, 1 - abs(self.state[0] - self.xref) / self.x_max)

    def normalizar_error_dx(self):
        return max(0.0, 1 - abs(self.state[1] - self.dxref) / self.dx_max)

    def normalizar_error_phi(self):
        return max(0.0, 1 - abs(self.state[2] - self.phiref) / self.phi_max)

    def normalizar_error_dphi(self):
        return max(0.0, 1 - abs(self.state[3] - self.dphiref) / self.dphi_max)

    def step(self, action):
        self.current_step += 1
        x, dx, phi, dphi = self.state

        action = int(np.clip(action, self.last_action - 1, self.last_action + 1))
        V_raw = (action / (self.n_acciones - 1) * 2 - 1) * self.Vmax
        self.voltaje_buffer.append(V_raw)
        V = float(np.mean(self.voltaje_buffer))
        F = self.Kt * V

        ddx = (F / self.M) - (self.bx / self.M) * dx + (self.m * self.g / self.M) * phi
        ddphi = -(self.g / self.L) * phi - (1 / self.L) * ddx

        dx = dx + self.dt * ddx
        x = x + self.dt * dx
        dphi = dphi + self.dt * ddphi
        phi = phi + self.dt * dphi

        self.state = np.array([x, dx, phi, dphi], dtype=np.float32)

        error_x = x - self.xref
        error_phi = phi - self.phiref
        error_dx = dx - self.dxref
        error_dphi = dphi - self.dphiref

        reward = -(self.w_x * error_x**2 + self.w_phi * error_phi**2 + self.w_dx * error_dx**2 + self.w_dphi * error_dphi**2)

        dif_action = abs(action - self.last_action)
        self.last_action = action

        reward -= 0.25 * dif_action**2
        reward -= 0.1 * abs(dif_action)
        reward -= 0.07 * abs(V)

        if abs(error_x) < 0.2 and abs(error_dphi) < 0.3:
            reward += 2

        if abs(error_x) < 0.15:
            reward -= 4 * abs(dx)
            reward -= 2 * abs(dphi)

        reward += 2.0 * max(0.0, 1.0 - abs(error_x) / 1.0)

        if abs(error_x) < 0.08 and abs(dphi) < 0.1 and abs(dx) < 0.1:
            reward += 3

        
        done = bool(abs(error_x) < self.x_limit and abs(error_dphi) < self.dphi_limit and abs(dx) < self.dx_limit)

        fin_fisico = bool(abs(x) > self.x_max or abs(phi) > self.phi_max or abs(dphi) > self.dphi_max)

        terminated = fin_fisico
        truncated = bool(self.current_step >= self.max_step)

        if fin_fisico:
            reward -= 500  # penalización por superar los límites físicos

        if done:
            reward += 100  # bonus por alcanzar objetivo

        info = {}
        if fin_fisico:
            info["termination_cause"] = "fin_fisico"
        elif done:
            info["termination_cause"] = "objetivo_alcanzado"
        elif truncated:
            info["termination_cause"] = "truncated"

        return self.state, reward, terminated, truncated, info