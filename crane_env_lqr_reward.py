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

        self.x_max = 10
        self.phi_max = 0.5
        self.dx_max = 5
        self.dphi_max = 5

        self.xref = 1
        self.max_step = 500
        self.current_step = 0

        self.x_limit = 0.1
        self.dx_limit = 0.2
        self.dphi_limit = 0.2

        self.Q = np.diag([7.0, 0.1, 1.0, 9.0])
        self.R = 1.0

        obs_low  = np.array([-self.x_max, -self.dx_max, -self.phi_max, -self.dphi_max, -1.0], dtype=np.float32)
        obs_high = np.array([ self.x_max,  self.dx_max,  self.phi_max,  self.dphi_max,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.n_acciones = 13
        self.action_space = spaces.Discrete(self.n_acciones)

        self.last_action = 6
        self.last_V = 0.0

        self.voltaje_buffer = deque(maxlen=3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        x = 0.0
        dx = 0.0
        phi = self.np_random.uniform(-0.05, 0.05)
        dphi = 0.0
        self.last_V = 0.0

        self.state = np.array([x, dx, phi, dphi, 0.0], dtype=np.float32)
        self.current_step = 0

        self.last_action = 6

        self.voltaje_buffer.clear()

        return self.state, {}

    def step(self, action):
        self.current_step += 1
        x, dx, phi, dphi, _ = self.state

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

        V_norm = V / self.Vmax
        self.state = np.array([x, dx, phi, dphi, V_norm], dtype=np.float32)

        state_error = np.array([x - self.xref, dx, phi, dphi])
        reward = -(state_error @ self.Q @ state_error + self.R * V**2) * self.dt

        error_x = state_error[0]
        error_dphi = dphi

        done = bool(abs(error_x) < self.x_limit and abs(error_dphi) < self.dphi_limit and abs(dx) < self.dx_limit)
        fin_fisico = bool(abs(x) > self.x_max or abs(phi) > self.phi_max or abs(dphi) > self.dphi_max)
        terminated = fin_fisico
        truncated = bool(self.current_step >= self.max_step)

        if fin_fisico:
            reward -=200.0
        
        if abs(error_x) < 0.6:
            reward += 6 * max(0, 1 - abs(dx) / self.dx_limit) \
            + 6 * max(0, 1 - abs(phi) / np.radians(5)) \
            +3 * max(0, 1 - abs(error_x) / 0.4) 

        if abs(error_x) >= 0.4:
            reward -= 1.5 * abs(error_x)

        
        #penalizamos las diferencias de voltaje de un paso a otro
        dif_action = abs(action - self.last_action)
        self.last_action = action
        reward -= 0.25*dif_action**2

    
        reward -=2*(x - self.xref)**2
        #reward -=0.3*dx**2    
        reward -=2*phi**2
        reward -=2*dphi**2

        reward-=0.07*abs(V)

        if abs(error_x) < 0.1 and abs(phi) < np.radians(8) and abs(dx) < 0.25 and abs(dphi) < 0.2:
            reward += 3
    
        info = {}
        if fin_fisico:
            info["termination_cause"] = "fin_fisico"
        elif done:
            info["termination_cause"] = "objetivo_alcanzado"
        elif truncated:
            info["termination_cause"] = "truncated"
            
        
        return self.state, reward, terminated, truncated, info

    def step_with_voltage(self, V):
        """Acepta voltaje continuo filtrado, sin pasar por la discretización."""
        self.current_step += 1
        x, dx, phi, dphi, _ = self.state

        V = np.clip(V, -self.Vmax, self.Vmax)
        F = self.Kt * V

        ddx   = (F / self.M) - (self.bx / self.M) * dx + (self.m * self.g / self.M) * phi
        ddphi = -(self.g / self.L) * phi - (1 / self.L) * ddx

        dx   = dx   + self.dt * ddx
        x    = x    + self.dt * dx
        dphi = dphi + self.dt * ddphi
        phi  = phi  + self.dt * dphi

        V_norm = V / self.Vmax
        self.state = np.array([x, dx, phi, dphi, V_norm], dtype=np.float32)

        fin_fisico = bool(abs(x) > self.x_max or abs(phi) > self.phi_max or abs(dphi) > self.dphi_max)
        truncated  = bool(self.current_step >= self.max_step)
        terminated = fin_fisico

        info = {}
        if fin_fisico:
            info["termination_cause"] = "fin_fisico"
        elif truncated:
            info["termination_cause"] = "truncated"

        return self.state, 0.0, terminated, truncated, info