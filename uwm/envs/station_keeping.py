"""定点悬停（station keeping）任务环境。"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch

from uwm.envs.base import UWEnvBase


def _symlog(x: float) -> float:
    """symlog(x) = sign(x)·log1p(|x|)，x 单位 m。"""
    return math.copysign(math.log1p(abs(x)), x)


class StationKeepingEnv(UWEnvBase):
    """目标：从随机初始偏移出发，回到原点并保持。

    observation: Box(7): [x, y, ψ, u, v, r, dist_to_goal]  （state 模式，≤10 维）
        单位 [m, m, rad, m/s, m/s, rad/s, m]。
    action: Box(3) ∈ [−1,1]
    reward: −w_pos·symlog(||p−p_goal||) − w_ctrl·||a||²，w_pos=1.0, w_ctrl=0.01
    terminated: False（v0 无终止）；truncated: t >= max_episode_steps
    reset: 初始位置在半径 init_radius 内均匀采样，ψ 均匀，ν=0

    task_cfg 字段（均有默认值）：
        init_radius (m, 默认 5.0)、max_episode_steps（默认 500）、
        w_pos（默认 1.0）、w_ctrl（默认 0.01）。目标固定为原点。
    """

    def __init__(self, vehicle_cfg: dict, task_cfg: dict | None = None):
        task_cfg = dict(task_cfg or {})
        super().__init__(vehicle_cfg, task_cfg)
        self.init_radius = float(task_cfg.get("init_radius", 5.0))
        self.max_episode_steps = int(task_cfg.get("max_episode_steps", 500))
        self.w_pos = float(task_cfg.get("w_pos", 1.0))
        self.w_ctrl = float(task_cfg.get("w_ctrl", 0.01))

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
        )

    def _get_obs(self) -> np.ndarray:
        """返回观测 [x, y, ψ, u, v, r, dist_to_goal]（numpy float64, shape (7,)）。"""
        dist = float(np.linalg.norm(self._eta[:2].numpy()))
        return np.concatenate([self._eta.numpy(), self._nu.numpy(), [dist]])

    def reset(self, *, seed=None, options=None):
        """重置环境：位置在 init_radius 圆盘内均匀采样，ψ ∈ (−π, π] 均匀，ν=0。"""
        super().reset(seed=seed)
        # 圆盘均匀采样：r = R·sqrt(u1)，θ = 2π·u2
        radius = self.init_radius * math.sqrt(self.np_random.uniform())
        angle = self.np_random.uniform(0.0, 2.0 * math.pi)
        psi = self.np_random.uniform(-math.pi, math.pi)
        self._eta = np.array(
            [radius * math.cos(angle), radius * math.sin(angle), psi],
            dtype=np.float64,
        )
        import torch

        self._eta = torch.as_tensor(self._eta)
        self._nu = torch.zeros(3, dtype=torch.float64)
        self._t = 0.0
        self._steps = 0
        return self._get_obs(), {"t": self._t}

    def step(self, action):
        """推进一个控制周期（dt × action_repeat 秒），返回 gymnasium 五元组。"""
        action = np.asarray(action, dtype=np.float64)
        self._physics_step(action)

        dist = float(np.linalg.norm(self._eta[:2].numpy()))
        reward = -self.w_pos * _symlog(dist) - self.w_ctrl * float(
            np.sum(np.clip(action, -1.0, 1.0) ** 2)
        )
        terminated = False  # v0 无终止
        truncated = self._steps >= self.max_episode_steps
        obs = self._get_obs()
        info = {"t": self._t, "dist_to_goal": dist}
        return obs, reward, terminated, truncated, info
