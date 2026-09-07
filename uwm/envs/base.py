"""gymnasium 环境统一基类。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from uwm.dynamics.fossen import Fossen3DOF
from uwm.dynamics.thrusters import ThrusterClip

_DTYPE = torch.float64


class UWEnvBase(gym.Env):
    """统一基类。持有 Fossen3DOF + ThrusterClip。

    __init__(self, vehicle_cfg: dict, task_cfg: dict)
    物理 dt 与 action_repeat 解耦：env.step = action_repeat 次 fossen.step。

    vehicle_cfg: configs/vehicle/*.yaml 载入的字典（SPEC §2.4）。
    task_cfg:    任务参数（init_radius、max_episode_steps、奖励权重等）。
    """

    metadata = {"render_modes": []}

    def __init__(self, vehicle_cfg: dict, task_cfg: dict):
        super().__init__()
        self.vehicle_cfg = vehicle_cfg
        self.task_cfg = task_cfg
        self.fossen = Fossen3DOF(vehicle_cfg)
        self.thrusters = ThrusterClip(vehicle_cfg["tau_max"])
        self.dt = float(vehicle_cfg["dt"])  # 物理步长（s）
        self.action_repeat = int(vehicle_cfg["action_repeat"])

        # 环境状态（torch.float64）
        self._eta = torch.zeros(3, dtype=_DTYPE)
        self._nu = torch.zeros(3, dtype=_DTYPE)
        self._t = 0.0  # 累计物理时间（s）
        self._steps = 0  # 累计 env.step 次数

        # 动作空间：Box(3) ∈ [-1, 1]（SPEC §2.3）
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float64
        )

    @property
    def control_period(self) -> float:
        """控制周期（s）：dt × action_repeat。"""
        return self.dt * self.action_repeat

    def _physics_step(self, action: np.ndarray) -> None:
        """执行一次 env 步：推力饱和后做 action_repeat 次 RK4 物理积分。

        action: (3,) numpy 数组，∈ [-1, 1]。
        """
        tau = self.thrusters(torch.as_tensor(np.asarray(action), dtype=_DTYPE))
        for _ in range(self.action_repeat):
            self._eta, self._nu = self.fossen.step(self._eta, self._nu, tau, self.dt)
        self._t += self.control_period
        self._steps += 1
