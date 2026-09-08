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

    def reset(self, *, seed=None, options=None):
        """播种并重置洋流（SPEC_M4 §1.3）。子类 super().reset(seed=seed) 即触发。

        常值流：reset_current 为 no-op，且不消耗 np_random（M0–M2 随机序列
        逐位不变，完全向后兼容）。OU 时变流：从 np_random 派生一个 per-episode
        种子传给 reset_current——同一 env seed 下整个 episode 序列可复现，
        不同 episode 的洋流实现又互不相同（防止策略背下单条洋流轨迹）。
        """
        super().reset(seed=seed)
        if self.fossen.ou_enabled:
            ou_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.fossen.reset_current(seed=ou_seed)
        else:
            self.fossen.reset_current()

    def current_truth(self) -> np.ndarray:
        """洋流真值 [u_c, v_c]（NED，m/s，numpy float64）。仅供记录/探针，不进 obs。"""
        return self.fossen.current.detach().cpu().numpy().astype(np.float64)

    def _physics_step(self, action: np.ndarray) -> None:
        """执行一次 env 步：推力饱和后做 action_repeat 次 RK4 物理积分。

        action: (3,) numpy 数组，∈ [-1, 1]。
        """
        tau = self.thrusters(torch.as_tensor(np.asarray(action), dtype=_DTYPE))
        for _ in range(self.action_repeat):
            self._eta, self._nu = self.fossen.step(self._eta, self._nu, tau, self.dt)
        self._t += self.control_period
        self._steps += 1
