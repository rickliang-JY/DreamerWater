"""推力模型：v0 仅做逐分量 clip 饱和（无分配矩阵、无滞后）。"""

from __future__ import annotations

from typing import Sequence

import torch

_DTYPE = torch.float64


class ThrusterClip:
    """v0：直接输出 3 维广义力，逐分量 clip 到 ±tau_max。

    tau_max: [Fx_max, Fy_max, Mz_max]，单位 [N, N, N m]。
    """

    def __init__(self, tau_max: Sequence[float]):
        self.tau_max = torch.tensor(list(tau_max), dtype=_DTYPE)

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        """action ∈ [−1, 1]^3（环境动作空间），返回 τ ∈ R^3（[N, N, N m]）。"""
        action = action.to(_DTYPE)
        return torch.clamp(action, -1.0, 1.0) * self.tau_max
