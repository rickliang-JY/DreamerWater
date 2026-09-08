"""3-DOF 水平面 Fossen 刚体动力学模型（torch, float64）。

状态约定：
    eta = [x, y, psi]  NED 北东地坐标，psi 为偏航角（rad，wrap 到 (-pi, pi]）
    nu  = [u, v, r]    机体系速度（u, v 单位 m/s，r 单位 rad/s）
    tau = [Fx, Fy, Mz] 机体系广义力（N, N, N m）

动力学方程：
    eta_dot = J(psi) nu
    M nu_dot + C(nu_rel) nu_rel + D(nu_rel) nu_rel = tau

    D(nu) = D_lin + D_quad |nu|（对角二次阻尼，v0.1 加入）。
    v0.1 变更原因：仅线性阻尼时 yaw 通道终端转速 ~171 rad/s（N_r=0.07 过弱），
    Coriolis 旋转耦合（纯虚特征值）在显式 RK4 下持续放大，饱和动作约 9.4s
    后数值爆炸出 NaN（M2 DreamerV3 训练中三次复现）。二次阻尼把终端转速
    压回 ~3.8 rad/s 的物理区间，同时增大衰减率使 RK4 稳定。详见 SPEC_M2 勘误 4。

其中 nu_rel = nu - nu_c^body 为相对水流速度。v0 采用 Fossen 常值无旋流
简化：惯性项不做精确流加速修正，C、D 中统一使用 nu_rel（见 SPEC §2.1）。

v0.2（M4，SPEC_M4 §1.1）：cfg["current"] 除列表形式（常值流，行为不变）外，
支持 Ornstein–Uhlenbeck 时变流字典形式：
    current: {type: ou, mean: [u_c, v_c], theta: 回归速率 (1/s),
              sigma: 扩散强度 (m/s/√s), seed: 独立随机流种子}
OU 状态存于 self.ou_state，在每个物理步 step(dt) 内按 Euler–Maruyama 推进：
    C ← C + theta*(mean−C)*dt + sigma*√dt*ξ,  ξ~N(0, I₂)
（用的是物理 dt，不是 env step；env.step = action_repeat 次 fossen.step，
洋流随每个物理子步演化）。随机流用独立 torch.Generator，与全局 RNG 及
env 的 np_random 互不混杂，保证 seed 级可复现。reset_current() 由
env.reset 调用：状态回到 mean 并重建随机流。下游经 self.current 读当前
C 值（属性语义不变），零改动。
"""

from __future__ import annotations

import math

import torch

_DTYPE = torch.float64


def _wrap_angle(psi: torch.Tensor) -> torch.Tensor:
    """把角度 wrap 到 (-pi, pi]（单位 rad）。"""
    return math.pi - torch.remainder(math.pi - psi, 2.0 * math.pi)


class Fossen3DOF:
    """torch 实现的 3-DOF Fossen 刚体模型。全部参数为 torch.float64 张量。

    动力学: M ν̇ + C(ν)ν + D(ν)ν = τ + τ_current
    v0: D 只含线性项; 洋流为常值（相对速度进入阻尼项）。

    cfg 字段（见 configs/vehicle/*.yaml，SPEC §2.4）：
        mass (kg), added_mass [3] (|X_udot|,|Y_vdot|,|N_rdot|, 正数),
        inertia_z (kg m^2), linear_damping [3] (X_u,Y_v,N_r, 正数),
        current [2] (u_c, v_c, m/s, NED) 或 OU 字典（v0.2，见模块 docstring）。
    """

    def __init__(self, cfg: dict):
        mass = float(cfg["mass"])
        am = [float(a) for a in cfg["added_mass"]]
        iz = float(cfg["inertia_z"])
        ld = [float(d) for d in cfg["linear_damping"]]

        # M = M_RB + M_A，均为对角阵（载体三面对称假设）
        self.M = torch.tensor(
            [
                [mass + am[0], 0.0, 0.0],
                [0.0, mass + am[1], 0.0],
                [0.0, 0.0, iz + am[2]],
            ],
            dtype=_DTYPE,
        )
        self.Minv = torch.linalg.inv(self.M)
        # D 线性阻尼对角阵（正值）
        self.D = torch.diag(torch.tensor(ld, dtype=_DTYPE))
        # 二次阻尼系数（v0.1；缺省为 0 = 纯线性，兼容旧配置）
        qd = [float(q) for q in cfg.get("quadratic_damping", [0.0, 0.0, 0.0])]
        self.Dq = torch.tensor(qd, dtype=_DTYPE)

        # 洋流（v0.2）：列表 [u_c, v_c] = 常值流（v0 行为，不变）；
        # 字典 = OU 时变流（SPEC_M4 §1.1）。
        current_cfg = cfg.get("current", [0.0, 0.0])
        if isinstance(current_cfg, dict):
            if current_cfg.get("type", "ou") != "ou":
                raise ValueError(f"未知 current 类型: {current_cfg.get('type')!r}")
            self.ou_enabled = True
            self.ou_mean = torch.tensor(
                [float(c) for c in current_cfg["mean"]], dtype=_DTYPE
            )
            self.ou_theta = float(current_cfg["theta"])
            self.ou_sigma = float(current_cfg["sigma"])
            self._ou_seed = int(current_cfg.get("seed", 0))
            self.ou_state = self.ou_mean.clone()
            self._ou_gen: torch.Generator | None = None
            self.reset_current()
            self._current_const: torch.Tensor | None = None
        else:
            self.ou_enabled = False
            # NED 常值洋流速度 [u_c, v_c]（m/s）
            self._current_const = torch.tensor(
                [float(c) for c in current_cfg], dtype=_DTYPE
            )
            self.ou_mean = None
            self.ou_theta = 0.0
            self.ou_sigma = 0.0
            self._ou_seed = 0
            self.ou_state = None
            self._ou_gen = None

    @property
    def current(self) -> torch.Tensor:
        """当前洋流速度 C = [u_c, v_c]（NED，m/s）。常值或 OU 瞬时值，语义不变。"""
        return self.ou_state if self.ou_enabled else self._current_const

    @current.setter
    def current(self, value) -> None:
        """常值流模式下允许直接赋值（如交互面板滑块）；OU 模式改的是瞬时值。"""
        value = torch.as_tensor(value, dtype=_DTYPE)
        if self.ou_enabled:
            self.ou_state = value.clone()
        else:
            self._current_const = value

    def reset_current(self, seed: int | None = None) -> None:
        """重置洋流（env.reset 时调用）。常值流为 no-op（向后兼容）。

        OU 模式：状态回到 mean，并用 seed（缺省用配置的 seed）重建独立
        torch.Generator 随机流——同 seed 的后续洋流序列逐位可复现。
        """
        if not self.ou_enabled:
            return
        self.ou_state = self.ou_mean.clone()
        self._ou_gen = torch.Generator()
        self._ou_gen.manual_seed(self._ou_seed if seed is None else int(seed))

    def _advance_current(self, dt: float) -> None:
        """OU 的 Euler–Maruyama 推进一个物理步 dt（常值流为 no-op）。"""
        if not self.ou_enabled:
            return
        xi = torch.randn(2, generator=self._ou_gen, dtype=_DTYPE)
        self.ou_state = (
            self.ou_state
            + self.ou_theta * (self.ou_mean - self.ou_state) * dt
            + self.ou_sigma * math.sqrt(dt) * xi
        )

    def C(self, nu: torch.Tensor) -> torch.Tensor:
        """Coriolis/向心矩阵 C(nu)，形状 (..., 3, 3)。

        nu: (..., 3)，单位 [m/s, m/s, rad/s]。
        对对角 M = diag(m11, m22, m33) 的标准 3-DOF 形式：
            C = [[0, 0, -m22 v], [0, 0, m11 u], [m22 v, -m11 u, 0]]
        """
        u = nu[..., 0]
        v = nu[..., 1]
        zero = torch.zeros_like(u)
        m11 = self.M[0, 0]
        m22 = self.M[1, 1]
        row0 = torch.stack([zero, zero, -m22 * v], dim=-1)
        row1 = torch.stack([zero, zero, m11 * u], dim=-1)
        row2 = torch.stack([m22 * v, -m11 * u, zero], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    def _current_body(self, psi: torch.Tensor) -> torch.Tensor:
        """把 NED 常值洋流转换到机体系：nu_c^body = J(psi)^{-1} [u_c, v_c, 0]。

        psi: (...)，返回 (..., 3)，单位 m/s（角速度通道为 0）。
        """
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        u_c = cos_psi * self.current[0] + sin_psi * self.current[1]
        v_c = -sin_psi * self.current[0] + cos_psi * self.current[1]
        zero = torch.zeros_like(u_c)
        return torch.stack([u_c, v_c, zero], dim=-1)

    def rhs(
        self, eta: torch.Tensor, nu: torch.Tensor, tau: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (η̇, ν̇)。

        η̇ = J(ψ)ν；ν̇ = M⁻¹(τ − C(ν_rel)ν_rel − D ν_rel)。
        eta: (..., 3) [m, m, rad]；nu: (..., 3) [m/s, m/s, rad/s]；
        tau: (..., 3) [N, N, N m]。支持批量。
        """
        psi = eta[..., 2]
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        # J(psi) nu：仅偏航的 3-DOF 旋转
        x_dot = cos_psi * nu[..., 0] - sin_psi * nu[..., 1]
        y_dot = sin_psi * nu[..., 0] + cos_psi * nu[..., 1]
        psi_dot = nu[..., 2]
        eta_dot = torch.stack([x_dot, y_dot, psi_dot], dim=-1)

        # 相对水流速度（v0 简化：ν_rel 同时进入 C 和 D，见模块 docstring）
        nu_rel = nu - self._current_body(psi)
        coriolis = (self.C(nu_rel) @ nu_rel.unsqueeze(-1)).squeeze(-1)
        damping = (self.D @ nu_rel.unsqueeze(-1)).squeeze(-1)
        # D(nu)nu = D_lin nu + D_q |nu| * nu（逐分量二次项，对角假设）
        damping = damping + self.Dq * nu_rel.abs() * nu_rel
        nu_dot = (self.Minv @ (tau - coriolis - damping).unsqueeze(-1)).squeeze(-1)
        return eta_dot, nu_dot

    def step(
        self,
        eta: torch.Tensor,
        nu: torch.Tensor,
        tau: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RK4 积分一个物理步。

        eta/nu/tau 形状 (..., 3)，支持批量；dt 单位 s。
        返回新 (eta, nu)，psi wrap 到 (-pi, pi]。
        """
        k1e, k1v = self.rhs(eta, nu, tau)
        k2e, k2v = self.rhs(eta + 0.5 * dt * k1e, nu + 0.5 * dt * k1v, tau)
        k3e, k3v = self.rhs(eta + 0.5 * dt * k2e, nu + 0.5 * dt * k2v, tau)
        k4e, k4v = self.rhs(eta + dt * k3e, nu + dt * k3v, tau)
        eta_new = eta + dt / 6.0 * (k1e + 2.0 * k2e + 2.0 * k3e + k4e)
        nu_new = nu + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        psi = _wrap_angle(eta_new[..., 2])
        eta_new = torch.cat([eta_new[..., :2], psi.unsqueeze(-1)], dim=-1)
        # OU 时变洋流：每个物理 dt 推进一步（常值流为 no-op）。C 在单个
        # RK4 子步内视为常值，子步间按 Euler–Maruyama 演化（SPEC_M4 §1.1）。
        self._advance_current(dt)
        return eta_new, nu_new
