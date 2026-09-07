"""解析解验证（SPEC §3.1）：指数衰减、定常推力终速、纯 yaw 积分、批量一致性。

说明：RK4 为数值积分，测试中采用较小 dt 以保证 <1% 误差门槛
（物理模型默认 dt=0.02 s 不变，见 yaml 配置）。
"""

import math

import pytest
import torch

from uwm.dynamics.fossen import Fossen3DOF

DT = 1e-3  # 测试用小步长（s），保证 RK4 精度余量

CFG = {
    "name": "test",
    "dof": 3,
    "mass": 10.0,
    "added_mass": [2.0, 4.0, 0.5],
    "inertia_z": 1.0,
    "linear_damping": [3.0, 6.0, 0.75],
    "tau_max": [50.0, 50.0, 10.0],
    "current": [0.0, 0.0],
    "dt": 0.02,
    "action_repeat": 5,
}


def _simulate(fossen, eta, nu, tau, dt, n_steps):
    """逐步积分，返回整条轨迹（含初始点），形状 (n+1, 3)。"""
    etas = [eta.clone()]
    nus = [nu.clone()]
    for _ in range(n_steps):
        eta, nu = fossen.step(eta, nu, tau, dt)
        etas.append(eta.clone())
        nus.append(nu.clone())
    return torch.stack(etas), torch.stack(nus)


def _fit_time_constant(t, x):
    """对数线性拟合衰减时间常数：x(t)=x0·exp(-t/tau)，返回 tau（s）。"""
    x = x / x[0]
    mask = (x > 0.05) & (x < 0.95)  # 用衰减中段，避开截断与舍入误差
    logx = torch.log(x[mask])
    slope = torch.sum((t[mask] - t[mask].mean()) * (logx - logx.mean())) / torch.sum(
        (t[mask] - t[mask].mean()) ** 2
    )
    return -1.0 / slope


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_exponential_decay_time_constant(channel):
    """ν(0)=e_channel·v0，τ=0，对角 M、D 下 v(t)=v0·exp(−D_ii/M_ii·t)。

    数值轨迹时间常数与解析值相对误差 < 1%。三个通道各测一次。
    """
    fossen = Fossen3DOF(CFG)
    tau_num = float(fossen.M[channel, channel] / fossen.D[channel, channel])
    v0 = 1.0

    eta = torch.zeros(3, dtype=torch.float64)
    nu = torch.zeros(3, dtype=torch.float64)
    nu[channel] = v0
    tau = torch.zeros(3, dtype=torch.float64)

    n_steps = int(4.0 * tau_num / DT)
    t = torch.arange(n_steps + 1, dtype=torch.float64) * DT
    _, nus = _simulate(fossen, eta, nu, tau, DT, n_steps)

    tau_fit = _fit_time_constant(t, nus[:, channel].abs())
    rel_err = abs(tau_fit - tau_num) / tau_num
    assert rel_err < 0.01, f"channel {channel}: tau_num={tau_num}, tau_fit={tau_fit}"


def test_constant_thrust_terminal_velocity():
    """恒定 τ=[Fx,0,0] → 终速 u∞ = Fx/X_u，相对误差 < 1%。"""
    fossen = Fossen3DOF(CFG)
    fx = 12.0
    u_inf = fx / float(fossen.D[0, 0])
    tau_num = float(fossen.M[0, 0] / fossen.D[0, 0])

    eta = torch.zeros(3, dtype=torch.float64)
    nu = torch.zeros(3, dtype=torch.float64)
    tau = torch.tensor([fx, 0.0, 0.0], dtype=torch.float64)

    n_steps = int(8.0 * tau_num / DT)  # 充分进入稳态
    _, nus = _simulate(fossen, eta, nu, tau, DT, n_steps)

    u_final = float(nus[-1, 0])
    rel_err = abs(u_final - u_inf) / u_inf
    assert rel_err < 0.01, f"u_final={u_final}, u_inf={u_inf}"


def test_pure_yaw_integration_and_wrap():
    """纯 yaw：阻尼与推力平衡使 r 恒定 → ψ 线性增长，且 wrap 到 (−π, π]。"""
    fossen = Fossen3DOF(CFG)
    r0 = 1.0  # rad/s
    eta = torch.zeros(3, dtype=torch.float64)
    nu = torch.tensor([0.0, 0.0, r0], dtype=torch.float64)
    # 平衡阻尼使 r 精确保持 r0（M、D 对角时 τ_r = N_r·r 为平衡点）
    tau = torch.tensor([0.0, 0.0, float(fossen.D[2, 2]) * r0], dtype=torch.float64)

    t_total = 3.0 * math.pi / r0  # ψ 走过 3π，必跨越 ±π 边界
    n_steps = int(t_total / DT)
    etas, nus = _simulate(fossen, eta, nu, tau, DT, n_steps)

    # r 恒定
    assert torch.allclose(
        nus[:, 2], torch.full_like(nus[:, 2], r0), rtol=1e-9, atol=1e-9
    )
    # ψ 线性增长并正确 wrap
    t = torch.arange(n_steps + 1, dtype=torch.float64) * DT
    psi_expected = math.pi - torch.remainder(math.pi - r0 * t, 2.0 * math.pi)
    torch.testing.assert_close(etas[:, 2], psi_expected, rtol=1e-6, atol=1e-6)
    assert bool(((etas[:, 2] > -math.pi) & (etas[:, 2] <= math.pi)).all())


def test_batch_consistency():
    """(B,3) 批量输入与逐样本循环结果完全一致。"""
    fossen = Fossen3DOF({**CFG, "current": [0.3, -0.2]})  # 带洋流，覆盖所有分支
    B = 32
    gen = torch.Generator().manual_seed(0)
    eta = torch.randn(B, 3, generator=gen, dtype=torch.float64)
    nu = torch.randn(B, 3, generator=gen, dtype=torch.float64) * 0.5
    tau = torch.randn(B, 3, generator=gen, dtype=torch.float64) * 10.0

    eta_b, nu_b = eta.clone(), nu.clone()
    eta_i, nu_i = eta.clone(), nu.clone()
    dt = 0.02
    for _ in range(50):
        eta_b, nu_b = fossen.step(eta_b, nu_b, tau, dt)
        for i in range(B):
            eta_i[i], nu_i[i] = fossen.step(eta_i[i], nu_i[i], tau[i], dt)

    torch.testing.assert_close(eta_b, eta_i, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(nu_b, nu_i, rtol=1e-12, atol=1e-12)
