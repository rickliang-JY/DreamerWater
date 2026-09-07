"""二次阻尼（v0.1）物理正确性与数值稳定性测试（SPEC_M2 勘误 4）。

背景：纯线性阻尼 v0 模型在饱和动作下 RK4 数值爆炸（Coriolis 旋转耦合
在显式积分器下持续放大），M2 训练中三次复现 NaN。本测试锁定修复。
"""

import numpy as np
import torch
import yaml

from uwm.dynamics.fossen import Fossen3DOF
from uwm.envs.station_keeping import StationKeepingEnv


def _cfg():
    with open("configs/vehicle/bluerov2.yaml") as f:
        return yaml.safe_load(f)


def test_quadratic_terminal_velocity():
    """二次阻尼终端速度：F = X_u·u + X_q·u² → u∞ 解析解对照 <2%。"""
    cfg = _cfg()
    model = Fossen3DOF(cfg)
    fx = cfg["tau_max"][0]
    xu, xq = cfg["linear_damping"][0], cfg["quadratic_damping"][0]
    u_inf = (-xu + np.sqrt(xu**2 + 4 * xq * fx)) / (2 * xq)  # 正根

    eta = torch.zeros(3, dtype=torch.float64)
    nu = torch.zeros(3, dtype=torch.float64)
    tau = torch.tensor([fx, 0.0, 0.0], dtype=torch.float64)
    for _ in range(int(60.0 / cfg["dt"])):  # 60 s 充分到达稳态
        eta, nu = model.step(eta, nu, tau, cfg["dt"])
    u_num = float(nu[0])
    assert abs(u_num - u_inf) / u_inf < 0.02, (u_num, u_inf)
    # 物理合理性：BlueROV2 满推前进应在 1.5–2.5 m/s 量级
    assert 1.0 < u_num < 3.0


def test_quadratic_yaw_terminal_bounded():
    """yaw 满转矩终端转速回到物理区间（<6 rad/s，纯线性时为 ~171）。"""
    cfg = _cfg()
    model = Fossen3DOF(cfg)
    eta = torch.zeros(3, dtype=torch.float64)
    nu = torch.zeros(3, dtype=torch.float64)
    tau = torch.tensor([0.0, 0.0, cfg["tau_max"][2]], dtype=torch.float64)
    for _ in range(int(30.0 / cfg["dt"])):
        eta, nu = model.step(eta, nu, tau, cfg["dt"])
    r_inf = float(abs(nu[2]))
    assert 1.0 < r_inf < 6.0, r_inf


def test_saturated_actions_no_nan():
    """饱和动作压测：v0 的 NaN 场景（满推 9.4s 爆炸）必须不再发生。"""
    env = StationKeepingEnv(vehicle_cfg=_cfg())
    for act in (
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
    ):
        obs, _ = env.reset(seed=0)
        for t in range(500):  # 50 s（v0 在 9.4 s 爆炸）
            obs, _, _, _, _ = env.step(act)
            assert np.all(np.isfinite(obs)), f"NaN with action {act} at step {t}"
        assert np.abs(obs[:2]).max() < 1e3, "位置发散"


def test_backward_compatible_linear_only():
    """不配 quadratic_damping 时退化为 v0 纯线性（指数衰减解析解仍成立）。"""
    cfg = {
        "mass": 10.0,
        "added_mass": [1.0, 2.0, 0.5],
        "inertia_z": 1.0,
        "linear_damping": [2.0, 4.0, 0.5],
        "current": [0.0, 0.0],
        "dt": 0.02,
    }  # 无 quadratic_damping 键
    model = Fossen3DOF(cfg)
    assert float(model.Dq.abs().max()) == 0.0
    m11 = 11.0
    tau_c = m11 / 2.0
    eta = torch.zeros(3, dtype=torch.float64)
    nu = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    t = 0.0
    for _ in range(int(3 * tau_c / 0.02)):
        eta, nu = model.step(eta, nu, torch.zeros(3, dtype=torch.float64), 0.02)
        t += 0.02
    u_expected = np.exp(-t / tau_c)
    assert abs(float(nu[0]) - u_expected) / u_expected < 0.01
