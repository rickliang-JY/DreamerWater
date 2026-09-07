#!/usr/bin/env python3
"""构建期脚本：生成 REMUS100 回归基准 tests/fixtures/remus100_reference.npz。

用法（需要网络 clone 一次，生成后 npz vendored 进仓库，测试无需网络）：

    git clone https://github.com/cybergalactic/PythonVehicleSimulator /tmp/pvs
    PYTHONPATH=/tmp/pvs/src python tests/fixtures/make_remus100_reference.py

做法：
    1. 从 python_vehicle_simulator.vehicles.remus100 提取水平面 3-DOF 子集
       （surge/sway/yaw 的 M、D 3x3 子块，即该类 self.M 的 [0,1,5]x[0,1,5]
       子块与 D = diag(M_11/T_surge, M_22/T_sway, M_33/T_yaw)）。
    2. 用与 uwm.dynamics.fossen.Fossen3DOF 完全相同的方程（numpy 实现，
       对角 M 的标准 3-DOF C 矩阵、线性 D、RK4）积分同一控制序列
       （阶跃 + 正弦，200 步，dt=0.1 s），得到参考轨迹。

免责说明（SPEC §3.2）：完整 remus100 模型含升阻力 / 横流阻力 /
D 的 exp(-3*U_r) 速度衰减 / 执行器一阶滞后等非线性耦合项，不属于
3-DOF 线性子集，参考轨迹不含这些项；因此本基准验证的是"参数提取 +
3-DOF 模型方程"与 Fossen 参考实现子集的一致性，而非完整 6-DOF 模型。
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "remus100_reference.npz")

DT = 0.1          # 参考轨迹积分步长（s）
N_STEPS = 200     # 控制序列长度


def extract_submatrices():
    """从 remus100 类提取 M、D 的 3x3 水平面子块。"""
    from python_vehicle_simulator.vehicles.remus100 import remus100

    vehicle = remus100()
    idx = [0, 1, 5]  # u, v, r
    M3 = vehicle.M[np.ix_(idx, idx)]
    D3 = np.diag(
        [
            M3[0, 0] / vehicle.T_surge,
            M3[1, 1] / vehicle.T_sway,
            M3[2, 2] / vehicle.T_yaw,
        ]
    )
    return M3, D3


def control_sequence(n_steps: int, dt: float) -> np.ndarray:
    """阶跃 + 正弦控制序列 tau(t) = [Fx, Fy, Mz]（N, N, N m），形状 (n, 3)。"""
    t = np.arange(n_steps) * dt
    tau = np.zeros((n_steps, 3))
    tau[:, 0] = 50.0 + 10.0 * np.sin(0.5 * t)   # 阶跃 50 N + 正弦
    tau[:, 1] = 5.0 * np.sin(0.3 * t)
    tau[:, 2] = 2.0 + 1.0 * np.sin(0.7 * t)     # 阶跃 2 N m + 正弦
    return tau


def rhs(eta, nu, tau, Minv, D, current):
    """与 Fossen3DOF.rhs 相同的 numpy 实现（current 为 NED [u_c, v_c]）。"""
    psi = eta[2]
    c, s = np.cos(psi), np.sin(psi)
    eta_dot = np.array([c * nu[0] - s * nu[1], s * nu[0] + c * nu[1], nu[2]])
    nu_c = np.array(
        [c * current[0] + s * current[1], -s * current[0] + c * current[1], 0.0]
    )
    nu_rel = nu - nu_c
    m11, m22 = 1.0 / Minv[0, 0], 1.0 / Minv[1, 1]
    C = np.array(
        [[0.0, 0.0, -m22 * nu_rel[1]], [0.0, 0.0, m11 * nu_rel[0]],
         [m22 * nu_rel[1], -m11 * nu_rel[0], 0.0]]
    )
    nu_dot = Minv @ (tau - C @ nu_rel - D @ nu_rel)
    return eta_dot, nu_dot


def rk4_step(eta, nu, tau, dt, Minv, D, current):
    k1e, k1v = rhs(eta, nu, tau, Minv, D, current)
    k2e, k2v = rhs(eta + 0.5 * dt * k1e, nu + 0.5 * dt * k1v, tau, Minv, D, current)
    k3e, k3v = rhs(eta + 0.5 * dt * k2e, nu + 0.5 * dt * k2v, tau, Minv, D, current)
    k4e, k4v = rhs(eta + dt * k3e, nu + dt * k3v, tau, Minv, D, current)
    return (
        eta + dt / 6.0 * (k1e + 2 * k2e + 2 * k3e + k4e),
        nu + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v),
    )


def wrap(psi):
    return np.pi - np.remainder(np.pi - psi, 2.0 * np.pi)


def main():
    M3, D3 = extract_submatrices()
    Minv = np.linalg.inv(M3)
    current = np.array([0.0, 0.0])
    tau_seq = control_sequence(N_STEPS, DT)

    eta = np.array([0.0, 0.0, 0.5])   # 初始 [x, y, psi]
    nu = np.array([0.2, -0.1, 0.05])  # 初始 [u, v, r]

    etas = np.zeros((N_STEPS + 1, 3))
    nus = np.zeros((N_STEPS + 1, 3))
    etas[0], nus[0] = eta, nu
    for k in range(N_STEPS):
        eta, nu = rk4_step(eta, nu, tau_seq[k], DT, Minv, D3, current)
        eta[2] = wrap(eta[2])
        etas[k + 1], nus[k + 1] = eta, nu

    np.savez(
        OUT,
        t=np.arange(N_STEPS + 1) * DT,
        eta=etas,
        nu=nus,
        tau=tau_seq,
        M=M3,
        D=D3,
        current=current,
        dt=np.array(DT),
        eta0=etas[0],
        nu0=nus[0],
    )
    print(f"written {OUT}")
    print("M3 =\n", M3)
    print("D3 =\n", D3)


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("PVS_SRC", "/tmp/pvs/src"))
    main()
