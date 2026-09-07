"""REMUS100 回归测试（SPEC §3.2）：与 PythonVehicleSimulator 对齐。

基准文件 tests/fixtures/remus100_reference.npz 由构建期脚本
tests/fixtures/make_remus100_reference.py 生成（vendored，测试无需网络）：
从 cybergalactic/PythonVehicleSimulator 的 remus100 模块提取 M、D 的
3x3 水平面子块，用与本项目相同的 3-DOF 方程（numpy RK4）生成参考轨迹。

取舍说明（SPEC §3.2 免责条款）：完整 remus100 模型的升阻力 / 横流阻力 /
D 的 exp(-3*U_r) 速度衰减 / 执行器动力学等非线性耦合项不在 3-DOF 线性
子集内，本测试对齐的是 M/D 子块 + 标准 3-DOF 方程，不含上述非线性项。
"""

import os

import numpy as np
import torch
import yaml

from uwm.dynamics.fossen import Fossen3DOF

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, "fixtures", "remus100_reference.npz")
CFG_PATH = os.path.join(REPO_ROOT, "configs", "vehicle", "remus100.yaml")


def _load():
    ref = np.load(FIXTURE)
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    return ref, cfg


def test_mass_damping_matrices_match():
    """remus100.yaml 导出的 M、D 与参考模块提取值逐项相对误差 < 1%。"""
    ref, cfg = _load()
    fossen = Fossen3DOF(cfg)

    for name, ours, ref_m in [
        ("M", fossen.M.numpy(), ref["M"]),
        ("D", fossen.D.numpy(), ref["D"]),
    ]:
        # 非零项按相对误差 < 1%；零项（对角结构的非对角元）按绝对误差核对
        nonzero = np.abs(ref_m) > 1e-12
        rel = np.abs(ours[nonzero] - ref_m[nonzero]) / np.abs(ref_m[nonzero])
        assert float(rel.max()) < 0.01, f"{name} max rel err = {rel.max()}"
        abs_err = np.abs(ours[~nonzero] - ref_m[~nonzero])
        assert float(abs_err.max(initial=0.0)) < 1e-12, (
            f"{name} off-diagonal abs err = {abs_err.max(initial=0.0)}"
        )


def test_trajectory_matches_reference():
    """torch 3-DOF 模型（remus100.yaml）与参考轨迹逐点对齐，相对误差 < 5%。"""
    ref, cfg = _load()
    fossen = Fossen3DOF(cfg)
    dt = float(ref["dt"])

    eta = torch.as_tensor(ref["eta0"].copy())
    nu = torch.as_tensor(ref["nu0"].copy())
    tau_seq = torch.as_tensor(ref["tau"])

    etas, nus = [eta.clone()], [nu.clone()]
    for k in range(tau_seq.shape[0]):
        eta, nu = fossen.step(eta, nu, tau_seq[k], dt)
        etas.append(eta.clone())
        nus.append(nu.clone())
    etas = torch.stack(etas).numpy()
    nus = torch.stack(nus).numpy()

    # 位置量与速度量分别做归一化相对误差（按各轨迹幅值归一）
    for name, ours, reftraj, scale in [
        ("eta", etas, ref["eta"], np.abs(ref["eta"]).max(axis=0)),
        ("nu", nus, ref["nu"], np.abs(ref["nu"]).max(axis=0)),
    ]:
        rel = np.abs(ours - reftraj) / scale
        assert float(rel.max()) < 0.05, f"{name} max rel err = {rel.max()}"
