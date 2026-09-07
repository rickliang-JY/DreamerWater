"""marimo 面板 D：BlueROV2 3-DOF 动力学交互 playground（SPEC §4）。

只读 configs/vehicle/bluerov2.yaml，不 import 任何训练代码。
滑块：线性阻尼三通道缩放（0.1x–10x）、附加质量缩放、洋流强度（0–1 m/s）、
推力限幅缩放。输出：阶跃推力下 u/v/r 响应曲线（标注解析时间常数
tau = m/d 参考线）+ 2D 俯视轨迹。

运行：marimo edit notebooks/dynamics_playground.py
     marimo run  notebooks/dynamics_playground.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell
def _():
    from pathlib import Path

    import yaml

    # 仓库根目录：marimo 单元中 __file__ 可能不存在，回退到当前工作目录
    try:
        root = Path(__file__).resolve().parents[1]
    except NameError:
        root = Path.cwd()
    with open(root / "configs" / "vehicle" / "bluerov2.yaml") as f:
        vehicle_cfg = yaml.safe_load(f)
    return (vehicle_cfg,)


@app.cell
def _(mo):
    s_xu = mo.ui.slider(0.1, 10.0, step=0.1, value=1.0, label="阻尼 X_u 缩放")
    s_yv = mo.ui.slider(0.1, 10.0, step=0.1, value=1.0, label="阻尼 Y_v 缩放")
    s_nr = mo.ui.slider(0.1, 10.0, step=0.1, value=1.0, label="阻尼 N_r 缩放")
    s_am = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="附加质量缩放")
    s_cur = mo.ui.slider(0.0, 1.0, step=0.05, value=0.0, label="洋流强度 (m/s, NED 北向)")
    s_tau = mo.ui.slider(0.1, 2.0, step=0.1, value=1.0, label="推力限幅缩放")
    mo.vstack(
        [
            mo.md("### BlueROV2 3-DOF 动力学面板 D"),
            mo.hstack([s_xu, s_yv, s_nr]),
            mo.hstack([s_am, s_cur, s_tau]),
        ]
    )
    return s_am, s_cur, s_nr, s_tau, s_xu, s_yv


@app.cell
def _(s_am, s_cur, s_nr, s_tau, s_xu, s_yv, vehicle_cfg):
    import torch

    from uwm.dynamics.fossen import Fossen3DOF

    # 按滑块缩放构造模型配置
    cfg = dict(vehicle_cfg)
    cfg["linear_damping"] = [
        vehicle_cfg["linear_damping"][0] * s_xu.value,
        vehicle_cfg["linear_damping"][1] * s_yv.value,
        vehicle_cfg["linear_damping"][2] * s_nr.value,
    ]
    cfg["added_mass"] = [a * s_am.value for a in vehicle_cfg["added_mass"]]
    cfg["current"] = [s_cur.value, 0.0]
    tau_max = [t * s_tau.value for t in vehicle_cfg["tau_max"]]

    fossen = Fossen3DOF(cfg)

    # 阶跃推力（t >= 1 s 后施加半幅指令），RK4 积分
    dt = float(cfg["dt"])
    n_steps = int(20.0 / dt)
    tau = torch.tensor(
        [0.5 * tau_max[0], 0.3 * tau_max[1], 0.3 * tau_max[2]], dtype=torch.float64
    )
    eta = torch.zeros(3, dtype=torch.float64)
    nu = torch.zeros(3, dtype=torch.float64)
    etas, nus = [eta.clone()], [nu.clone()]
    for k in range(n_steps):
        eta, nu = fossen.step(eta, nu, tau if (k + 1) * dt >= 1.0 else tau * 0.0, dt)
        etas.append(eta.clone())
        nus.append(nu.clone())
    t = torch.arange(n_steps + 1, dtype=torch.float64) * dt
    traj_eta = torch.stack(etas).numpy()
    traj_nu = torch.stack(nus).numpy()
    ts = t.numpy()

    # 解析时间常数 tau_i = M_ii / D_ii（一阶通道近似）
    tau_consts = (fossen.M.diagonal() / fossen.D.diagonal()).numpy()
    return fossen, tau_consts, traj_eta, traj_nu, ts


@app.cell
def _(fossen, plt, tau_consts, traj_nu, ts):
    labels = ["u (m/s)", "v (m/s)", "r (rad/s)"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(ts, traj_nu[:, i], label=labels[i])
        ax.axvline(
            1.0 + tau_consts[i],
            color="r",
            ls="--",
            label=f"analytic τ={tau_consts[i]:.2f} s",
        )
        ax.axvline(1.0, color="k", ls=":", alpha=0.4, label="step onset")
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    d = fossen.D.diagonal()
    fig.suptitle(f"step response (scaled D = {d.numpy().round(3)})")
    fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _(plt, traj_eta):
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(traj_eta[:, 0], traj_eta[:, 1], "b-")
    ax2.plot(traj_eta[0, 0], traj_eta[0, 1], "go", label="start")
    ax2.plot(traj_eta[-1, 0], traj_eta[-1, 1], "r^", label="end")
    ax2.set_xlabel("x NED (m)")
    ax2.set_ylabel("y NED (m)")
    ax2.set_title("2D top-down trajectory")
    ax2.grid(alpha=0.3)
    ax2.legend()
    ax2.set_aspect("equal", adjustable="datalim")
    fig2.tight_layout()
    fig2
    return (fig2,)


if __name__ == "__main__":
    app.run()
