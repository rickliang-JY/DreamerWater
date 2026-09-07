"""UWM 项目 molab 审查入口（一键打开即审查）。

在 molab 中打开本文件后，notebook 会：
1. 自动从 GitHub 克隆 DreamerWater 仓库（自举，无需本地环境）
2. 实际运行全部 pytest 测试并展示结果
3. 内嵌动力学交互面板（滑块调参看阶跃响应）
4. 内嵌 episode 回放（随机 vs PD 策略轨迹对比）

实现要点：
- uwm 是仓库内的本地包（不在 PyPI），本 notebook 不出现任何字面
  `import uwm` 语句——统一在自举 cell 中用 importlib 动态导入，
  避免 marimo/molab 的包管理器误把 uwm 当 PyPI 包安装。
- uwm 是纯 Python 包，克隆后直接 sys.path.insert 即可导入，
  不用 pip install——规避 molab 的 uv 沙箱与系统 pip 环境错位。
- pytest 子进程通过 PYTHONPATH 环境变量拿到仓库路径。
- marimo 要求跨 cell 变量名唯一，未导出的局部变量一律下划线前缀。

本地运行：marimo edit notebooks/molab_quickstart.py
molab 打开：https://molab.marimo.io/github/rickliang-JY/DreamerWater/blob/main/notebooks/molab_quickstart.py
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24",
#     "torch",
#     "gymnasium",
#     "pyyaml",
#     "matplotlib",
#     "numpy",
#     "pytest",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    # 仓库地址（molab 自举时会从这里克隆）
    REPO_URL = "https://github.com/rickliang-JY/DreamerWater.git"
    REPO_BRANCH = "main"
    return REPO_BRANCH, REPO_URL, mo


@app.cell
def _(REPO_BRANCH, REPO_URL, mo):
    import importlib
    import subprocess as _sp_boot
    import sys
    from pathlib import Path

    work = Path("/tmp/uwm_repo")
    if not (work / "uwm").exists():
        # 自举：克隆仓库（molab 容器有完整网络）
        _sp_boot.run(
            ["git", "clone", "-q", "-b", REPO_BRANCH, REPO_URL, str(work)], check=True
        )
    else:
        # 已存在则拉取最新（失败不致命，用本地缓存）
        _sp_boot.run(["git", "-C", str(work), "pull", "-q"], check=False)

    # uwm 是纯 Python 包：直接挂 sys.path 即可导入，无需 pip install
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))

    # 动态导入本地包 uwm（不写字面 import，防止 molab 误判为 PyPI 依赖）
    Fossen3DOF = importlib.import_module("uwm.dynamics.fossen").Fossen3DOF
    load_episode = importlib.import_module("uwm.eval.recorder").load_episode

    py_exec = sys.executable  # 当前内核解释器，供 pytest 子进程使用
    mo.md(f"✅ 仓库已就绪：`{work}`（{REPO_URL} @ {REPO_BRANCH}）")
    return Fossen3DOF, load_episode, py_exec, work


@app.cell
def _(mo, work):
    mo.md(
        """
    ## UWM — 水下机器人世界模型训练与观测框架（M0 审查）

    | 层 | 交付物 |
    |---|---|
    | 动力学 | `uwm/dynamics/fossen.py` — 3-DOF Fossen 模型，torch.float64 + RK4，支持批量 |
    | 环境 | `uwm/envs/station_keeping.py` — gymnasium 定点悬停环境 |
    | 回归基准 | REMUS100 与 Fossen 本人 PythonVehicleSimulator 对齐（M/D 矩阵 <1%，轨迹 <5%） |
    | 观测 | 面板 D 动力学 playground + 面板 B episode 回放（下方内嵌） |

    仓库文件树：
    """
    )
    tree = sorted(
        str(p.relative_to(work))
        for p in work.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and "egg-info" not in p.parts
    )
    return (mo.md("\n".join(f"- `{t}`" for t in tree)),)


@app.cell
def _(mo, py_exec, work):
    import os
    import subprocess as _sp_test

    # 实际运行全部测试（审查的硬证据）。
    # 用当前内核解释器 + PYTHONPATH 指向仓库，保证子进程环境与 notebook 一致。
    env = dict(os.environ)
    env["PYTHONPATH"] = str(work) + os.pathsep + env.get("PYTHONPATH", "")
    r = _sp_test.run(
        [py_exec, "-m", "pytest", "tests/", "-p", "no:cacheprovider", "-q"],
        cwd=work,
        capture_output=True,
        text=True,
        env=env,
    )
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
    mo.md(f"### pytest 实测\n```\n{tail}\n```")
    return ()


@app.cell
def _(mo):
    mo.md("---\n### 面板 D：动力学 playground（滑块调参，实时重积分）")
    s_xu = mo.ui.slider(0.1, 10.0, step=0.1, value=1.0, label="阻尼 X_u 缩放")
    s_am = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="附加质量缩放")
    s_cur = mo.ui.slider(0.0, 1.0, step=0.05, value=0.0, label="洋流强度 (m/s)")
    mo.hstack([s_xu, s_am, s_cur])
    return s_am, s_cur, s_xu


@app.cell
def _(Fossen3DOF, s_am, s_cur, s_xu, work):
    import matplotlib.pyplot as _plt_d
    import numpy as np
    import torch
    import yaml

    with open(work / "configs" / "vehicle" / "bluerov2.yaml") as f:
        vehicle_cfg = yaml.safe_load(f)
    cfg = dict(vehicle_cfg)
    cfg["linear_damping"] = [
        vehicle_cfg["linear_damping"][0] * s_xu.value,
        *vehicle_cfg["linear_damping"][1:],
    ]
    cfg["added_mass"] = [a * s_am.value for a in vehicle_cfg["added_mass"]]
    cfg["current"] = [s_cur.value, 0.0]
    fossen = Fossen3DOF(cfg)

    dt = float(cfg["dt"])
    n = int(20.0 / dt)
    tau = torch.tensor([0.5 * cfg["tau_max"][0], 0.0, 0.0], dtype=torch.float64)
    _eta, _nu = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    nus = [_nu.clone()]
    for _k in range(n):
        _eta, _nu = fossen.step(
            _eta, _nu, tau if (_k + 1) * dt >= 1.0 else tau * 0.0, dt
        )
        nus.append(_nu.clone())

    ts = np.arange(n + 1) * dt
    u = torch.stack(nus)[:, 0].numpy()
    tau_u = float((fossen.M[0, 0] / fossen.D[0, 0]).item())

    fig, ax = _plt_d.subplots(figsize=(7, 3.5))
    ax.plot(ts, u, label="surge u (m/s)")
    ax.axvline(1.0 + tau_u, color="r", ls="--", label=f"analytic tau = {tau_u:.2f} s")
    ax.axvline(1.0, color="k", ls=":", alpha=0.4)
    ax.set_xlabel("t (s)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig
    return ()


@app.cell
def _(mo, work):
    import glob

    eps = sorted(glob.glob(str(work / "runs" / "demo" / "episodes" / "*.npz")))
    mo.md("---\n### 面板 B：episode 回放（ep 0–4 随机策略，ep 5–9 PD 回正）")
    if not eps:
        ep_slider, t_slider = None, None
        mo.md("⚠️ 未找到 runs/demo 数据")
    else:
        ep_slider = mo.ui.slider(0, len(eps) - 1, step=1, value=5, label="episode")
        t_slider = mo.ui.slider(0, 200, step=1, value=200, label="时间步")
        mo.hstack([ep_slider, t_slider])
    return ep_slider, eps, t_slider


@app.cell
def _(ep_slider, eps, load_episode, t_slider):
    if ep_slider is not None:
        import matplotlib.pyplot as _plt_b

        d = load_episode(eps[ep_slider.value])
        _kk = min(t_slider.value, len(d["t"]) - 1)
        _traj = d["eta"]

        fig2, ax2 = _plt_b.subplots(figsize=(5, 5))
        ax2.plot(
            _traj[: _kk + 1, 0], _traj[: _kk + 1, 1], "b-", lw=1.5, label="traversed"
        )
        ax2.plot(_traj[_kk:, 0], _traj[_kk:, 1], "b-", alpha=0.2)
        ax2.plot(
            _traj[_kk, 0], _traj[_kk, 1], "bo", ms=10, label=f"t={d['t'][_kk]:.1f}s"
        )
        ax2.plot(0, 0, "r*", ms=15, label="goal")
        ax2.set_xlabel("x NED (m)")
        ax2.set_ylabel("y NED (m)")
        ax2.set_title(
            f"ep {ep_slider.value} | return={float(d['reward'].sum()):.1f} | step {_kk}"
        )
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best", fontsize=8)
        ax2.set_aspect("equal", adjustable="datalim")
        fig2.tight_layout()
        fig2
    return ()


@app.cell
def _(mo):
    mo.md(
        """
    ---
    **审查清单**：① 上方 pytest 输出应全绿；② 拖面板 D 滑块，阶跃响应时间常数应跟随阻尼缩放移动；
    ③ 面板 B 拖 episode 到 5–9，应看到轨迹收敛到红星（原点）。
    """
    )
    return ()


if __name__ == "__main__":
    app.run()
