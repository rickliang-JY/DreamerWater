"""marimo 回放面板 B：逐 episode 轨迹回放（只读 runs/*/，不 import 训练代码）。

控件：run 目录下拉框（扫描 runs/*/）+ episode index 滑块 + 时间步滑块。
显示：位置 x,y / 速度 u,v,r / 动作三维 三组曲线（当前时间步竖线标记）+
2D 俯视轨迹（已走实心线、未走淡色、当前位置大圆点、目标原点星标），
顶部显示累计 return 与当前步 reward。mo.ui.refresh 每 5s 轮询 run 目录，
录制进行中也能看到新 episode（滑块范围固定，新 episode 超出范围时重启 notebook）。

运行：marimo edit notebooks/episode_replay.py
     marimo run  notebooks/episode_replay.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, mo, np, plt


@app.cell
def _(Path):
    # 仓库根目录：marimo 单元中 __file__ 可能不存在，回退到当前工作目录
    try:
        root = Path(__file__).resolve().parents[1]
    except NameError:
        root = Path.cwd()
    return (root,)


@app.cell
def _(mo):
    # 每 5s 触发一次引用它的单元重跑，轮询 run 目录里的新 episode
    refresh = mo.ui.refresh(default_interval="5s")
    return (refresh,)


@app.cell
def _(Path, mo, root):
    # 扫描 runs/*/（含 episodes/ 子目录的才算 run），默认选 runs/demo
    runs_root = root / "runs"
    run_dirs = (
        sorted(p for p in runs_root.iterdir() if (p / "episodes").is_dir())
        if runs_root.is_dir()
        else []
    )
    options = {p.name: str(p) for p in run_dirs}
    default_run = "demo" if "demo" in options else (next(iter(options), None))
    run_dd = mo.ui.dropdown(
        options=options or {"<no runs found>": ""},
        value=default_run or "<no runs found>",
        label="run directory (runs/*)",
    )
    # 滑块范围在 notebook 启动时固定；录制出的新 episode 超出范围时重启 notebook 即可
    ep_sl = mo.ui.slider(0, 99, step=1, value=0, label="episode index")
    step_sl = mo.ui.slider(0, 999, step=1, value=0, label="time step")
    mo.vstack(
        [
            mo.md("### Episode Replay (panel B)"),
            run_dd,
            mo.hstack([ep_sl, step_sl]),
        ]
    )
    return ep_sl, run_dd, step_sl


@app.cell
def _(Path, ep_sl, mo, np, refresh, root, run_dd, step_sl):
    from uwm.eval.recorder import load_episode

    _ = refresh  # 依赖 refresh：每 5s 重新扫描目录并重读当前 episode（录制中可增长）
    run_path = Path(run_dd.value) if run_dd.value else None
    ep_files = (
        sorted(run_path.glob("episodes/ep_*.npz"))
        if run_path and (run_path / "episodes").is_dir()
        else []
    )
    ep_idx = min(ep_sl.value, len(ep_files) - 1) if ep_files else 0
    data = load_episode(ep_files[ep_idx]) if ep_files else None
    n_steps = len(data["t"]) if data else 0
    k = min(step_sl.value, n_steps - 1) if data else 0

    if data is None:
        header = mo.md(f"**No episodes found** in `{run_path}` — run "
                       f"`python scripts/record_demo_episodes.py` first.")
    else:
        total_return = float(np.sum(data["reward"]))
        header = mo.md(
            f"**run** `{run_path.name}` | **episode** {ep_idx} "
            f"({len(ep_files)} available, {n_steps} steps) | "
            f"**return** {total_return:.2f} | "
            f"**reward @ step {k}** (t={data['t'][k]:.1f}s): {data['reward'][k]:.4f}"
        )
    header
    return data, ep_files, ep_idx, k, n_steps, run_path


@app.cell
def _(data, k, np, plt):
    if data is not None:
        t, eta, nu, action = data["t"], data["eta"], data["nu"], data["action"]
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        series = [
            (eta[:, 0], "x (m)", "C0"), (eta[:, 1], "y (m)", "C1"),
        ], [
            (nu[:, 0], "u (m/s)", "C0"), (nu[:, 1], "v (m/s)", "C1"),
            (nu[:, 2], "r (rad/s)", "C2"),
        ], [
            (action[:, 0], "action[0]", "C0"), (action[:, 1], "action[1]", "C1"),
            (action[:, 2], "action[2]", "C2"),
        ]
        titles = ["position", "velocity", "action"]
        for ax, group, title in zip(axes, series, titles):
            for y, label, color in group:
                ax.plot(t, y, color=color, label=label)
            # 当前时间步竖线标记
            ax.axvline(t[k], color="k", ls="--", alpha=0.7, label="current step")
            ax.set_ylabel(title)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8, ncols=len(group) + 1)
        axes[-1].set_xlabel("t (s)")
        fig.suptitle("episode signals")
        fig.tight_layout()
        fig
    else:
        fig = None
        "no data"
    return (fig,)


@app.cell
def _(data, k, plt):
    if data is not None:
        pos = data["eta"]
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        # 未走的路径淡色，已走过的实心线
        ax2.plot(pos[:, 0], pos[:, 1], color="C0", alpha=0.2, lw=1, label="future path")
        ax2.plot(pos[: k + 1, 0], pos[: k + 1, 1], "C0-", lw=2, label="traveled path")
        ax2.plot(pos[0, 0], pos[0, 1], "gs", ms=8, label="start")
        ax2.plot(pos[k, 0], pos[k, 1], "C0o", ms=14, label="current")
        ax2.plot(0.0, 0.0, marker="*", color="red", ms=18, ls="none", label="goal (origin)")
        ax2.set_xlabel("x NED (m)")
        ax2.set_ylabel("y NED (m)")
        ax2.set_title("2D top-down trajectory")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best", fontsize=8)
        ax2.set_aspect("equal", adjustable="datalim")
        fig2.tight_layout()
        fig2
    else:
        fig2 = None
        "no data"
    return (fig2,)


if __name__ == "__main__":
    app.run()
