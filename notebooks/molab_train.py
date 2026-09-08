"""UWM molab 训练控制台：在浏览器里直接训练各阶段实验。

功能：
1. 自举：自动从 GitHub 克隆 DreamerWater 仓库并挂载 sys.path（无需本地环境）
2. 阶段选择：M1 SAC / M1-OU / M2 DreamerV3 / M2-H30 / M4 DreamerV3-OU
   （检测到 GPU 时 Dreamer 阶段自动使用 *_gpu 配置）
3. 开始/停止训练（后台子进程，UI 不阻塞；中断后重新开始 = 断点续训）
4. 实时监控：5s 轮询训练指标，eval return 曲线自动刷新
5. 轨迹回放：episode/时间步滑块（dv3 阶段先点"转换轨迹"）
6. 打包下载 runs/ 结果（metrics + 轨迹 + 图，不含大型 ckpt/buffer）

注意：uwm 为仓库内本地包（不在 PyPI），本 notebook 不出现字面 import uwm，
统一在自举 cell 用 importlib 动态导入（防止 molab 包管理器误装）。
marimo 跨 cell 变量名须唯一，未导出局部变量用下划线前缀。

molab 打开：https://molab.marimo.io/github/rickliang-JY/DreamerWater/blob/main/notebooks/molab_train.py
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
#     "ruamel.yaml",
#     "einops",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    REPO_URL = "https://github.com/rickliang-JY/DreamerWater.git"
    REPO_BRANCH = "main"
    return REPO_BRANCH, REPO_URL, mo


@app.cell
def _(REPO_BRANCH, REPO_URL, mo):
    import importlib
    import subprocess as _sp
    import sys as _sys
    from pathlib import Path

    work = Path("/tmp/uwm_repo")
    if not (work / "uwm").exists():
        _sp.run(["git", "clone", "-q", "-b", REPO_BRANCH, REPO_URL, str(work)], check=True)
    else:
        _sp.run(["git", "-C", str(work), "pull", "-q"], check=False)
    if str(work) not in _sys.path:
        _sys.path.insert(0, str(work))

    load_episode = importlib.import_module("uwm.eval.recorder").load_episode

    import torch as _torch

    has_gpu = _torch.cuda.is_available()
    gpu_note = "🎮 检测到 GPU——Dreamer 阶段将使用全量配置" if has_gpu else "💻 CPU 模式——Dreamer 阶段使用降档配置"
    mo.md(f"✅ 仓库就绪 `{work}`（{REPO_BRANCH}） | {gpu_note}")
    return has_gpu, load_episode, work


@app.cell
def _(has_gpu, mo):
    # 阶段 → (描述, 训练脚本, exp 配置)
    stages = {
        "M1: SAC 定点悬停（常值流）": (
            "scripts/train.py",
            "configs/exp/m1_sac_station_keeping.yaml",
        ),
        "M1-OU: SAC 定点悬停（时变洋流）": (
            "scripts/train.py",
            "configs/exp/m1_sac_station_keeping_ou.yaml",
        ),
        "M2: DreamerV3 H=15（常值流）": (
            "scripts/train_dreamer.py",
            "configs/exp/m2_dreamerv3_gpu.yaml"
            if has_gpu
            else "configs/exp/m2_dreamerv3_station_keeping.yaml",
        ),
        "M4: DreamerV3 OU 洋流 H=30（P2 探针对象）": (
            "scripts/train_dreamer.py",
            "configs/exp/m4_dreamerv3_ou_gpu.yaml"
            if has_gpu
            else "configs/exp/m4_dreamerv3_ou_smoke.yaml",
        ),
    }
    stage = mo.ui.dropdown(options=list(stages), label="训练阶段")
    n_seed = mo.ui.number(0, 9, step=1, value=0, label="seed")
    n_steps = mo.ui.number(
        0, 10**7, step=10000, value=0, label="步数覆盖（0 = 用配置默认）"
    )
    btn_start = mo.ui.run_button(label="▶ 开始训练")
    btn_stop = mo.ui.run_button(label="⏹ 停止")
    mo.vstack(
        [
            mo.md("### ① 选择阶段并启动（中断后再次开始 = 断点续训）"),
            stage,
            mo.hstack([n_seed, n_steps]),
            mo.hstack([btn_start, btn_stop]),
        ]
    )
    return btn_start, btn_stop, n_seed, n_steps, stage, stages


@app.cell
def _(btn_start, mo, n_seed, n_steps, stage, stages, work):
    import os as _os
    import subprocess as _sp2
    import sys as _sys2

    get_proc, set_proc = mo.state(None)

    if btn_start.value and get_proc() is None:
        script, exp_rel = stages[stage.value]
        cmd = [_sys2.executable, "-u", str(work / script), str(work / exp_rel),
               "--seed", str(n_seed.value)]
        if "train.py" in script and n_steps.value > 0:
            cmd += ["--total-steps", str(n_steps.value)]
        elif n_steps.value > 0:
            # dv3：生成带步数覆盖的临时 exp 配置
            import yaml as _yaml

            _cfg = _yaml.safe_load(open(work / exp_rel))
            _cfg["dv3_overrides"]["steps"] = int(n_steps.value)
            _tmp = work / "runs" / "_molab_tmp_exp.yaml"
            _tmp.parent.mkdir(exist_ok=True)
            _yaml.safe_dump(_cfg, open(_tmp, "w"), allow_unicode=True)
            cmd = [_sys2.executable, "-u", str(work / script), str(_tmp),
                   "--seed", str(n_seed.value)]
        _env = dict(_os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        _log = open(work / "runs" / "molab_train.log", "a")
        _p = _sp2.Popen(cmd, cwd=work, env=_env, stdout=_log, stderr=_log)
        set_proc({"popen": _p, "stage": stage.value, "seed": n_seed.value})
    elif btn_start.value and get_proc() is not None:
        pass  # 已有训练在跑

    _cur = get_proc()
    if _cur is None:
        mo.md("**状态**：空闲。选择阶段后点 ▶ 开始。")
    else:
        _rc = _cur["popen"].poll()
        _st = "运行中 🟢" if _rc is None else f"已退出（code {_rc}）"
        mo.md(f"**状态**：{_st} — {_cur['stage']} / seed{_cur['seed']} / pid {_cur['popen'].pid}")
    return get_proc, set_proc


@app.cell
def _(btn_stop, get_proc, mo, set_proc):
    if btn_stop.value and get_proc() is not None:
        get_proc()["popen"].terminate()
        set_proc(None)
    return ()


@app.cell
def _(get_proc, mo, work):
    import json as _json
    import os as _osm

    refresh = mo.ui.refresh(default_interval="5s")
    refresh  # 触发轮询

    _cur = get_proc()
    _fig_out = None
    _info = "尚未开始训练。"
    _logtail = ""
    _progress = ""

    # 日志尾部（无论是否在跑都显示，出错第一时间可见）
    _logf = work / "runs" / "molab_train.log"
    if _logf.exists():
        _lines = open(_logf).read().strip().splitlines()
        _logtail = "\n".join(_lines[-8:])

    if _cur is not None:
        _rc = _cur["popen"].poll()
        if _rc is not None and _rc != 0:
            _info = f"⚠️ **训练进程已退出（code {_rc}）**——错误见下方日志尾部"
        import glob as _glob

        _runs = sorted(_glob.glob(str(work / "runs" / "*" / f"seed{_cur['seed']}")))
        _xs, _ys, _src = [], [], None
        if _runs:
            _rd = _runs[0]
            _csv = f"{_rd}/metrics.csv"
            _jsonl = f"{_rd}/dv3_logdir/metrics.jsonl"
            try:
                if _cur["stage"].startswith(("M2", "M4")) and _osm.path.exists(_jsonl):
                    _last_step, _last_tr = 0, None
                    for _l in open(_jsonl):
                        _d = _json.loads(_l)
                        if "eval_return" in _d:
                            _xs.append(_d["step"])
                            _ys.append(_d["eval_return"])
                        if "train_return" in _d:
                            _last_step, _last_tr = _d["step"], _d["train_return"]
                    _src = "dv3 eval_return"
                    _ds = None
                    _progress = f"当前训练进度：**{_last_step} 步**" + (
                        f"，最近 train_return {_last_tr:.1f}" if _last_tr is not None else "（prefill 中）"
                    )
                elif _osm.path.exists(_csv):
                    import csv as _csv_mod

                    _rows = list(_csv_mod.DictReader(open(_csv)))
                    _agg = {}
                    _aggd = {}
                    for _row in _rows:
                        if _row.get("is_eval") == "1":
                            _agg.setdefault(int(_row["env_steps"]), []).append(
                                float(_row["return_"])
                            )
                            _aggd.setdefault(int(_row["env_steps"]), []).append(
                                float(_row["final_dist"])
                            )
                    _xs = sorted(_agg)
                    _ys = [sum(_agg[_x]) / len(_agg[_x]) for _x in _xs]
                    _ds = [sum(_aggd[_x]) / len(_aggd[_x]) for _x in _xs]
                    _src = "SAC eval (mean per eval point)"
                    if _rows:
                        _progress = (
                            f"当前训练进度：**{_rows[-1]['env_steps']} 步**，"
                            f"最近 episode return {float(_rows[-1]['return_']):.1f}，"
                            f"final_dist {float(_rows[-1]['final_dist']):.3f} m"
                        )
            except Exception as _e:  # noqa: BLE001
                _info = f"读取指标出错：{_e}"
        if _xs:
            import matplotlib.pyplot as _plt

            if _ds:
                _f, (_ax, _ax2) = _plt.subplots(1, 2, figsize=(10, 3.2))
                _ax.plot(_xs, _ys, "o-", ms=3)
                _ax.set_ylabel("eval return")
                _ax.set_title(f"return -- seed{_cur['seed']}")
                _ax2.plot(_xs, _ds, "s-", ms=3, color="darkred")
                _ax2.axhline(0.3, color="gray", ls="--", lw=1)
                _ax2.set_ylabel("final_dist (m)")
                _ax2.set_title("final distance vs steps (threshold=0.3m)")
                _ax2.grid(alpha=0.3)
                for _a in (_ax, _ax2):
                    _a.set_xlabel("env steps")
                    _a.grid(alpha=0.3)
            else:
                _f, _ax = _plt.subplots(figsize=(7, 3.2))
                _ax.plot(_xs, _ys, "o-", ms=3)
                _ax.set_xlabel("env steps")
                _ax.set_ylabel("eval return")
                _ax.set_title(f"{_src} -- seed{_cur['seed']}")
                _ax.grid(alpha=0.3)
            _f.tight_layout()
            _fig_out = _f
        if _rc is None:
            _info = _progress or "进程已启动，等待第一批数据写入……"
    mo.vstack(
        [
            mo.md(f"### ② 训练状态（5s 刷新）\n{_info}"),
            _fig_out or mo.md(""),
            mo.md(f"**日志尾部**：\n```\n{_logtail or '（暂无）'}\n```"),
        ]
    )
    return ()


@app.cell
def _(get_proc, mo, work):
    # dv3 阶段：训练产生的 episode 需要转换成回放格式
    btn_convert = mo.ui.run_button(label="🔄 转换 dv3 轨迹（供下方回放）")
    mo.md("### ③ 轨迹回放（SAC 直接可用；Dreamer 先点转换）")
    btn_convert
    return (btn_convert,)


@app.cell
def _(btn_convert, get_proc, mo, work):
    if btn_convert.value and get_proc is not None:
        import glob as _g2
        import subprocess as _sp3
        import sys as _s3

        _runs = _g2.glob(str(work / "runs" / "m2*")) + _g2.glob(str(work / "runs" / "m4*"))
        for _rd in _runs:
            for _sd in _g2.glob(f"{_rd}/seed*"):
                _sp3.run(
                    [_s3.executable, str(work / "scripts/convert_dv3_episodes.py"), _sd],
                    cwd=work, capture_output=True,
                )
    return ()


@app.cell
def _(get_proc, load_episode, mo, work):
    import glob as _g3

    _cur = get_proc()
    _ep_files = []
    if _cur is not None:
        _ep_files = sorted(
            _g3.glob(str(work / "runs" / "*" / f"seed{_cur['seed']}" / "episodes" / "*.npz"))
        )
    if not _ep_files:
        mo.md("暂无可回放 episode（SAC 每 20 episode 存一份；dv3 请先点转换）")
        ep_s, t_s = None, None
    else:
        ep_s = mo.ui.slider(0, len(_ep_files) - 1, step=1, value=len(_ep_files) - 1,
                            label="episode（最新在右）")
        t_s = mo.ui.slider(0, 500, step=5, value=500, label="时间步")
        mo.hstack([ep_s, t_s])
    return _ep_files, ep_s, t_s


@app.cell
def _(ep_s, load_episode, t_s, _ep_files):
    if ep_s is not None:
        import matplotlib.pyplot as _plt2

        _d = load_episode(_ep_files[ep_s.value])
        _k = min(t_s.value, len(_d["t"]) - 1)
        _tr = _d["eta"]
        _f2, _ax2 = _plt2.subplots(figsize=(5, 5))
        _ax2.plot(_tr[: _k + 1, 0], _tr[: _k + 1, 1], "b-", lw=1.5)
        _ax2.plot(_tr[_k:, 0], _tr[_k:, 1], "b-", alpha=0.2)
        _ax2.plot(_tr[_k, 0], _tr[_k, 1], "bo", ms=10)
        _ax2.plot(0, 0, "r*", ms=15)
        _ax2.set_xlabel("x NED (m)"); _ax2.set_ylabel("y NED (m)")
        _ax2.set_title(f"ep file #{ep_s.value} | step {_k} | return={float(_d['reward'].sum()):.1f}")
        _ax2.grid(alpha=0.3); _ax2.set_aspect("equal", adjustable="datalim")
        _f2.tight_layout()
        _f2
    return ()


@app.cell
def _(mo):
    btn_zip = mo.ui.run_button(label="📦 打包 runs/ 结果")
    mo.vstack([mo.md("### ④ 导出结果"), btn_zip])
    return (btn_zip,)


@app.cell
def _(btn_zip, mo, work):
    _dl = None
    if btn_zip.value:
        import glob as _g4
        import os as _o4
        import zipfile as _zf

        _zp = "/tmp/uwm_runs_export.zip"
        with _zf.ZipFile(_zp, "w", _zf.ZIP_DEFLATED) as _z:
            for _fp in _g4.glob(str(work / "runs" / "**"), recursive=True):
                # 排除大文件：ckpt、buffer、dv3 原始 episode 缓存
                if any(x in _fp for x in ("ckpt", "train_eps", "dv3_logdir/scores")):
                    continue
                if _o4.path.isfile(_fp):
                    _z.write(_fp, _o4.path.relpath(_fp, work))
        _dl = mo.download(
            data=open(_zp, "rb").read(),
            filename="uwm_runs_export.zip",
            mimetype="application/zip",
            label="⬇ 下载结果包（metrics + 轨迹 + 图）",
        )
    _dl or mo.md("点击上方按钮打包。")
    return ()


@app.cell
def _(mo):
    mo.md(
        """
    ---
    **提示**：① molab 会话断开 = 训练停止，长时间训练请保持页面开启（或把 runs 下载后到本地/GPU 续训）；
    ② Dreamer 阶段在 GPU 会话下自动用全量配置（200k 步），CPU 会话用降档配置（20k 步）；
    ③ 完整实验矩阵与正式流程见仓库 `scripts/gpu_run_all.sh` 与 README「GPU 全量复现实验指南」。
    """
    )
    return ()


if __name__ == "__main__":
    app.run()
