"""M2 测试（SPEC_M2 §2.6）：环境适配器、冒烟训练、convert、P1 探针。

冒烟训练（test_smoke_training）通过子进程跑 scripts/train_dreamer.py，
与真实用法一致；其产物（tmp_path 下的 dv3_logdir）由 module 级 fixture
共享给 convert / P1 测试，避免重复训练。
"""

import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from uwm.compat import ensure_upstream_compat
from uwm.eval.dv3 import add_vendored_path

ROOT = Path(__file__).resolve().parents[1]

# SPEC_M2 §2.6 的 tiny 参数（上限），logdir 由 fixture 换成 tmp_path
TINY_OVERRIDES = {
    "task": "uwm_station_keeping",
    "steps": 600,
    "eval_every": 500,
    "log_every": 500,
    "eval_episode_num": 1,
    "prefill": 100,
    "envs": 1,
    "action_repeat": 1,
    "time_limit": 100,  # 冒烟专用：缩短 episode 提速（非 SPEC 合同项）
    "size": [16, 16],
    "device": "cpu",
    "compile": False,
    "precision": 32,
    "train_ratio": 32,
    "batch_size": 4,
    "batch_length": 8,
    "dyn_hidden": 32,
    "dyn_deter": 32,
    "dyn_stoch": 16,
    "dyn_discrete": 16,
    "units": 32,
    "imag_horizon": 15,
    "encoder": {"mlp_keys": "state", "cnn_keys": "$^", "mlp_layers": 2, "mlp_units": 64},
    "decoder": {"mlp_keys": "state", "cnn_keys": "$^", "mlp_layers": 2, "mlp_units": 64},
    "video_pred_log": False,
    "reward_EMA": True,
}


def _make_adapter():
    ensure_upstream_compat()
    add_vendored_path()
    import envs.uwm as uwm_envs

    return uwm_envs.UWMStationKeeping("uwm_station_keeping", seed=0)


def test_adapter_interface():
    """① 适配器接口：obs dict 键/形状/dtype、action 界、is_first/is_terminal、4 元组 step。"""
    env = _make_adapter()

    # observation_space 只声明 "state"（SPEC_M2 §2.1）
    assert set(env.observation_space.spaces.keys()) == {"state"}
    state_space = env.observation_space.spaces["state"]
    assert state_space.shape == (7,) and state_space.dtype == np.float32
    assert env.action_space.shape == (3,)
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)

    obs = env.reset()
    assert obs["state"].shape == (7,) and obs["state"].dtype == np.float32
    assert obs["is_first"] is True
    assert obs["is_terminal"] is False
    assert {"state", "image", "is_terminal", "is_first"} <= set(obs.keys())

    action = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    out = env.step(action)
    assert len(out) == 4, "旧 gym 4 元组 step"
    obs, reward, done, info = out
    assert obs["state"].shape == (7,) and obs["state"].dtype == np.float32
    assert np.all(np.isfinite(obs["state"]))
    assert obs["is_first"] is False
    assert obs["is_terminal"] is False
    assert np.isfinite(reward)
    assert done is False
    assert float(info["discount"]) == 1.0  # 截断非终止，照常 bootstrap

    # 环境内部限长（max_episode_steps=500）截断 → done=True 且 is_terminal False
    for _ in range(499):
        _, _, done, _ = env.step(np.zeros(3, dtype=np.float32))
    assert done is True


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """冒烟训练产物（module 级共享）：返回 (run_dir, logdir, 训练耗时秒)。"""
    tmp = tmp_path_factory.mktemp("dv3_smoke")
    logdir = tmp / "run" / "dv3_logdir"
    overrides = dict(TINY_OVERRIDES, logdir=str(logdir))
    exp = {
        "exp_id": "m2_dreamer_smoke_test",
        "seeds": [0],
        "vehicle": "configs/vehicle/bluerov2.yaml",
        "task": {},
        "dv3_overrides": overrides,
        "seed": 0,
    }
    exp_path = tmp / "exp.yaml"
    with open(exp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(exp, f)

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_dreamer.py"),
         str(exp_path), "--no-post"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    duration = time.time() - t0
    assert proc.returncode == 0, (
        f"冒烟训练失败（{duration:.0f}s）：\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    return logdir.parent, logdir, duration


def test_smoke_training(smoke_run):
    """② 冒烟训练 10 分钟内跑完且生成 ckpt 与 episode 数据。"""
    _, logdir, duration = smoke_run
    assert (logdir / "latest.pt").exists(), "ckpt latest.pt 未生成"
    assert list((logdir / "train_eps").glob("*.npz")), "train_eps 无数据"
    assert list((logdir / "eval_eps").glob("*.npz")), "eval_eps 无数据"
    assert duration < 600, f"冒烟训练超时：{duration:.0f}s >= 600s"


def test_convert_episodes(smoke_run):
    """③ convert：冒烟 logdir → recorder schema，load_episode 读回字段齐全。"""
    from uwm.eval.dv3_episodes import convert_logdir
    from uwm.eval.recorder import load_episode

    run_dir, logdir, _ = smoke_run
    summary = convert_logdir(logdir, run_dir)
    assert summary["n_eval_episodes"] >= 1

    ep_path = run_dir / "episodes" / "ep_0000.npz"
    assert ep_path.exists()
    ep = load_episode(ep_path)
    T = len(ep["t"])
    assert T >= 20, "P1 探针需要 >= 5+15 步"
    assert ep["eta"].shape == (T, 3) and ep["nu"].shape == (T, 3)
    assert ep["action"].shape == (T, 3) and ep["reward"].shape == (T,)
    assert np.allclose(np.diff(ep["t"]), 0.1)  # dt 0.02 × action_repeat 5
    assert np.all(np.isfinite(ep["eta"])) and np.all(np.isfinite(ep["nu"]))

    with open(run_dir / "metrics.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == [
        "step", "episode", "env_steps", "return_", "final_dist",
        "ep_len", "q_loss", "actor_loss", "alpha", "is_eval",
    ]  # 与 M1 同表头
    assert any(r["is_eval"] == "1" for r in rows)


def test_p1_probe(smoke_run):
    """④ P1：冒烟 ckpt 上 open_loop_imagination_error 跑通，输出 dict 值 finite。"""
    from uwm.eval.wm_probes import run_p1

    run_dir, _, _ = smoke_run
    errors = run_p1(run_dir, horizons=(1, 5, 10, 15))
    assert set(errors.keys()) == {1, 5, 10, 15}
    for h, v in errors.items():
        assert np.isfinite(v), f"horizon {h} 的 L2 误差非 finite: {v}"
    assert (run_dir / "p1_imagination_error.png").exists()
