"""M4 测试（SPEC_M4 §4）：OU 时变洋流 + env 集成 + dv3 暗通道 + P2 探针冒烟。

覆盖 SPEC_M4 §4 的 6 项：
    ① OU 统计性质（均值 / lag-1 自相关 / 稳态 std 数量级）
    ② 可复现（同 seed 逐位相同，不同 seed 不同）
    ③ 向后兼容（列表形式常值流行为与 v0.1 完全一致；既有 30 项不破）
    ④ env 集成（reset 可复现、obs 7 维无泄漏、info["current"] 真值通道）
    ⑤ dv3 适配器（current_gt 暗通道 + encoder/decoder 结构上看不到它
       + OU tiny 冒烟训练跑通且 episode 落盘含 current_gt）
    ⑥ P2 smoke（冒烟 ckpt 上 hidden_state_current_probe 管线跑通，
       字段齐全、R² 有限、基线同算；不断言 R² 大小）
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from uwm.compat import ensure_upstream_compat
from uwm.eval.dv3 import add_vendored_path, apply_cpu_fallback, build_dv3_config

ROOT = Path(__file__).resolve().parents[1]

OU_CURRENT = {
    "type": "ou",
    "mean": [0.3, 0.0],
    "theta": 0.1,
    "sigma": 0.05,
    "seed": 0,
}

# 与 tests/test_dreamer.py 的 TINY_OVERRIDES 一致（logdir 由 fixture 换成 tmp_path）
TINY_OVERRIDES = {
    "task": "uwm_station_keeping",
    "steps": 600,
    "eval_every": 500,
    "log_every": 500,
    "eval_episode_num": 1,
    "prefill": 100,
    "envs": 1,
    "action_repeat": 1,
    "time_limit": 100,
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


def _ou_vehicle_cfg(ou_seed: int = 0) -> dict:
    """bluerov2 参数 + OU 洋流（测试内构造，不依赖配置文件）。"""
    with open(ROOT / "configs" / "vehicle" / "bluerov2.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["current"] = dict(OU_CURRENT, seed=ou_seed)
    return cfg


def _zeros() -> torch.Tensor:
    return torch.zeros(3, dtype=torch.float64)


# ---------------------------------------------------------------- ① OU 统计

def test_ou_statistics():
    """① OU 统计性质：样本均值 ≈ mean；lag-1 自相关 ≈ exp(-θ·dt)。"""
    from uwm.dynamics.fossen import Fossen3DOF

    theta, sigma, dt, n = 0.1, 0.05, 0.02, 10_000
    fossen = Fossen3DOF(_ou_vehicle_cfg())
    eta, nu, tau = _zeros(), _zeros(), _zeros()
    samples = []
    for _ in range(n):
        eta, nu = fossen.step(eta, nu, tau, dt)
        samples.append(fossen.current.numpy().copy())
    samples = np.asarray(samples)  # (N, 2)
    assert samples.shape == (n, 2)

    # 均值容差：SPEC_M4 §4-1 的形式 3σ/√N·√(2/θ) = 3σ_stat/√N。OU 序列
    # 强相关（相关时间 1/θ=10s ≫ dt），N 必须取去相关后的有效样本数
    # N_eff = N·dt·θ/2 = T/(2τ_c)，σ_stat = σ/√(2θ) 为稳态 std——
    # 两者代回即 3σ/√N_eff·… 与 SPEC 公式同形，只是 N→N_eff。
    # （若按字面 N=10⁴，容差 0.0067 远小于时间均值真实 3σ ≈ 0.106，必误杀。）
    sigma_stat = sigma / np.sqrt(2.0 * theta)
    n_eff = n * dt * theta / 2.0
    tol_mean = 3.0 * sigma_stat / np.sqrt(n_eff)
    np.testing.assert_allclose(
        samples.mean(axis=0), OU_CURRENT["mean"], atol=tol_mean,
        err_msg=f"OU 样本均值偏离 mean（容差 {tol_mean:.4f}）",
    )

    # lag-1 经验自相关 ≈ exp(-θ·dt)（SPEC 容差 0.05）
    x = samples[:, 0] - samples[:, 0].mean()
    ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(ac1 - np.exp(-theta * dt)) < 0.05
    # 附加判据：lag=1/(θ·dt)=500（一个相关时间）处 ≈ e^{-1}，更具区分度
    lag = int(round(1.0 / (theta * dt)))
    ac_lag = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert abs(ac_lag - np.exp(-1.0)) < 0.05

    # 稳态 std 数量级 sanity（链从 mean 出发，有限样本略偏低，宽松容差）
    np.testing.assert_allclose(
        samples.std(axis=0), [sigma_stat, sigma_stat], rtol=0.4
    )


# ---------------------------------------------------------------- ② 可复现

def _rollout_current(fossen, n: int) -> np.ndarray:
    eta, nu, tau = _zeros(), _zeros(), _zeros()
    seq = []
    for _ in range(n):
        eta, nu = fossen.step(eta, nu, tau, 0.02)
        seq.append(fossen.current.numpy().copy())
    return np.asarray(seq)


def test_ou_reproducibility():
    """② 同 seed 两次 reset_current + 100 步序列逐位相同；不同 seed 不同。"""
    from uwm.dynamics.fossen import Fossen3DOF

    fa = Fossen3DOF(_ou_vehicle_cfg(ou_seed=0))
    fb = Fossen3DOF(_ou_vehicle_cfg(ou_seed=0))
    fc = Fossen3DOF(_ou_vehicle_cfg(ou_seed=1))

    seq_a1 = _rollout_current(fa, 100)
    seq_b = _rollout_current(fb, 100)  # 独立实例同 seed
    fa.reset_current()
    seq_a2 = _rollout_current(fa, 100)  # 同实例 reset_current 复播
    seq_c = _rollout_current(fc, 100)  # 不同 seed

    assert np.array_equal(seq_a1, seq_b), "同 seed 独立实例应逐位相同"
    assert np.array_equal(seq_a1, seq_a2), "reset_current 后应可逐位复播"
    assert not np.array_equal(seq_a1, seq_c), "不同 seed 应产生不同序列"

    # reset_current(seed=...) 显式种子：同参数逐位相同
    fa.reset_current(seed=123)
    fb.reset_current(seed=123)
    assert np.array_equal(_rollout_current(fa, 100), _rollout_current(fb, 100))
    # 与全局 torch RNG 无关（独立 Generator）：全局 seed 改变不影响 OU 流
    fa.reset_current()
    torch.manual_seed(999)
    seq1 = _rollout_current(fa, 100)
    fa.reset_current()
    torch.manual_seed(12345)
    seq2 = _rollout_current(fa, 100)
    assert np.array_equal(seq1, seq2), "OU 随机流必须独立于全局 torch RNG"


# ---------------------------------------------------------------- ③ 向后兼容

def test_constant_current_backward_compat():
    """③ 列表形式 current：常值流行为与 v0.1 完全一致（reset_current 为 no-op）。"""
    from uwm.dynamics.fossen import Fossen3DOF

    with open(ROOT / "configs" / "vehicle" / "bluerov2.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["current"] = [0.2, -0.1]
    fossen = Fossen3DOF(cfg)
    assert not fossen.ou_enabled
    assert fossen.ou_state is None

    before = fossen.current.numpy().copy()
    fossen.reset_current()  # no-op
    seq = _rollout_current(fossen, 100)
    np.testing.assert_array_equal(
        seq, np.tile(before, (100, 1)), err_msg="常值流不应随 step 变化"
    )
    np.testing.assert_allclose(fossen.current.numpy(), [0.2, -0.1])

    # current 属性可赋值（交互面板滑块依赖）
    fossen.current = [0.5, 0.5]
    np.testing.assert_allclose(fossen.current.numpy(), [0.5, 0.5])


# ---------------------------------------------------------------- ④ env 集成

def test_env_ou_integration():
    """④ OU 环境：reset 可复现、obs 7 维无洋流泄漏、info["current"] 真值通道。"""
    from uwm.envs.station_keeping import StationKeepingEnv

    env = StationKeepingEnv(_ou_vehicle_cfg(), {})

    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    assert obs1.shape == (7,), "obs 必须保持 7 维（洋流不可观测，无泄漏）"
    np.testing.assert_array_equal(obs1, obs2, err_msg="同 seed reset obs 应一致")
    np.testing.assert_allclose(
        info1["current"], OU_CURRENT["mean"],
        err_msg="reset 后洋流应回到 mean（同 seed 同初始洋流）",
    )
    np.testing.assert_allclose(info1["current"], info2["current"])

    # info["current"] 形状正确且随时间变化；同 seed 整条洋流序列可复现
    def _episode_current(seed):
        env.reset(seed=seed)
        seq = []
        for _ in range(50):
            _, _, _, truncated, info = env.step(np.zeros(3))
            seq.append(np.asarray(info["current"], dtype=np.float64))
            if truncated:
                break
        return np.asarray(seq)

    seq_a = _episode_current(42)
    seq_b = _episode_current(42)
    seq_c = _episode_current(7)
    assert seq_a.shape == (50, 2), "info['current'] 应为 [u_c, v_c]"
    assert np.all(np.isfinite(seq_a))
    assert seq_a.std(axis=0).max() > 1e-4, "OU 洋流应随时间变化"
    np.testing.assert_array_equal(seq_a, seq_b, err_msg="同 seed 洋流序列应逐位复现")
    assert not np.array_equal(seq_a, seq_c), "不同 env seed 应产生不同洋流实现"

    # 常值流环境：info["current"] 也存在但恒定（通道语义一致）
    with open(ROOT / "configs" / "vehicle" / "bluerov2.yaml", encoding="utf-8") as f:
        const_cfg = yaml.safe_load(f)
    env_const = StationKeepingEnv(const_cfg, {})
    env_const.reset(seed=0)
    _, _, _, _, info = env_const.step(np.zeros(3))
    np.testing.assert_allclose(info["current"], [0.0, 0.0])
    assert env_const.observation_space.shape == (7,)


# ---------------------------------------------------------------- ⑤ dv3 适配器

def _make_adapter_ou(monkeypatch):
    monkeypatch.setenv(
        "UWM_VEHICLE_CFG", str(ROOT / "configs" / "vehicle" / "bluerov2_ou.yaml")
    )
    ensure_upstream_compat()
    add_vendored_path()
    import envs.uwm as uwm_envs

    return uwm_envs.UWMStationKeeping("uwm_station_keeping", seed=0)


def test_dv3_adapter_current_gt_channel(monkeypatch):
    """⑤a 适配器暗通道：obs dict 含 current_gt、space 声明它、OU 下随时间变化。"""
    env = _make_adapter_ou(monkeypatch)

    spaces = env.observation_space.spaces
    assert set(spaces.keys()) == {"state", "current_gt"}
    assert spaces["current_gt"].shape == (2,)
    assert spaces["current_gt"].dtype == np.float32
    assert spaces["state"].shape == (7,)

    obs = env.reset()
    assert {"state", "current_gt", "image", "is_terminal", "is_first"} <= set(obs)
    np.testing.assert_allclose(obs["current_gt"], [0.3, 0.0], atol=1e-6)
    assert obs["current_gt"].dtype == np.float32

    gt = []
    for _ in range(20):
        obs, _, done, _ = env.step(np.zeros(3, dtype=np.float32))
        gt.append(obs["current_gt"].copy())
        if done:
            break
    gt = np.asarray(gt)
    assert gt.shape == (20, 2) and np.all(np.isfinite(gt))
    assert gt.std(axis=0).max() > 1e-4, "OU 下 current_gt 应随时间变化"

    # 默认（无环境变量）仍为常值流 bluerov2（向后兼容）
    monkeypatch.delenv("UWM_VEHICLE_CFG")
    import envs.uwm as uwm_envs

    env_default = uwm_envs.UWMStationKeeping("uwm_station_keeping", seed=0)
    obs = env_default.reset()
    np.testing.assert_allclose(obs["current_gt"], [0.0, 0.0], atol=1e-6)


def test_dv3_encoder_decoder_blind_to_current_gt(monkeypatch):
    """⑤b 结构性证明：encoder/decoder 的键选择正则不匹配 current_gt。

    MultiEncoder/MultiDecoder 只消费 mlp_keys/cnn_keys 正则匹配到的键；
    mlp_keys='state' 时 current_gt 被排除 → RSSM 输入 embed 不含洋流真值，
    模型在结构上永远看不到暗通道（SPEC_M4 §1.4）。
    """
    env = _make_adapter_ou(monkeypatch)
    apply_cpu_fallback()
    import networks

    shapes = {k: tuple(v.shape) for k, v in env.observation_space.spaces.items()}
    cfg = build_dv3_config(TINY_OVERRIDES)
    assert cfg.encoder["mlp_keys"] == "state"
    assert cfg.decoder["mlp_keys"] == "state"

    encoder = networks.MultiEncoder(shapes, **cfg.encoder)
    assert set(encoder.mlp_shapes.keys()) == {"state"}, (
        f"encoder 不得消费 current_gt: {encoder.mlp_shapes}"
    )
    assert not encoder.cnn_shapes

    feat_size = cfg.dyn_stoch * cfg.dyn_discrete + cfg.dyn_deter
    decoder = networks.MultiDecoder(feat_size, shapes, **cfg.decoder)
    assert set(decoder.mlp_shapes.keys()) == {"state"}, (
        f"decoder 不得重建 current_gt: {decoder.mlp_shapes}"
    )
    assert not decoder.cnn_shapes


@pytest.fixture(scope="module")
def ou_smoke_run(tmp_path_factory):
    """⑤/⑥ 共享的 OU 冒烟训练产物：返回 (run_dir, logdir, 训练耗时秒)。"""
    tmp = tmp_path_factory.mktemp("dv3_ou_smoke")
    logdir = tmp / "run" / "dv3_logdir"
    overrides = dict(TINY_OVERRIDES, logdir=str(logdir))
    exp = {
        "exp_id": "m4_dreamer_ou_smoke_test",
        "seeds": [0],
        "vehicle": "configs/vehicle/bluerov2_ou.yaml",
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
        f"OU 冒烟训练失败（{duration:.0f}s）：\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    return logdir.parent, logdir, duration


def test_ou_smoke_training(ou_smoke_run):
    """⑤c OU tiny 冒烟训练跑通，且 episode 落盘含 current_gt 暗通道。"""
    _, logdir, duration = ou_smoke_run
    assert (logdir / "latest.pt").exists(), "ckpt latest.pt 未生成"
    train_eps = list((logdir / "train_eps").glob("*.npz"))
    eval_eps = list((logdir / "eval_eps").glob("*.npz"))
    assert train_eps and eval_eps, "train/eval episodes 缺失"
    assert duration < 600, f"冒烟训练超时：{duration:.0f}s >= 600s"

    for ep_path in (train_eps[0], eval_eps[0]):
        ep = np.load(ep_path)
        assert "current_gt" in ep.files, f"{ep_path.name} 缺 current_gt 暗通道"
        gt = ep["current_gt"]
        assert gt.ndim == 2 and gt.shape[1] == 2 and np.all(np.isfinite(gt))
        assert gt.std(axis=0).max() > 1e-4, "OU run 的 current_gt 应随时间变化"


# ---------------------------------------------------------------- ⑥ P2 smoke

def test_p2_probe_smoke(ou_smoke_run):
    """⑥ P2 管线：返回字段齐全、R² 有限、基线同算（不断言 R² 大小）。"""
    from uwm.eval.wm_probes import run_p2

    run_dir, _, _ = ou_smoke_run
    result = run_p2(run_dir)
    expected = {
        "r2_u", "r2_v", "r2_mean", "baseline_r2_mean", "n_train", "n_test",
    }
    assert expected <= set(result.keys()), f"缺字段: {expected - set(result.keys())}"
    for key in ("r2_u", "r2_v", "r2_mean", "baseline_r2_mean"):
        assert np.isfinite(result[key]), f"{key} 非有限值: {result[key]}"
    assert result["n_train"] > 0 and result["n_test"] > 0
    assert result["n_test_episodes"] >= 1, "应按 episode 划分出 test 集"
    assert (run_dir / "p2_probe.png").exists()
