"""EpisodeRecorder / load_episode 测试（提案 §7.1 文件契约）。"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from uwm.eval.recorder import EpisodeRecorder, load_episode


def _fake_traj(n_steps: int = 200):
    """构造一条假 episode：t (T,)，eta/nu/action (T,3)，reward (T,)。"""
    rng = np.random.default_rng(0)
    t = np.arange(n_steps, dtype=np.float64) * 0.1
    eta = rng.normal(size=(n_steps, 3))
    nu = rng.normal(size=(n_steps, 3))
    action = rng.uniform(-1.0, 1.0, size=(n_steps, 3))
    reward = -np.abs(rng.normal(size=n_steps))
    return t, eta, nu, action, reward


def test_episode_roundtrip(tmp_path):
    """save_episode 后 load_episode 逐元素一致（eta/nu/action 存 float32）。"""
    rec = EpisodeRecorder(tmp_path / "run", config={"vehicle": "bluerov2"})
    t, eta, nu, action, reward = _fake_traj()
    path = rec.save_episode(3, t, eta, nu, action, reward)
    rec.close()

    data = load_episode(path)
    assert set(data) == {"t", "eta", "nu", "action", "reward"}
    np.testing.assert_array_equal(data["t"], t)
    np.testing.assert_array_equal(data["reward"], reward)
    np.testing.assert_allclose(data["eta"], eta, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(data["nu"], nu, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(data["action"], action, rtol=1e-6, atol=1e-6)


def test_config_yaml_written(tmp_path):
    """初始化时写 config.yaml，内容可回读。"""
    cfg = {"vehicle": "bluerov2", "episodes": 10}
    rec = EpisodeRecorder(tmp_path / "run", config=cfg)
    rec.close()
    with open(tmp_path / "run" / "config.yaml", encoding="utf-8") as f:
        assert yaml.safe_load(f) == cfg


def test_metrics_csv_append(tmp_path):
    """metrics.csv 追加写：两次 log 后 = 表头 + 2 行，列名正确。"""
    rec = EpisodeRecorder(tmp_path / "run", config={})
    rec.log_metrics(step=0, episode=0, return_=-1.5, final_dist=3.2)
    rec.log_metrics(step=1, episode=1, return_=-0.8, final_dist=1.1)
    rec.close()

    lines = (tmp_path / "run" / "metrics.csv").read_text().strip().splitlines()
    assert len(lines) == 3
    assert lines[0].split(",") == ["step", "episode", "return_", "final_dist"]
    assert lines[1].split(",")[1:] == ["0", "-1.5", "3.2"]
    assert lines[2].split(",")[1:] == ["1", "-0.8", "1.1"]

    # 键集合与表头不一致必须报错（CSV 无法动态加列）
    rec2 = EpisodeRecorder(tmp_path / "run2", config={})
    rec2.log_metrics(step=0, a=1)
    with pytest.raises(ValueError):
        rec2.log_metrics(step=1, b=2)
    rec2.close()


def test_episode_filename_zero_padded(tmp_path):
    """ep 文件命名四位补零：ep_0007.npz。"""
    rec = EpisodeRecorder(tmp_path / "run", config={})
    path = rec.save_episode(7, *_fake_traj(n_steps=10))
    rec.close()
    assert path.name == "ep_0007.npz"
    assert (tmp_path / "run" / "episodes" / "ep_0007.npz").is_file()


def test_episode_file_size(tmp_path):
    """200 步 episode 的 npz 体积 < 200 KB（几十 KB 级）。"""
    rec = EpisodeRecorder(tmp_path / "run", config={})
    path = rec.save_episode(0, *_fake_traj(n_steps=200))
    rec.close()
    size_kb = path.stat().st_size / 1024
    assert size_kb < 200, f"npz 体积 {size_kb:.1f} KB 超限"
