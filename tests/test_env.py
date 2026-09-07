"""环境测试（SPEC §3.3）：check_env、随机策略稳定性、reset 可复现、action_repeat。"""

import os

import numpy as np
import pytest
import yaml
from gymnasium.utils.env_checker import check_env

from uwm.envs.station_keeping import StationKeepingEnv

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "..", "configs", "vehicle", "bluerov2.yaml")


def _make_env(**task_overrides):
    with open(CFG_PATH) as f:
        vehicle_cfg = yaml.safe_load(f)
    task_cfg = {"init_radius": 5.0, "max_episode_steps": 500}
    task_cfg.update(task_overrides)
    return StationKeepingEnv(vehicle_cfg, task_cfg)


def test_check_env():
    """gymnasium 官方 env checker 必须通过。"""
    check_env(_make_env(), skip_render_check=True)


def test_random_policy_bounded_1000_steps():
    """随机策略 1000 步不发散：|x|,|y| 有界（< 1e3），无 NaN/Inf。"""
    env = _make_env(max_episode_steps=10000)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        action = rng.uniform(-1.0, 1.0, size=3)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)), f"non-finite obs: {obs}"
        assert np.isfinite(reward)
        assert abs(obs[0]) < 1e3 and abs(obs[1]) < 1e3, f"diverged: {obs[:2]}"
        if terminated or truncated:
            env.reset()


def test_reset_reproducible():
    """同 seed 两次 reset 观测相同。"""
    env = _make_env()
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
    # 不同 seed 应得到（几乎必然）不同初始观测
    obs3, _ = env.reset(seed=43)
    assert not np.array_equal(obs1, obs3)


def test_action_repeat_advances_physical_time():
    """env.step 一次，内部物理时间前进 dt × action_repeat。"""
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert info["t"] == 0.0
    expected = env.dt * env.action_repeat
    _, _, _, _, info = env.step(np.zeros(3))
    assert info["t"] == pytest.approx(expected)
    _, _, _, _, info = env.step(np.zeros(3))
    assert info["t"] == pytest.approx(2.0 * expected)


def test_truncation_at_max_episode_steps():
    """truncated: 步数达到 max_episode_steps；terminated 恒为 False。"""
    env = _make_env(max_episode_steps=10)
    env.reset(seed=0)
    truncated = False
    for i in range(10):
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        assert terminated is False
        assert truncated == (i == 9)
    assert truncated
