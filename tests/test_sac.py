"""SAC + ReplayBuffer 测试（SPEC_M1 §3）。"""

from __future__ import annotations

import math

import numpy as np
import torch
import yaml

from uwm.algos.sac import SAC, ReplayBuffer
from uwm.envs.station_keeping import StationKeepingEnv

TINY_CFG = {
    "hidden": 32, "lr": 3e-4, "gamma": 0.99, "tau": 0.005,
    "batch_size": 32, "buffer_size": 500, "warmup_steps": 200,
}


def _make_env() -> StationKeepingEnv:
    """tiny 环境（episode 200 步，加快测试）。"""
    with open("configs/vehicle/bluerov2.yaml", encoding="utf-8") as f:
        vehicle_cfg = yaml.safe_load(f)
    return StationKeepingEnv(vehicle_cfg, {"max_episode_steps": 200})


def test_smoke_training_finite_losses():
    """冒烟：tiny 配置训练 1500 步跑通，loss 全部 finite。"""
    env = _make_env()
    agent = SAC(7, 3, TINY_CFG)
    buffer = ReplayBuffer(TINY_CFG["buffer_size"], 7, 3)
    torch.manual_seed(0)
    np.random.seed(0)

    obs, _ = env.reset(seed=0)
    obs = obs.astype(np.float32)
    last = None
    for step in range(1500):
        if step < TINY_CFG["warmup_steps"]:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = next_obs.astype(np.float32)
        # done 用 terminated；truncated 按未终止处理（TimeLimit bootstrap）
        buffer.add(obs, action, float(reward), next_obs, bool(terminated))
        obs = next_obs
        if step >= TINY_CFG["warmup_steps"] and len(buffer) >= TINY_CFG["batch_size"]:
            last = agent.update(buffer.sample(TINY_CFG["batch_size"]))
        if terminated or truncated:
            obs, _ = env.reset()
            obs = obs.astype(np.float32)

    assert last is not None, "1500 步内未触发任何 update"
    for k in ("q_loss", "actor_loss", "alpha", "alpha_loss"):
        assert math.isfinite(last[k]), f"{k} 非 finite: {last[k]}"
    assert last["alpha"] > 0.0


def test_action_bounds():
    """动作界：select_action 输出恒 ∈ [−1,1]（含极端 obs 与确定性模式）。"""
    agent = SAC(7, 3, TINY_CFG)
    rng = np.random.default_rng(0)
    obs_list = [rng.normal(size=7) * s for s in (1.0, 1e3, 1e6)]
    obs_list.append(np.zeros(7))
    for obs in obs_list:
        for deterministic in (False, True):
            for _ in range(20):
                a = agent.select_action(obs.astype(np.float32), deterministic=deterministic)
                assert a.shape == (3,)
                assert np.all(a <= 1.0) and np.all(a >= -1.0), f"动作越界: {a}"
                assert np.all(np.isfinite(a))


def test_ckpt_consistency(tmp_path):
    """ckpt：save→load 后同 obs 确定性动作逐位相同；resume 后 update 正常。"""
    agent = SAC(7, 3, TINY_CFG)
    torch.manual_seed(1)
    np.random.seed(1)
    buffer = ReplayBuffer(TINY_CFG["buffer_size"], 7, 3)
    rng = np.random.default_rng(2)
    for _ in range(100):
        o = rng.normal(size=7).astype(np.float32)
        a = rng.uniform(-1, 1, size=3).astype(np.float32)
        buffer.add(o, a, -1.0, o, False)
    for _ in range(5):  # 让优化器/alpha/rng 状态偏离初始值
        agent.update(buffer.sample(TINY_CFG["batch_size"]))

    path = tmp_path / "ckpt.pt"
    agent.save(path)
    obs = rng.normal(size=7).astype(np.float32)
    a_before = agent.select_action(obs, deterministic=True)

    agent2 = SAC(7, 3, TINY_CFG)
    agent2.load(path)
    a_after = agent2.select_action(obs, deterministic=True)
    np.testing.assert_array_equal(a_before, a_after)  # 逐位相同
    assert agent2.alpha == agent.alpha

    # resume 后 update 正常（loss finite，参数继续变化）
    out = agent2.update(buffer.sample(TINY_CFG["batch_size"]))
    for k, v in out.items():
        assert math.isfinite(v), f"resume 后 {k} 非 finite: {v}"


def test_replay_buffer_ring():
    """buffer 环形：超过 capacity 后覆盖正确（保留最新 capacity 条），sample 形状正确。"""
    cap = 10
    buf = ReplayBuffer(cap, obs_dim=2, act_dim=1)
    # 写入 cap + 3 条可辨识数据：obs = [i, i]
    for i in range(cap + 3):
        o = np.array([i, i], dtype=np.float32)
        buf.add(o, np.array([0.5], dtype=np.float32), float(i), o + 1, i % 2 == 0)
    assert len(buf) == cap
    assert buf._ptr == 3  # 环形指针：写了 13 条，下一条写位置 3
    # 覆盖正确：槽位 3..9 是第 3..9 条，槽位 0..2 被第 10..12 条覆盖
    for slot in range(cap):
        expected = slot if slot >= 3 else slot + cap
        np.testing.assert_array_equal(buf.obs[slot], [expected, expected])
        np.testing.assert_array_equal(buf.next_obs[slot], [expected + 1, expected + 1])
        assert buf.rew[slot] == float(expected)
        assert buf.done[slot] == float(expected % 2 == 0)

    batch = buf.sample(32)
    assert batch["obs"].shape == (32, 2)
    assert batch["act"].shape == (32, 1)
    assert batch["next_obs"].shape == (32, 2)
    assert batch["rew"].shape == (32,)
    assert batch["done"].shape == (32,)
    assert batch["obs"].dtype == torch.float32
    # 采样到的都是被保留的最新数据（rew >= 3）
    assert float(batch["rew"].min()) >= 3.0
