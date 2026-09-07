"""录制演示 episode 到 runs/demo/（供 notebooks/episode_replay.py 回放）。

一半 episode 用随机策略（发散），一半用手写 PD 向原点回正的启发式策略
（收敛），让回放面板能看到"发散 vs 收敛"的对比。只依赖 repo 内模块 + 标准库。

用法：python scripts/record_demo_episodes.py [--episodes 10] [--steps 200] [--out runs/demo]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

from uwm.envs.station_keeping import StationKeepingEnv
from uwm.eval.recorder import EpisodeRecorder


def make_env(root: Path) -> StationKeepingEnv:
    """用 bluerov2.yaml 构建环境（录制参数与回放契约一致）。"""
    with open(root / "configs" / "vehicle" / "bluerov2.yaml", encoding="utf-8") as f:
        vehicle_cfg = yaml.safe_load(f)
    return StationKeepingEnv(vehicle_cfg, {"init_radius": 5.0})


def pd_action(obs: np.ndarray, tau_max: np.ndarray) -> np.ndarray:
    """简单 PD 向原点回正：世界系 PD 力 → 旋转到机体系 → 按 tau_max 归一化。"""
    x, y, psi, u, v, r, _ = obs
    kp, kd, kp_psi, kd_psi = 30.0, 40.0, 6.0, 4.0
    c, s = math.cos(psi), math.sin(psi)
    fx_w = -kp * x - kd * (c * u - s * v)  # 世界系速度 = J(ψ)ν
    fy_w = -kp * y - kd * (s * u + c * v)
    tau = np.array([c * fx_w + s * fy_w, -s * fx_w + c * fy_w,
                    -kp_psi * psi - kd_psi * r])
    return np.clip(tau / tau_max, -1.0, 1.0)


def run_episode(env: StationKeepingEnv, policy, n_steps: int, seed: int):
    """跑一条 episode，返回 (t, eta, nu, action, reward) 轨迹数组与累计回报。

    约定：第 k 项记录执行 action[k] 之前的状态（t=k·控制周期），
    因此 eta[0] 为 reset 初始状态；reward[k] 为该步收到的奖励。
    """
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    ts, etas, nus, actions, rewards = [], [], [], [], []
    ep_return = 0.0
    for k in range(n_steps):
        action = policy(obs, rng)
        ts.append(env._t)
        etas.append(env._eta.numpy().copy())
        nus.append(env._nu.numpy().copy())
        actions.append(np.asarray(action, dtype=np.float64))
        obs, reward, _, _, _ = env.step(action)
        rewards.append(reward)
        ep_return += reward
    traj = (np.array(ts), np.array(etas), np.array(nus),
            np.array(actions), np.array(rewards))
    return traj, ep_return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10, help="episode 数（前一半随机，后一半 PD）")
    parser.add_argument("--steps", type=int, default=200, help="每 episode 步数")
    parser.add_argument("--out", type=str, default="runs/demo", help="输出 run 目录")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    env = make_env(root)
    tau_max = np.asarray(env.vehicle_cfg["tau_max"], dtype=np.float64)

    run_dir = root / args.out
    recorder = EpisodeRecorder(
        run_dir,
        config={
            "vehicle": "bluerov2",
            "policy": "half random / half PD-to-origin",
            "episodes": args.episodes,
            "steps_per_episode": args.steps,
            "init_radius": 5.0,
        },
    )
    for ep in range(args.episodes):
        use_pd = ep >= args.episodes // 2
        if use_pd:
            policy = lambda obs, rng: pd_action(obs, tau_max)  # noqa: E731
        else:
            policy = lambda obs, rng: rng.uniform(-1.0, 1.0, size=3)  # noqa: E731
        traj, ep_return = run_episode(env, policy, args.steps, seed=ep)
        recorder.save_episode(ep, *traj)
        final_dist = float(np.linalg.norm(traj[1][-1][:2]))
        recorder.log_metrics(
            step=ep, episode=ep, policy="pd" if use_pd else "random",
            return_=round(ep_return, 4), final_dist=round(final_dist, 4),
        )
        print(f"ep {ep:3d} [{'pd' if use_pd else 'random'}] "
              f"return={ep_return:8.2f} final_dist={final_dist:.3f} m")
    recorder.close()
    print(f"已写入 {run_dir}")


if __name__ == "__main__":
    main()
