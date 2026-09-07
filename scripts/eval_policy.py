"""确定性策略评估 + M1 通过判据（SPEC_M1 §2.4）。

用法：
    python scripts/eval_policy.py <run_dir> [--episodes 20]

从 run_dir/config.yaml 重建环境与 SAC（ckpt/latest.pt），用确定性策略
（tanh(μ)，不采样）跑 N 个 episode，episodes 存到 <run_dir>/eval/episodes/
（面板 B 可直接回放）。

输出指标：
- final_dist 均值 / 中位数 / max（episode 结束时刻到目标的距离，m）
- 最后 5 秒位置 std（振荡指标）：每 episode 取最后 5 s 的 (x, y)，
  std = sqrt(std(x)² + std(y)²)，报告均值 / max
- return 均值

M1 通过判据（同时满足判定 PASS，否则 FAIL）：
1. final_dist 中位数 < 0.3 m
2. 最后 5 秒位置 std 均值 < 0.05 m（无持续振荡）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from uwm.algos.sac import SAC
from uwm.envs.station_keeping import StationKeepingEnv
from uwm.eval.recorder import EpisodeRecorder

ROOT = Path(__file__).resolve().parents[1]

# M1 通过判据阈值（m）
FINAL_DIST_MEDIAN_MAX = 0.3
POS_STD_LAST5S_MEAN_MAX = 0.05


def position_std_last5s(eta: np.ndarray, control_period: float) -> float:
    """最后 5 s 位置振荡指标：sqrt(std(x)² + std(y)²)（m）。

    eta: (T, 3) 轨迹 [x, y, ψ]；control_period: 控制周期（s）。
    episode 不足 5 s 时用整段。
    """
    n = max(2, int(round(5.0 / control_period)))
    xy = eta[-n:, :2]
    return float(np.sqrt(np.std(xy[:, 0]) ** 2 + np.std(xy[:, 1]) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str, help="runs/<exp_id>/seed<N>/ 目录")
    parser.add_argument("--episodes", type=int, default=20, help="评估 episode 数")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    with open(run_dir / "config.yaml", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)

    with open(ROOT / run_cfg["vehicle"], encoding="utf-8") as f:
        vehicle_cfg = yaml.safe_load(f)
    env = StationKeepingEnv(vehicle_cfg, dict(run_cfg.get("task") or {}))

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    agent = SAC(obs_dim, act_dim, dict(run_cfg["sac"]), device="cpu")
    ckpt_path = run_dir / "ckpt" / "latest.pt"
    agent.load(ckpt_path)
    print(f"加载 ckpt: {ckpt_path}（确定性策略，{args.episodes} episodes）")

    recorder = EpisodeRecorder(
        run_dir / "eval",
        config={"source_run": str(run_dir.relative_to(ROOT)),
                "ckpt": "ckpt/latest.pt", "episodes": args.episodes,
                "policy": "deterministic (tanh(mu))"},
    )

    returns, final_dists, pos_stds = [], [], []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            ts, etas, nus, actions, rewards = [], [], [], [], []
            ep_return = 0.0
            while True:
                action = agent.select_action(obs.astype(np.float32), deterministic=True)
                ts.append(env._t)
                etas.append(env._eta.numpy().copy())
                nus.append(env._nu.numpy().copy())
                actions.append(np.asarray(action, dtype=np.float64))
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                ep_return += reward
                if terminated or truncated:
                    break
            traj = tuple(np.array(a) for a in (ts, etas, nus, actions, rewards))
            recorder.save_episode(ep, *traj)
            final_dist = float(info["dist_to_goal"])
            pos_std = position_std_last5s(traj[1], env.control_period)
            returns.append(ep_return)
            final_dists.append(final_dist)
            pos_stds.append(pos_std)
            recorder.log_metrics(
                step=ep, episode=ep, return_=round(ep_return, 4),
                final_dist=round(final_dist, 4), pos_std_last5s=round(pos_std, 5),
                ep_len=len(ts),
            )
            print(f"ep {ep:3d} return={ep_return:9.2f} "
                  f"final_dist={final_dist:6.3f} pos_std_5s={pos_std:6.4f}")
    finally:
        recorder.close()

    returns = np.asarray(returns)
    final_dists = np.asarray(final_dists)
    pos_stds = np.asarray(pos_stds)
    med_dist = float(np.median(final_dists))
    mean_std = float(np.mean(pos_stds))
    pass_final = med_dist < FINAL_DIST_MEDIAN_MAX
    pass_std = mean_std < POS_STD_LAST5S_MEAN_MAX

    print("\n==== 评估汇总 ====")
    print(f"return      mean={returns.mean():.2f}")
    print(f"final_dist  mean={final_dists.mean():.3f} median={med_dist:.3f} "
          f"max={final_dists.max():.3f} (m)")
    print(f"pos_std_5s  mean={mean_std:.4f} max={pos_stds.max():.4f} (m)")
    print("==== M1 通过判据 ====")
    print(f"[{'PASS' if pass_final else 'FAIL'}] final_dist 中位数 {med_dist:.3f} "
          f"< {FINAL_DIST_MEDIAN_MAX} m")
    print(f"[{'PASS' if pass_std else 'FAIL'}] 最后 5 s 位置 std 均值 {mean_std:.4f} "
          f"< {POS_STD_LAST5S_MEAN_MAX} m")
    verdict = "PASS" if (pass_final and pass_std) else "FAIL"
    print(f"M1 判据结论: {verdict}")
    print(f"eval episodes 已存到 {run_dir / 'eval' / 'episodes'}")
    return 0 if (pass_final and pass_std) else 1


if __name__ == "__main__":
    sys.exit(main())
