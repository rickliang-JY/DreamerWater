"""SAC 训练入口（SPEC_M1 §2.3）。

用法：
    python scripts/train.py <exp.yaml> [--seed N] [--resume <ckpt>] [--total-steps M]

- 每个 (exp_id, seed) 一个 run 目录：runs/<exp_id>/seed<N>/，复用 EpisodeRecorder
- metrics.csv 每 episode 追加：episode, env_steps, return_, final_dist, ep_len,
  q_loss, actor_loss, alpha, is_eval（eval 行为 is_eval=1）
- 每 eval_every 步（episode 边界对齐）跑一次确定性评估
- 每 ckpt_every 步存 ckpt/latest.pt（含网络/优化器/buffer/env_steps/rng），
  --resume 续训（沙箱超时友好：分段跑）
- 结束画样本效率曲线 runs/<exp_id>/learning_curve_seed<N>.png
  （eval 点 + 训练 return 滑动平均，窗口 20；headless 无 CJK 字体，用英文标注）

done 语义：buffer 里存 terminated；TimeLimit 截断（truncated）按未终止处理，
其 bootstrap 照常进行（gymnasium TimeLimit 的正确做法）。本环境 terminated 恒
False，因此所有转移都做 bootstrap。
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from uwm.algos.sac import SAC, ReplayBuffer
from uwm.envs.station_keeping import StationKeepingEnv
from uwm.eval.recorder import EpisodeRecorder

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    """取当前 git commit hash（取不到则返回 'unknown'）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — 无 git 环境不阻塞训练
        return "unknown"


def make_env(exp_cfg: dict) -> StationKeepingEnv:
    """按 exp 配置构建 StationKeepingEnv。"""
    vehicle_path = ROOT / exp_cfg["vehicle"]
    with open(vehicle_path, encoding="utf-8") as f:
        vehicle_cfg = yaml.safe_load(f)
    return StationKeepingEnv(vehicle_cfg, dict(exp_cfg.get("task") or {}))


def run_eval_episode(env: StationKeepingEnv, agent: SAC, seed: int | None = None):
    """确定性策略跑一条 episode，返回 (return, final_dist, ep_len)。"""
    obs, _ = env.reset(seed=seed)
    ep_return, ep_len = 0.0, 0
    while True:
        action = agent.select_action(obs.astype(np.float32), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_len += 1
        if terminated or truncated:
            break
    return ep_return, float(info["dist_to_goal"]), ep_len


def save_ckpt(path: Path, agent: SAC, buffer: ReplayBuffer, env_steps: int,
              episode: int, env: StationKeepingEnv) -> None:
    """组合 ckpt：SAC 全部状态 + buffer + 训练进度 + 环境 RNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sac": agent._state_dict(),
            "buffer": buffer.state_dict(),
            "env_steps": env_steps,
            "episode": episode,
            "env_rng": env.np_random.bit_generator.state,
        },
        path,
    )


def plot_learning_curve(metrics_path: Path, out_path: Path, window: int = 20) -> None:
    """画样本效率曲线：训练 return 滑动平均（窗口 20）+ eval 点描点。

    headless 环境无 CJK 字体，统一用英文标注（SPEC_M1 工程约定）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_steps, train_ret = [], []
    eval_points: dict[int, list[float]] = {}
    with open(metrics_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps = int(row["env_steps"])
            ret = float(row["return_"])
            if row.get("is_eval") == "1":
                eval_points.setdefault(steps, []).append(ret)
            else:
                train_steps.append(steps)
                train_ret.append(ret)

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_steps:
        train_ret_arr = np.asarray(train_ret)
        if len(train_ret_arr) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(train_ret_arr, kernel, mode="valid")
            smooth_steps = np.asarray(train_steps)[window - 1:]
            ax.plot(smooth_steps, smooth, label=f"train return (MA-{window})",
                    color="tab:blue")
            ax.plot(train_steps, train_ret, color="tab:blue", alpha=0.15, linewidth=0.5)
        else:  # episode 数不足一个窗口：直接画原始 return
            ax.plot(train_steps, train_ret, marker=".", color="tab:blue",
                    label="train return (raw)")
    for steps in sorted(eval_points):
        rets = eval_points[steps]
        mean_ret = float(np.mean(rets))
        ax.errorbar([steps], [mean_ret], yerr=[float(np.std(rets))],
                    fmt="o", color="tab:red", capsize=3,
                    label="eval (deterministic)" if steps == sorted(eval_points)[0] else None)
    ax.set_xlabel("env steps")
    ax.set_ylabel("episode return")
    ax.set_title(out_path.stem)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="exp 配置 yaml 路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（默认取 seeds[0]）")
    parser.add_argument("--resume", type=str, default=None, help="续训 ckpt 路径（ckpt/latest.pt）")
    parser.add_argument("--total-steps", type=int, default=None, help="覆盖 total_steps（分段跑用）")
    args = parser.parse_args()

    with open(ROOT / args.config if not Path(args.config).is_absolute() else args.config,
              encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    exp_id = exp_cfg["exp_id"]
    seed = int(args.seed if args.seed is not None else exp_cfg["seeds"][0])
    total_steps = int(args.total_steps or exp_cfg["total_steps"])
    eval_every = int(exp_cfg["eval_every"])
    eval_episodes = int(exp_cfg["eval_episodes"])
    record_ep_every = int(exp_cfg["record_ep_every"])
    ckpt_every = int(exp_cfg.get("ckpt_every", 2000))
    sac_cfg = dict(exp_cfg["sac"])

    # ---- 训练侧 RNG 播种（resume 时会被 ckpt 内 rng 覆盖）----
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(exp_cfg)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    agent = SAC(obs_dim, act_dim, sac_cfg, device="cpu")
    buffer = ReplayBuffer(agent.buffer_size, obs_dim, act_dim)

    run_dir = ROOT / "runs" / exp_id / f"seed{seed}"
    env_steps, episode = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        agent._load_state_dict(ckpt["sac"])
        buffer.load_state_dict(ckpt["buffer"])
        env_steps = int(ckpt["env_steps"])
        episode = int(ckpt["episode"])
        env.np_random.bit_generator.state = ckpt["env_rng"]
        print(f"[resume] {args.resume} -> env_steps={env_steps} episode={episode}")

    recorder = EpisodeRecorder(
        run_dir,
        config={
            "exp_id": exp_id, "seed": seed, "git_commit": git_commit(),
            "vehicle": exp_cfg["vehicle"], "task": exp_cfg.get("task") or {},
            "total_steps": total_steps, "eval_every": eval_every,
            "eval_episodes": eval_episodes, "record_ep_every": record_ep_every,
            "sac": sac_cfg,
        },
        resume=bool(args.resume),
    )

    warmup = agent.warmup_steps
    next_eval = ((env_steps // eval_every) + 1) * eval_every
    next_ckpt = ((env_steps // ckpt_every) + 1) * ckpt_every
    t0 = time.time()

    def log_episode(ep: int, steps: int, ret: float, dist: float, ep_len: int,
                    losses: dict[str, list[float]], is_eval: int) -> None:
        """写一行 metrics.csv（train 行带 loss 均值；eval 行 loss 留空）。"""
        recorder.log_metrics(
            step=steps, episode=ep, env_steps=steps, return_=round(ret, 4),
            final_dist=round(dist, 4), ep_len=ep_len,
            q_loss=round(float(np.mean(losses["q"])), 6) if losses["q"] else "",
            actor_loss=round(float(np.mean(losses["a"])), 6) if losses["a"] else "",
            alpha=round(agent.alpha, 6), is_eval=is_eval,
        )

    try:
        obs, _ = env.reset(seed=seed if not args.resume else None)
        obs = obs.astype(np.float32)
        ep_return, ep_len = 0.0, 0
        losses: dict[str, list[float]] = {"q": [], "a": []}
        recording = episode % record_ep_every == 0
        traj: dict[str, list] = {"t": [], "eta": [], "nu": [], "action": [], "reward": []}

        while env_steps < total_steps:
            # warmup 期间均匀随机动作；之后走策略
            if env_steps < warmup:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = agent.select_action(obs)

            if recording:
                traj["t"].append(env._t)
                traj["eta"].append(env._eta.numpy().copy())
                traj["nu"].append(env._nu.numpy().copy())
                traj["action"].append(np.asarray(action, dtype=np.float64))

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs.astype(np.float32)
            # done 只用 terminated；truncated（TimeLimit）按未终止处理，照常 bootstrap
            buffer.add(obs, action, float(reward), next_obs, bool(terminated))
            obs = next_obs
            env_steps += 1
            ep_return += reward
            ep_len += 1
            if recording:
                traj["reward"].append(float(reward))

            # warmup 结束后每步一次 update
            if env_steps >= warmup and len(buffer) >= agent.batch_size:
                out = agent.update(buffer.sample(agent.batch_size))
                losses["q"].append(out["q_loss"])
                losses["a"].append(out["actor_loss"])

            if env_steps % ckpt_every == 0:
                save_ckpt(run_dir / "ckpt" / "latest.pt", agent, buffer,
                          env_steps, episode, env)

            if terminated or truncated:
                final_dist = float(info["dist_to_goal"])
                if recording:
                    recorder.save_episode(
                        episode, traj["t"], traj["eta"], traj["nu"],
                        traj["action"], traj["reward"],
                    )
                log_episode(episode, env_steps, ep_return, final_dist, ep_len,
                            losses, is_eval=0)
                print(f"ep {episode:4d} steps={env_steps:7d} "
                      f"return={ep_return:9.2f} final_dist={final_dist:6.3f} "
                      f"alpha={agent.alpha:.4f}")
                episode += 1
                obs, _ = env.reset()
                obs = obs.astype(np.float32)
                ep_return, ep_len = 0.0, 0
                losses = {"q": [], "a": []}
                recording = episode % record_ep_every == 0
                traj = {"t": [], "eta": [], "nu": [], "action": [], "reward": []}

                # episode 边界对齐的周期性确定性评估
                if env_steps >= next_eval and env_steps < total_steps:
                    for k in range(eval_episodes):
                        eret, edist, elen = run_eval_episode(env, agent)
                        log_episode(episode + k, env_steps, eret, edist, elen,
                                    {"q": [], "a": []}, is_eval=1)
                    print(f"[eval @ {env_steps}] {eval_episodes} eps done")
                    next_eval += eval_every
                    obs, _ = env.reset()
                    obs = obs.astype(np.float32)

        save_ckpt(run_dir / "ckpt" / "latest.pt", agent, buffer, env_steps, episode, env)
    finally:
        recorder.close()

    curve_path = ROOT / "runs" / exp_id / f"learning_curve_seed{seed}.png"
    plot_learning_curve(run_dir / "metrics.csv", curve_path, window=20)
    dt = time.time() - t0
    print(f"训练结束：env_steps={env_steps} episodes={episode} 用时 {dt:.1f}s")
    print(f"ckpt: {run_dir / 'ckpt' / 'latest.pt'}")
    print(f"learning curve: {curve_path}")


if __name__ == "__main__":
    sys.exit(main())
