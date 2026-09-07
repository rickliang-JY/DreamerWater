"""上游 dv3 episode npz → UWM recorder schema（SPEC_M2 §2.4）。

上游 episode schema（对照 third_party/dreamerv3-torch/tools.py 的
simulate / add_to_cache / save_episodes，长度均为 T+1）：

    state[k]      第 k 步后的观测（k=0 为 reset 初始观测），(T+1, 7) float32
    action[k]     k=0 为补位全零；k>=1 为从 state[k-1] 施加的动作，(T+1, 3)
    reward[k]     k=0 为 0；k>=1 为第 k 步奖励，(T+1,)
    discount[k]   截断 episode 恒 1.0（本环境 terminated 恒 False）
    is_first/is_terminal/image/logprob  辅助键（logprob 仅策略步有）

我们的 recorder schema（uwm/eval/recorder.py，长度 T，与 M1 一致）：

    t[k]      = k * control_period（dt 0.02s × 内部 action_repeat 5 = 0.1s）
    eta[k]    = state[k][0:3]，nu[k] = state[k][3:6]
    action[k] = 上游 action[k+1]（施加于 state[k] 的动作）
    reward[k] = 上游 reward[k+1]
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

METRICS_HEADER = [
    "step", "episode", "env_steps", "return_", "final_dist",
    "ep_len", "q_loss", "actor_loss", "alpha", "is_eval",
]  # 与 M1 metrics.csv 同表头


def _control_period(run_dir: Path | None) -> float:
    """控制周期（s）：优先读 run_dir/config.yaml 的 vehicle，缺省读仓库 bluerov2。"""
    from uwm.eval.dv3 import REPO_ROOT

    vehicle_rel = None
    if run_dir is not None and (run_dir / "config.yaml").exists():
        with open(run_dir / "config.yaml", encoding="utf-8") as f:
            vehicle_rel = (yaml.safe_load(f) or {}).get("vehicle")
    vehicle_path = REPO_ROOT / (vehicle_rel or "configs/vehicle/bluerov2.yaml")
    with open(vehicle_path, encoding="utf-8") as f:
        vehicle_cfg = yaml.safe_load(f)
    return float(vehicle_cfg["dt"]) * int(vehicle_cfg["action_repeat"])


def _convert_episode(dv3_ep: dict, out_path: Path, control_period: float) -> dict:
    """单条 dv3 episode → recorder npz；返回该行 metrics 需要的标量。"""
    state = np.asarray(dv3_ep["state"], dtype=np.float32)  # (T+1, 7)
    action = np.asarray(dv3_ep["action"], dtype=np.float32)  # (T+1, 3)
    reward = np.asarray(dv3_ep["reward"], dtype=np.float64)  # (T+1,)
    T = len(reward) - 1
    t = np.arange(T, dtype=np.float64) * control_period
    np.savez(
        out_path,
        t=t,
        eta=state[:T, 0:3],
        nu=state[:T, 3:6],
        action=action[1:T + 1],
        reward=reward[1:T + 1],
    )
    return {
        "return_": round(float(reward[1:].sum()), 4),
        "final_dist": round(float(state[T, 6]), 4),
        "ep_len": T,
    }


def _read_metrics_jsonl(logdir: Path) -> tuple[list[dict], list[dict]]:
    """读 logdir/metrics.jsonl → (train 行列表, eval 轮次列表)。"""
    train_rows, eval_rounds = [], []
    path = logdir / "metrics.jsonl"
    if not path.exists():
        return train_rows, eval_rounds
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if "eval_return" in entry:
                eval_rounds.append(entry)
            elif "train_return" in entry:
                train_rows.append(entry)
    return train_rows, eval_rounds


def convert_logdir(logdir: str | Path, run_dir: str | Path | None = None) -> dict:
    """把 dv3 logdir 的 eval episodes 转成 recorder schema + metrics.csv。

    logdir:  上游 dv3_logdir（含 eval_eps/ train_eps/ metrics.jsonl）
    run_dir: 输出目录（缺省取 logdir 的父目录）；episodes 写到
             run_dir/episodes/ep_XXXX.npz，指标写 run_dir/metrics.csv。
    返回摘要 dict（转换条数、输出路径等）。
    """
    logdir = Path(logdir)
    run_dir = Path(run_dir) if run_dir is not None else logdir.parent
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    control_period = _control_period(run_dir)

    evaldir = logdir / "eval_eps"
    eval_files = sorted(evaldir.glob("*.npz")) if evaldir.exists() else []
    train_rows, eval_rounds = _read_metrics_jsonl(logdir)
    n_per_round = int(eval_rounds[0]["eval_episodes"]) if eval_rounds else 0

    rows: list[dict] = []
    for entry in train_rows:
        rows.append({
            "step": entry["step"], "episode": "", "env_steps": entry["step"],
            "return_": round(float(entry["train_return"]), 4), "final_dist": "",
            "ep_len": entry.get("train_length", ""), "q_loss": "",
            "actor_loss": "", "alpha": "", "is_eval": 0,
        })

    n_converted = 0
    for j, path in enumerate(eval_files):
        with np.load(path) as z:
            dv3_ep = {k: z[k] for k in z.keys()}
        scalars = _convert_episode(
            dv3_ep, episodes_dir / f"ep_{j:04d}.npz", control_period
        )
        if eval_rounds and n_per_round:
            round_idx = min(j // n_per_round, len(eval_rounds) - 1)
            step = eval_rounds[round_idx]["step"]
        else:
            step = j
        rows.append({
            "step": step, "episode": j, "env_steps": step,
            "q_loss": "", "actor_loss": "", "alpha": "", "is_eval": 1,
            **scalars,
        })
        n_converted += 1

    rows.sort(key=lambda r: (int(r["env_steps"]), -int(r["is_eval"])))
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "logdir": str(logdir),
        "run_dir": str(run_dir),
        "episodes_dir": str(episodes_dir),
        "metrics_csv": str(run_dir / "metrics.csv"),
        "n_eval_episodes": n_converted,
        "n_metrics_rows": len(rows),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="dv3 logdir（含 eval_eps/）或 run_dir（其下含 dv3_logdir/）",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if (path / "eval_eps").exists() or (path / "train_eps").exists():
        logdir, run_dir = path, path.parent
    elif (path / "dv3_logdir").exists():
        logdir, run_dir = path / "dv3_logdir", path
    else:
        raise SystemExit(f"找不到 dv3 logdir：{path}（既无 eval_eps/ 也无 dv3_logdir/）")

    summary = convert_logdir(logdir, run_dir)
    print(f"converted {summary['n_eval_episodes']} eval episodes "
          f"-> {summary['episodes_dir']}")
    print(f"metrics.csv ({summary['n_metrics_rows']} rows) -> {summary['metrics_csv']}")


if __name__ == "__main__":
    main()
