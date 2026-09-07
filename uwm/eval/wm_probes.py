"""P1 探针：RSSM 开环想象误差（SPEC_M2 §2.5）。

从 evaldir 的真实 episode 取前 5 步做 observe，再用真实动作序列开环想象
H 步，解码出 state 与真实后续 state 的前 6 维 [x,y,ψ,u,v,r] 对比 L2。

用法：
    python -m uwm.eval.wm_probes <run_dir> [--horizons 1,5,10,15]

run_dir 需包含 dv3_logdir/（上游训练产物，含 latest.pt 与 eval_eps/）和
config.yaml（train_dreamer 写入，含 dv3_overrides）。结果打印到 stdout 并
存 run_dir/p1_imagination_error.png。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from uwm.compat import ensure_upstream_compat
from uwm.eval.dv3 import add_vendored_path, apply_cpu_fallback, build_dv3_config

CONTEXT_STEPS = 5  # observe 上下文长度


def load_dreamer(logdir: str | Path, exp_cfg: dict):
    """加载上游 Dreamer agent（latest.pt），返回 agent。

    exp_cfg: exp yaml 字典（用其 dv3_overrides 重建模型结构配置）。
    """
    import torch

    ensure_upstream_compat()
    add_vendored_path()
    apply_cpu_fallback()
    import dreamer as dv3_dreamer
    import tools as dv3_tools

    logdir = Path(logdir)
    overrides = dict(exp_cfg["dv3_overrides"])
    overrides["logdir"] = str(logdir)
    overrides["device"] = "cpu"
    overrides["compile"] = False
    config = build_dv3_config(overrides)

    import envs.uwm as uwm_envs

    probe_env = uwm_envs.UWMStationKeeping("uwm_station_keeping", seed=0)
    obs_space, act_space = probe_env.observation_space, probe_env.action_space
    config.num_actions = int(act_space.shape[0])

    logger = dv3_tools.Logger(logdir, 0)
    agent = dv3_dreamer.Dreamer(obs_space, act_space, config, logger, dataset=None)
    ckpt_path = logdir / "latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.requires_grad_(requires_grad=False)
    agent.eval()
    return agent


def open_loop_imagination_error(
    agent, episodes, horizons: tuple[int, ...] = (1, 5, 10, 15)
) -> dict:
    """RSSM 开环想象误差。

    每条 episode：前 CONTEXT_STEPS 步 observe 得 posterior，再用真实动作
    imagine H 步，decoder 重建 state，与真实后续 state 前 6 维比 L2。
    返回 {horizon: mean_l2_error}（对可用 episode 取均值）。
    """
    import torch

    wm = agent._wm
    H = max(horizons)
    need = CONTEXT_STEPS + H  # 需要的 cache 长度（state 索引 0..need-1）
    per_step_errors: list[np.ndarray] = []  # 每条 episode 的 (H,) L2

    with torch.no_grad():
        for ep in episodes:
            if len(ep["reward"]) < need:
                continue
            data = {
                k: np.asarray(v[:need])[None] for k, v in ep.items()
            }  # (batch=1, time, ...)
            data = wm.preprocess(data)
            embed = wm.encoder(data)
            post, _ = wm.dynamics.observe(
                embed[:, :CONTEXT_STEPS],
                data["action"][:, :CONTEXT_STEPS],
                data["is_first"][:, :CONTEXT_STEPS],
            )
            init = {k: v[:, -1] for k, v in post.items()}
            prior = wm.dynamics.imagine_with_action(
                data["action"][:, CONTEXT_STEPS:CONTEXT_STEPS + H], init
            )
            feat = wm.dynamics.get_feat(prior)
            imagined = wm.heads["decoder"](feat)["state"].mode()[0, :, :6]  # (H, 6)
            truth = data["state"][0, CONTEXT_STEPS:CONTEXT_STEPS + H, :6]
            l2 = torch.linalg.norm(imagined - truth, dim=-1).cpu().numpy()  # (H,)
            per_step_errors.append(l2)

    if not per_step_errors:
        raise ValueError(
            f"没有长度 >= {need} 的 episode 可用于 P1 探针（共 {len(episodes)} 条）"
        )
    mean_l2 = np.mean(np.stack(per_step_errors), axis=0)  # (H,)
    return {h: float(mean_l2[h - 1]) for h in horizons}


def _plot_p1(errors: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted(errors)
    values = [errors[h] for h in horizons]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons, values, marker="o", color="tab:blue")
    ax.set_xlabel("imagination horizon (env steps, 0.1 s each)")
    ax.set_ylabel("mean L2 error on [x,y,psi,u,v,r]")
    ax.set_title("P1: open-loop imagination error")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_p1(run_dir: str | Path, horizons: tuple[int, ...] = (1, 5, 10, 15)) -> dict:
    """对 run_dir 跑 P1 探针：返回 {horizon: error} 并存 p1_imagination_error.png。"""
    ensure_upstream_compat()
    add_vendored_path()
    import tools as dv3_tools

    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    exp_cfg = {"dv3_overrides": run_cfg["dv3_overrides"]}
    logdir = run_dir / "dv3_logdir"

    agent = load_dreamer(logdir, exp_cfg)
    episodes = list(dv3_tools.load_episodes(logdir / "eval_eps").values())
    errors = open_loop_imagination_error(agent, episodes, horizons)
    out_path = run_dir / "p1_imagination_error.png"
    _plot_p1(errors, out_path)
    print(f"P1 open-loop imagination error: {errors}")
    print(f"P1 figure -> {out_path}")
    return errors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="runs/<exp_id>/seed<N>（含 dv3_logdir/ 与 config.yaml）")
    parser.add_argument("--horizons", default="1,5,10,15",
                        help="逗号分隔的想象步数（默认 1,5,10,15）")
    args = parser.parse_args(argv)
    horizons = tuple(int(x) for x in args.horizons.split(","))
    run_p1(args.run_dir, horizons)


if __name__ == "__main__":
    main()
