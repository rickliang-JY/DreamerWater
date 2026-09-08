"""P1/P2 探针（SPEC_M2 §2.5 / SPEC_M4 §2）。

P1：RSSM 开环想象误差。从真实 episode 取前 5 步做 observe，再用真实动作
序列开环想象 H 步，解码出 state 与真实后续 state 的前 6 维 [x,y,ψ,u,v,r]
对比 L2。

P2：RSSM 确定性隐状态 h_t → 洋流真值 [u_c, v_c] 的线性探针。检验 world
model 是否把**不可观测**的时变洋流编码进隐状态（world model 相对 SAC
唯一的结构性优势，且不需要策略收敛即可测量）。洋流真值经由 dv3 适配器
的 current_gt 暗通道（SPEC_M4 §1.4）随 episode cache 落盘，探针直接从
logdir 读取，模型本身从未见过该通道（encoder/decoder mlp_keys='state'）。

用法：
    python -m uwm.eval.wm_probes <run_dir> [--horizons 1,5,10,15]   # P1
    python -m uwm.eval.wm_probes <run_dir> --p2 [--test-ratio 0.3]   # P2

run_dir 需包含 dv3_logdir/（上游训练产物，含 latest.pt 与 eval_eps/）和
config.yaml（train_dreamer 写入，含 dv3_overrides）。P1 结果存
run_dir/p1_imagination_error.png；P2 结果存 run_dir/p2_probe.png。

P2 判读标准：R² > 0.5 = RSSM 明确编码洋流；0.1–0.5 = 弱编码；
< 0.1 = 未编码（若策略已收敛则更说明问题）。对照基线为瞬时 state obs
（7 维）的同样回归，应 ≈ 0——单帧观测不含洋流信息。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from uwm.compat import ensure_upstream_compat
from uwm.eval.dv3 import (
    REPO_ROOT,
    add_vendored_path,
    apply_cpu_fallback,
    build_dv3_config,
)

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

    # 探针环境只提供 obs/act space，但载体配置须与训练 run 一致
    # （OU run 的 current_gt 暗通道语义相同，space 形状不依赖于载体）。
    import os

    vehicle = exp_cfg.get("vehicle")
    if vehicle:
        vpath = Path(vehicle)
        os.environ["UWM_VEHICLE_CFG"] = str(
            vpath if vpath.is_absolute() else REPO_ROOT / vpath
        )
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
    exp_cfg = {
        "dv3_overrides": run_cfg["dv3_overrides"],
        "vehicle": run_cfg.get("vehicle"),
    }
    logdir = run_dir / "dv3_logdir"

    agent = load_dreamer(logdir, exp_cfg)
    episodes = list(dv3_tools.load_episodes(logdir / "eval_eps").values())
    errors = open_loop_imagination_error(agent, episodes, horizons)
    out_path = run_dir / "p1_imagination_error.png"
    _plot_p1(errors, out_path)
    print(f"P1 open-loop imagination error: {errors}")
    print(f"P1 figure -> {out_path}")
    return errors


def _ridge_fit_predict(X_train, Y_train, X_pred, alpha: float):
    """Ridge 闭式解（含截距，训练集中心化）。返回 X_pred 上的预测。"""
    mx = X_train.mean(axis=0, keepdims=True)
    my = Y_train.mean(axis=0, keepdims=True)
    Xc = X_train - mx
    Yc = Y_train - my
    gram = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    W = np.linalg.solve(gram, Xc.T @ Yc)
    return (X_pred - mx) @ W + my


def _r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """逐维 R² = 1 − SS_res/SS_tot（SS_tot 相对 y_true 均值）。"""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


def hidden_state_current_probe(
    agent, episodes, test_ratio: float = 0.3, ridge_alpha: float = 1e-2,
    seed: int = 0, return_predictions: bool = False,
) -> dict:
    """P2：RSSM 确定性状态 h_t → 洋流 [u_c, v_c] 的线性探针（SPEC_M4 §2）。

    1) 重放 episode：obs_step 序列取 deter h_t（不用 stoch，deter 是信息载体）；
       preprocess 与 P1 一致（wm.preprocess 处理 is_first/reward 等键，
       observe 在 is_first 边界自动重置 latent，episode 开头无泄漏）。
    2) 按 episode 划分 train/test（防泄漏；仅 1 条 episode 时退化为按时间
       前后切分并打印警告——仅供冒烟验证管线）。
    3) Ridge 回归（闭式解），报告 {r2_u, r2_v, r2_mean}（test 集）。
    4) 对照基线：从瞬时 state obs（7 维）做同样回归（应 ≈ 0——单帧不含洋流信息）。

    episodes 需含 "current_gt" 键（M4 dv3 适配器暗通道落盘，SPEC_M4 §1.4）。
    返回 {"r2_u":..,"r2_v":..,"r2_mean":..,"baseline_r2_mean":..,
          "n_train":..,"n_test":..}（n_* 为时间步数；附
          n_train_episodes/n_test_episodes）。判读标准见模块 docstring。
    """
    import torch

    episodes = [ep for ep in episodes if "current_gt" in ep]
    if not episodes:
        raise ValueError(
            "episodes 中没有 current_gt 键——需要 M4+ 适配器训练的 run "
            "（obs 暗通道落盘，见 SPEC_M4 §1.4）"
        )

    wm = agent._wm
    hs, ys, ss = [], [], []  # 每条 episode 的 (T, deter) / (T, 2) / (T, 7)
    with torch.no_grad():
        for ep in episodes:
            data = {k: np.asarray(v)[None] for k, v in ep.items()}  # (1, T, ...)
            data = wm.preprocess(data)
            embed = wm.encoder(data)
            post, _ = wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )
            hs.append(post["deter"][0].cpu().numpy())
            ys.append(data["current_gt"][0].cpu().numpy())
            ss.append(data["state"][0].cpu().numpy())

    n_ep = len(episodes)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_ep)
    if n_ep >= 2:
        n_test_ep = min(max(1, int(round(n_ep * test_ratio))), n_ep - 1)
        test_idx, train_idx = perm[:n_test_ep], perm[n_test_ep:]
        split_by = "episode"
    else:  # 冒烟兜底：单条 episode 按时间前后切分（有泄漏，仅验证管线）
        test_idx = train_idx = np.array([0])
        split_by = "time"
        print("[P2] 警告：只有 1 条 episode，退化为按时间切分（仅供冒烟）")

    def _concat(idxs, arrays):
        return np.concatenate([arrays[i] for i in idxs], axis=0)

    H_train, H_test = _concat(train_idx, hs), _concat(test_idx, hs)
    Y_train, Y_test = _concat(train_idx, ys), _concat(test_idx, ys)
    S_train, S_test = _concat(train_idx, ss), _concat(test_idx, ss)
    if split_by == "time":
        cut = int(round(len(H_train) * (1.0 - test_ratio)))
        cut = min(max(cut, 1), len(H_train) - 1)
        H_train, H_test = H_train[:cut], H_train[cut:]
        Y_train, Y_test = Y_train[:cut], Y_train[cut:]
        S_train, S_test = S_train[:cut], S_train[cut:]

    # h_t → current 探针
    pred_test = _ridge_fit_predict(H_train, Y_train, H_test, ridge_alpha)
    r2 = _r2_per_dim(Y_test, pred_test)
    # 瞬时 state 基线（应 ≈ 0）
    base_pred_test = _ridge_fit_predict(S_train, Y_train, S_test, ridge_alpha)
    base_r2 = _r2_per_dim(Y_test, base_pred_test)

    result = {
        "r2_u": float(r2[0]),
        "r2_v": float(r2[1]),
        "r2_mean": float(np.nanmean(r2)),
        "baseline_r2_u": float(base_r2[0]),
        "baseline_r2_v": float(base_r2[1]),
        "baseline_r2_mean": float(np.nanmean(base_r2)),
        "n_train": int(len(H_train)),
        "n_test": int(len(H_test)),
        "n_train_episodes": int(len(train_idx)),
        "n_test_episodes": int(len(test_idx)),
        "split_by": split_by,
    }
    if return_predictions:  # 供 run_p2 画图（train 预测也返回，分色用）
        result["predictions"] = {
            "y_train": Y_train,
            "pred_train": _ridge_fit_predict(H_train, Y_train, H_train, ridge_alpha),
            "y_test": Y_test,
            "pred_test": pred_test,
        }
    return result


def _plot_p2(result: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = result["predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, j, name, r2_key in (
        (axes[0], 0, "u_c", "r2_u"),
        (axes[1], 1, "v_c", "r2_v"),
    ):
        ax.scatter(preds["y_train"][:, j], preds["pred_train"][:, j],
                   s=6, alpha=0.35, color="tab:blue", label="train")
        ax.scatter(preds["y_test"][:, j], preds["pred_test"][:, j],
                   s=10, alpha=0.7, color="tab:red", label="test")
        lo = min(preds["y_train"][:, j].min(), preds["y_test"][:, j].min())
        hi = max(preds["y_train"][:, j].max(), preds["y_test"][:, j].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
        ax.set_xlabel(f"true {name} (m/s)")
        ax.set_ylabel(f"predicted {name} (m/s)")
        ax.set_title(f"{name}: test R^2 = {result[r2_key]:.3f}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"P2: hidden-state current probe (R^2_mean={result['r2_mean']:.3f}, "
        f"state baseline={result['baseline_r2_mean']:.3f})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_p2(run_dir: str | Path, test_ratio: float = 0.3) -> dict:
    """对 run_dir 跑 P2 探针：返回指标 dict 并存 p2_probe.png。"""
    ensure_upstream_compat()
    add_vendored_path()
    import tools as dv3_tools

    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    exp_cfg = {
        "dv3_overrides": run_cfg["dv3_overrides"],
        "vehicle": run_cfg.get("vehicle"),
    }
    logdir = run_dir / "dv3_logdir"

    agent = load_dreamer(logdir, exp_cfg)
    episodes = list(dv3_tools.load_episodes(logdir / "eval_eps").values())
    n_eval = len(episodes)
    if len([ep for ep in episodes if "current_gt" in ep]) < 2:
        # 冒烟 run 的 eval episode 太少：并入 train_eps（仅为保证按 episode
        # 划分可执行；正式 run 的 eval_eps 足够，不触发此分支）
        episodes += list(dv3_tools.load_episodes(logdir / "train_eps").values())
        print(f"[P2] eval_eps 仅 {n_eval} 条，并入 train_eps（共 {len(episodes)} 条）")
    result = hidden_state_current_probe(
        agent, episodes, test_ratio=test_ratio, return_predictions=True
    )
    out_path = run_dir / "p2_probe.png"
    _plot_p2(result, out_path)
    printable = {k: v for k, v in result.items() if k != "predictions"}
    print(f"P2 hidden-state current probe: {printable}")
    print(f"P2 figure -> {out_path}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="runs/<exp_id>/seed<N>（含 dv3_logdir/ 与 config.yaml）")
    parser.add_argument("--horizons", default="1,5,10,15",
                        help="逗号分隔的想象步数（默认 1,5,10,15）")
    parser.add_argument("--p2", action="store_true",
                        help="跑 P2 隐状态洋流探针（默认跑 P1）")
    parser.add_argument("--test-ratio", type=float, default=0.3,
                        help="P2 test 集 episode 比例（默认 0.3）")
    args = parser.parse_args(argv)
    if args.p2:
        run_p2(args.run_dir, test_ratio=args.test_ratio)
    else:
        horizons = tuple(int(x) for x in args.horizons.split(","))
        run_p1(args.run_dir, horizons)


if __name__ == "__main__":
    main()
