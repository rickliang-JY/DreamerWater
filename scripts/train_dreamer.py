"""DreamerV3 训练驱动（SPEC_M2 §2.3）。

用法：
    python scripts/train_dreamer.py <exp.yaml> [--seed N]

- 读 exp yaml → dv3_overrides 叠加到 vendored configs.yaml 的 defaults +
  dmc_proprio preset → 调上游 dreamer.main（薄封装，不 fork 训练循环）
- 断点续训：上游 logdir 存在 latest.pt 时 dreamer.main 自动加载网络与优化器
  状态，且 train_eps 已有数据计入 count_steps、跳过对应 prefill（上游内建
  行为，无需本层干预）；重复执行本脚本即续训
- 每个 (exp_id, seed) 一个 run 目录 runs/<exp_id>/seed<N>/，内含
  config.yaml（本层写）与 dv3_logdir/（上游产物：latest.pt、metrics.jsonl、
  train_eps/、eval_eps/）
- 训练结束自动跑 convert（SPEC_M2 §2.4）与 P1 探针（§2.5）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uwm.compat import ensure_upstream_compat  # noqa: E402
from uwm.eval.dv3 import (  # noqa: E402
    add_vendored_path,
    apply_chunked_ckpt_patch,
    apply_cpu_fallback,
    build_dv3_config,
)


def git_commit() -> str:
    """取当前 git commit hash（取不到则返回 'unknown'）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — 无 git 环境不阻塞训练
        return "unknown"


def _resolve_logdir(logdir: str, seed: int) -> Path:
    """logdir 相对仓库根解析；--seed 非 0 时把路径中的 seed0 替换为 seed<N>。"""
    if seed != 0:
        logdir = logdir.replace("seed0", f"seed{seed}")
    path = Path(logdir)
    return path if path.is_absolute() else REPO_ROOT / path


def train(exp_path: str | Path, seed: int | None = None) -> Path:
    """按 exp yaml 跑上游 DreamerV3 训练，返回 run_dir。"""
    exp_path = Path(exp_path)
    if not exp_path.is_absolute():
        exp_path = REPO_ROOT / exp_path
    with open(exp_path, encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    exp_id = exp_cfg["exp_id"]
    seed = int(seed if seed is not None else exp_cfg["seeds"][0])
    overrides = dict(exp_cfg["dv3_overrides"])
    overrides["seed"] = seed
    logdir = _resolve_logdir(str(overrides["logdir"]), seed)
    overrides["logdir"] = str(logdir)
    run_dir = logdir.parent

    # run 级 config.yaml（wm_probes / convert 据此重建配置）
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    stored_overrides = dict(overrides)
    stored_overrides["logdir"] = str(logdir.relative_to(run_dir))
    if not config_path.exists():  # 续训不覆盖首次记录（git_commit 为首次提交）
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "exp_id": exp_id,
                    "seed": seed,
                    "git_commit": git_commit(),
                    "exp_config": (
                        str(exp_path.relative_to(REPO_ROOT))
                        if exp_path.is_relative_to(REPO_ROOT)
                        else str(exp_path)  # 仓库外的临时配置（如测试 tmp_path）
                    ),
                    "vehicle": exp_cfg.get("vehicle", "configs/vehicle/bluerov2.yaml"),
                    "task": exp_cfg.get("task") or {},
                    "dv3_overrides": stored_overrides,
                },
                f, allow_unicode=True, sort_keys=False,
            )

    ensure_upstream_compat()
    add_vendored_path()
    apply_cpu_fallback()
    apply_chunked_ckpt_patch(ckpt_every=500)  # 分段沙箱友好：每 500 步存 ckpt
    from uwm.eval.dv3 import apply_nan_canary

    apply_nan_canary(logdir / "nan_debug.npz")  # NaN 金丝雀：崩溃前 dump 现场
    from uwm.eval.dv3 import apply_encoder_canary

    apply_encoder_canary(logdir / "nan_debug_enc.npz")  # encoder/obs 金丝雀
    import dreamer as dv3_dreamer

    config = build_dv3_config(overrides)
    dv3_dreamer.main(config)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="exp 配置 yaml 路径")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（默认取 seeds[0]）")
    parser.add_argument("--no-post", action="store_true",
                        help="跳过训练后的 convert 与 P1 探针")
    args = parser.parse_args(argv)

    run_dir = train(args.config, args.seed)

    if not args.no_post:
        from uwm.eval.dv3_episodes import convert_logdir
        from uwm.eval.wm_probes import run_p1

        summary = convert_logdir(run_dir / "dv3_logdir", run_dir)
        print(f"[convert] {summary['n_eval_episodes']} eval episodes "
              f"-> {summary['episodes_dir']}")
        run_p1(run_dir)


if __name__ == "__main__":
    sys.exit(main())
