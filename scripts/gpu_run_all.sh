#!/bin/bash
# GPU 全量复现实验一键脚本（SPEC_M4 §3.4，用户在 GPU 机器上执行）。
#
# 顺序与理由：
#   1) SAC × 3 seed（常值 + OU）：model-free 基线，单 run 最快（CPU 即可，
#      ~2-4h/120k 步），先跑可先拿到对照锚点；
#   2) DreamerV3 常值 × 3 seed（M2 全量复刻）：回答"标准算力下 dv3 能否
#      达到 SAC 水平"，是 OU 实验的对照组；
#   3) DreamerV3 OU × 3 seed（M4 主实验）：每次训练后自动 convert + P1，
#      再补 P2 隐状态洋流探针（不需要策略收敛即可测量）。
#   Dreamer 单 run 预计 1–2 天/200k 步（RTX 级别单卡，显存 8–12GB）；
#   全脚本总计约 6–12 天（Dreamer 部分可多张卡手工并行，改 device 即可）。
#
# 断点续训：所有命令重复执行即续训（dv3 上游 latest.pt 自动恢复；
# SAC 需 --resume，见 README）。脚本幂等，中断后重跑会跳过已完成的步数。
#
# 可选追加（P4 horizon 网格）：改 m4_dreamerv3_ou_gpu.yaml 的 imag_horizon
# 为 45（或把 m2_dreamerv3_gpu.yaml 改为 30/45）重跑 §2/§3 即可。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================================"
echo " UWM GPU 全量实验：SAC(常值+OU) → dv3 常值(M2) → dv3 OU(M4)"
echo " 预计总时长 6–12 天（单卡），可随时中断重跑（幂等续训）"
echo "============================================================"

for seed in 0 1 2; do
  echo "=== [1/6] SAC 常值流 seed${seed}（120k 步，预计 ~2-4h，CPU 可跑） ==="
  python scripts/train.py configs/exp/m1_sac_station_keeping.yaml --seed "${seed}"
done

for seed in 0 1 2; do
  echo "=== [2/6] SAC OU 时变流 seed${seed}（120k 步，预计 ~2-4h，CPU 可跑） ==="
  python scripts/train.py configs/exp/m1_sac_station_keeping_ou.yaml --seed "${seed}"
done

for seed in 0 1 2; do
  echo "=== [3/6] DreamerV3 常值流 M2 全量 seed${seed}（200k 步，预计 1–2 天，显存 8–12GB） ==="
  # 训练结束自动 convert + P1（train_dreamer.py 内建 post 步骤）
  python scripts/train_dreamer.py configs/exp/m2_dreamerv3_gpu.yaml --seed "${seed}"
done

for seed in 0 1 2; do
  echo "=== [4/6] DreamerV3 OU 时变流 M4 seed${seed}（200k 步，预计 1–2 天，显存 8–12GB） ==="
  python scripts/train_dreamer.py configs/exp/m4_dreamerv3_ou_gpu.yaml --seed "${seed}"
  echo "=== [5/6] P2 隐状态洋流探针 seed${seed}（几分钟，CPU 即可） ==="
  python -m uwm.eval.wm_probes "runs/m4_dreamerv3_ou/seed${seed}" --p2
done

echo "=== [6/6] 全部完成。产出清单见 README「GPU 全量复现实验指南」 ==="
