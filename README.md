# UWM — 水下航行器世界模型（M0 最小切片）

**☁️ 免安装在线审查（molab，点击即开）：**

- [🚀 审查入口（自动克隆装环境 + 实跑 pytest + 内嵌双交互面板）](https://molab.marimo.io/github/rickliang-JY/DreamerWater/blob/main/notebooks/molab_quickstart.py)

> molab 只镜像单个 notebook 文件，故审查入口内置了自举逻辑（从本仓库克隆安装）。
> 面板 D/B（`notebooks/dynamics_playground.py`、`episode_replay.py`）需在本地仓库内运行。

3-DOF Fossen 水平面动力学（torch.float64, RK4）+ gymnasium 定点悬停环境 +
单元测试 + marimo 交互面板 D。单一事实来源见 `SPEC.md`。

## 安装与测试

```bash
pip install -e .[dev]
pytest tests/ -v
```

## 交互面板

```bash
marimo run notebooks/dynamics_playground.py
```

## 轨迹回放（面板 B）

```bash
python scripts/record_demo_episodes.py     # 录制 10 条演示 episode 到 runs/demo/
marimo run notebooks/episode_replay.py     # 回放：run 下拉框 + episode/时间步滑块
```

`uwm/eval/recorder.py` 提供 `EpisodeRecorder`（提案 §7.1 契约：`config.yaml` +
追加写 `metrics.csv` + `episodes/ep_XXXX.npz`，含 t/eta/nu/action/reward）与
`load_episode(path)` 读取函数。回放面板每 5s 轮询 run 目录，录制/训练进行中
也能看到新 episode。演示数据中 ep 0–4 为随机策略（发散）、ep 5–9 为 PD 回正
（收敛到原点）。

## 目录

- `configs/vehicle/` — BlueROV2（主力载体）与 REMUS100（回归对照）水动力参数
- `uwm/dynamics/` — `Fossen3DOF`、`ThrusterClip`
- `uwm/envs/` — `UWEnvBase`、`StationKeepingEnv`
- `tests/` — 解析解验证、REMUS100 回归（vendored fixture）、环境测试
- `notebooks/dynamics_playground.py` — marimo 面板 D

## M0 验证结果（2026-09-07，干净 venv 独立复验）

- `pytest tests/`：**13/13 通过**（解析解验证 6 项、REMUS100 回归 2 项、环境 5 项）
- 指数衰减时间常数与解析值 m/d 相对误差 < 1%（三通道）
- 定常推力终速 u∞=F/X_u 相对误差 < 1%
- REMUS100 回归：M、D 矩阵逐项 < 1%；积分轨迹 < 5%（fixture vendored，测试无网络依赖）
- 随机策略 1000 步有界不发散；gymnasium `check_env` 通过
- `marimo run notebooks/dynamics_playground.py` 冒烟通过（面板 D：阻尼/附加质量/洋流/推力限幅滑块 + 阶跃响应 + 2D 轨迹）

## 参数来源

- BlueROV2：文献值（von Benzon et al., NTNU; Scaradozzi 2024），见 `configs/vehicle/bluerov2.yaml` 注释
- REMUS100：直接提取自 `cybergalactic/PythonVehicleSimulator` 的 `remus100.py` 水平面 3×3 子块

## M1：SAC 基线

单文件 SAC（`uwm/algos/sac.py`，cleanrl 风格、纯 torch）：tanh-squashed 高斯
策略（log_prob 带 tanh 修正项）、双 Q 取 min、target 软更新（tau）、自适应
alpha（target_entropy = −act_dim）。done 用 terminated；TimeLimit 截断按未
终止处理（照常 bootstrap）。

```bash
# 训练（可分段：每 2000 步自动存 ckpt/latest.pt，--resume 续训）
python scripts/train.py configs/exp/m1_sac_station_keeping.yaml --seed 0
python scripts/train.py configs/exp/m1_sac_station_keeping.yaml --seed 0 \
    --total-steps 120000 --resume runs/m1_sac_station_keeping/seed0/ckpt/latest.pt

# 确定性评估（episodes 存 <run_dir>/eval/episodes/ 供面板 B 回放）
python scripts/eval_policy.py runs/m1_sac_station_keeping/seed0 --episodes 20
```

**M1 通过判据**（eval_policy.py 输出 PASS/FAIL）：

1. final_dist 中位数 < 0.3 m
2. 最后 5 秒位置 std 均值 < 0.05 m（无持续振荡）

训练产物：`runs/<exp_id>/seed<N>/`（config.yaml 含 git commit、metrics.csv、
ckpt/、episodes/）+ `runs/<exp_id>/learning_curve_seed<N>.png`
（eval 点 + 训练 return 滑动平均，窗口 20）。

## M1 验证结果（2026-09-07，SAC 基线，3 seed × 120k 步）

| seed | return (eval 均值) | final_dist 中位数 | 最后 5s 位置 std | 判据 |
|---|---|---|---|---|
| 0 | −33.2 | **0.008 m** | 0.0000 m | ✅ PASS |
| 1 | −32.8 | **0.010 m** | 0.0066 m | ✅ PASS |
| 2 | −52.2 | **0.050 m** | 0.0187 m | ✅ PASS |

> **v0.1 动力学修正后重测（二次阻尼，SPEC_M2 勘误 4）**：3 seed 40k 步即通过判据——
> final_dist 中位数 0.018 / 0.006 / 0.007 m，振荡 std ≈ 0。修正后任务更物理
> （满推终端速度 surge 1.9 m/s、yaw 3.8 rad/s）。M2 对比以此为基准。
> 注：eval 各 episode 的 final_dist 在同一 seed 内恒定（策略把载体停在固定小偏移处，
> return 仍有 episode 间差异），记录为已知现象，疑似稳态偏差与位置/控制奖励权衡有关。

判据（SPEC_M1 §2.4）：final_dist 中位数 < 0.3 m 且无持续振荡（std < 0.05 m）。
学习曲线见 `docs/figures/m1_learning_curve_seed{0,1,2}.png`；
eval 轨迹可用面板 B 回放（`runs/m1_sac_station_keeping/seed*/eval/episodes/`）。

**结论**：环境与奖励设计正确，SAC 基线确立。M2（DreamerV3）的对照基准：
同等性能 = final_dist 中位数 ~0.01–0.05 m、return ~−33~−52 @ 120k 步。

## M2：DreamerV3 薄封装 + P1 探针

- 环境适配器 `third_party/dreamerv3-torch/envs/uwm.py`（gymnasium 5 元组 →
  旧 gym 4 元组），`dreamer.py` 的 `make_env` 注册 `uwm_*` 前缀（`# UWM-PATCH`）；
  驱动 `scripts/train_dreamer.py`，转接 `scripts/convert_dv3_episodes.py` +
  `uwm/eval/wm_probes.py`（P1 开环想象误差）
- 配置：`configs/exp/m2_dreamerv3_station_keeping.yaml`（正式 50k 步）与
  `_smoke.yaml`（CPU 冒烟，~3.5 min）
- 用法：
  `python scripts/train_dreamer.py configs/exp/m2_dreamerv3_station_keeping_smoke.yaml`
  （训练结束自动跑 convert + P1；重复执行即续训，上游 latest.pt 断点恢复）

**P4 物理时间账**（SPEC_M2 §2.5）：imag_horizon 已在配置层可配
（`dv3_overrides.imag_horizon`）。当前 H=15 × 0.1s/env step = **1.5s** 想象
时长，而载体纵向时间常数 τ = M_11/D_11 = (11.5+5.5)/4.03 ≈ **4.2s**
（SPEC_M2 文中按 17/4.03 的写法为 τ≈2.9s 的口径*），H=15 不足以覆盖一次
完整的速度响应。**H=30（3.0s）是下一轮扫描的首要候选**，待算力到位后做
H ∈ {15, 30, 45} 网格扫描（正确性检查，不是调参）。

\* 若按裸质量 m/X_u = 11.5/4.03 ≈ 2.9s（SPEC_M2 口径）；含附加质量
m+|X_udot| 则为 17/4.03 ≈ 4.2s。两种口径下 H=15（1.5s）都偏短，结论一致。

## M2 结果（2026-09-07，DreamerV3，20k 步 seed0，CPU 降档配置）

**一句话：世界模型学会了水下动力学，但 20k 步内策略尚未学会任务——与提案 §8.3 的预期一致。**

| 维度 | 结果 | 判读 |
|---|---|---|
| world model 损失 | reward_loss 5.06→0.9，state_loss 9.6→0.9 | 健康下降 |
| P1 imagination 开环误差 | H=1: 1.45 / H=5: 1.94 / H=10: 2.43 / H=15: 3.02 | 随步数平缓增长（冒烟期 H=1 高达 42），模型确实学到了可预测的动力学 |
| eval_return | −1315 → −720 后平台（SAC 同期 −40 量级） | 策略未收敛 |
| 判据对比 | 未达到 M1 的 final_dist<0.3m | 预期内（见下） |

**未收敛的三重原因（如实记录）**：① 算力降档——train_ratio 64 使每步梯度更新仅为
标准配置的 1/2，20k 环境步 ≈ 标准 dv3 的 ~2k 等效步（dmc_proprio 通常 100k+ 起步）；
② H=15 × 0.1s = 1.5s 想象时长 < 载体时间常数（提案 §8.1 预警的头号坑）；
③ 低维状态 + 密集奖励本就是 SAC 的主场。

**下一步优先级**：P4 horizon 扫描（H=30 首选）> 全量步数重训 > 多 seed。
对比图：`docs/figures/m2_sac_vs_dreamerv3.png`；P1：`docs/figures/m2_p1_imagination_error.png`；
eval 轨迹可用面板 B 回放 `runs/m2_dreamerv3_station_keeping/seed0/episodes/`。

## P4 初扫结果（2026-09-07，H=15 vs H=30，seed0，对齐步数对比）

**结论：horizon 不是当前瓶颈——算力（梯度更新量）才是。**

- eval_return：H=15 与 H=30 在 0–10k 对齐区间均停在 −720~−780 平台，**无分离**——
  把想象时长从 1.5s 加到 3.0s 并未解锁策略学习。§8.1 的"horizon 头号坑"假设
  在此算力档位下**未被证实为主要约束**（有价值的负结果）。
- P1：H=30 模型 imagination 误差 4.58@15步，高于 H=15 模型的 3.02——但注意
  **混淆变量**：H=15 模型训了 20k 步、H=30 只有 10k 步，误差差异主要反映训练量
  而非 horizon 本身。严格的多 seed 等步数网格（H∈{15,30,45}×3 seed）留待算力到位。
- 图：`docs/figures/m2_p4_horizon_scan.png`

**对路线图的修正**：优先级从"P4 扫描"调整为"**先解决等效更新量**"
（全量步数 + 标准 train_ratio，或 GPU），horizon 网格在其后。
