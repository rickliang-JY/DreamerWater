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

判据（SPEC_M1 §2.4）：final_dist 中位数 < 0.3 m 且无持续振荡（std < 0.05 m）。
学习曲线见 `docs/figures/m1_learning_curve_seed{0,1,2}.png`；
eval 轨迹可用面板 B 回放（`runs/m1_sac_station_keeping/seed*/eval/episodes/`）。

**结论**：环境与奖励设计正确，SAC 基线确立。M2（DreamerV3）的对照基准：
同等性能 = final_dist 中位数 ~0.01–0.05 m、return ~−33~−52 @ 120k 步。

## 下一步（M2）

接入 DreamerV3（薄封装 NM512/dreamerv3-torch）+ state 观测 + 同任务，
同步执行 P4 horizon 扫描（提案 §8.1：水下 dt=0.1s/env step，H=15 仅覆盖 1.5s
物理时间，需扫描 H×action_repeat 网格，这是正确性检查不是调参）。
