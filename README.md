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

## 下一步（M1）

接入 SAC（Stable-Baselines3 或 cleanrl 单文件风格）+ state 观测 + station_keeping，
通过判据：稳态位置误差收敛到阈值内、轨迹无持续振荡。见提案 v0.2 §5。
