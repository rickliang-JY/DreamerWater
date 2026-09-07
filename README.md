# UWM — 水下航行器世界模型（M0 最小切片）

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

## 目录

- `configs/vehicle/` — BlueROV2（主力载体）与 REMUS100（回归对照）水动力参数
- `uwm/dynamics/` — `Fossen3DOF`、`ThrusterClip`
- `uwm/envs/` — `UWEnvBase`、`StationKeepingEnv`
- `tests/` — 解析解验证、REMUS100 回归（vendored fixture）、环境测试
- `notebooks/dynamics_playground.py` — marimo 面板 D
