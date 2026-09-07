# UWM — SPEC v0 (M0 最小切片)

单一事实来源。实现必须严格遵守接口合同，不得擅自更改模块边界与函数签名。

## 0. 范围

只做提案 §12 的 M0：3-DOF Fossen 动力学（torch）+ gymnasium 环境 + 单元测试 + marimo 面板 D。
**不做**：SAC/Dreamer 接入、二次阻尼、推力分配矩阵、推力滞后、时变洋流、像素观测、6-DOF。

依赖上限：`torch`、`gymnasium`、`marimo`、`pyyaml`、`numpy`、`matplotlib`、`pytest`。禁止引入其他依赖。

## 1. 目录结构

```
uwm/
├── README.md
├── SPEC.md                       # 本文件副本
├── pyproject.toml                # 包名 uwm, pip install -e .
├── configs/
│   └── vehicle/
│       ├── bluerov2.yaml         # 主力载体：开架悬停式，近全驱动
│       └── remus100.yaml         # 回归对照载体（与 PythonVehicleSimulator 对齐用）
├── uwm/
│   ├── __init__.py
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── fossen.py             # 3-DOF Fossen 刚体模型，torch
│   │   └── thrusters.py          # v0 仅 clip 饱和
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── base.py               # UWEnvBase(gymnasium.Env)
│   │   └── station_keeping.py    # StationKeepingEnv
│   └── eval/                     # M0 留空包，占位
│       └── __init__.py
├── tests/
│   ├── fixtures/
│   │   └── remus100_reference.npz   # 由 PythonVehicleSimulator 生成的回归基准（vendored）
│   ├── test_dynamics.py
│   ├── test_regression_remus100.py
│   └── test_env.py
└── notebooks/
    └── dynamics_playground.py    # marimo 面板 D
```

## 2. 接口合同

### 2.1 `uwm/dynamics/fossen.py`

3-DOF 水平面模型：η = [x, y, ψ]（NED），ν = [u, v, r]。

```python
class Fossen3DOF:
    """torch 实现的 3-DOF Fossen 刚体模型。全部参数为 torch.float64 张量。

    动力学: M ν̇ + C(ν)ν + D(ν)ν = τ + τ_current
    v0: D 只含线性项; 洋流为常值（相对速度进入阻尼项）。
    """
    def __init__(self, cfg: dict): ...
        # cfg 来自 vehicle yaml，字段见 §2.4

    def step(self, eta: torch.Tensor, nu: torch.Tensor,
             tau: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """RK4 积分一个物理步。eta/nu/tau 形状 (..., 3)，支持批量。返回新 (eta, nu)。"""

    def rhs(self, eta, nu, tau) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (η̇, ν̇)。η̇ = J(ψ)ν；ν̇ = M⁻¹(τ − C(ν)ν − D(ν)ν_rel)。"""

    # 属性（torch 张量，供测试逐项对照）:
    #   M: (3,3) 惯性+附加质量; C(nu)->(3,3); D: (3,3) 线性阻尼; current: (2,) 常值洋流速度
```

约定：
- J(ψ) 为标准 3-DOF 旋转矩阵（ψ 仅偏航）。
- 洋流处理：阻尼作用于相对速度 ν_rel = ν − ν_c^body（ν_c^body = J(ψ)⁻¹ [u_c, v_c, 0]）。惯性项按 Fossen 常值无旋流近似：ν̇ 方程中使用 ν_rel 进 C、D（v0 简化，注释说明）。
- 姿态角 wrap 到 (−π, π]。
- `action_repeat` 不在本类；由 env 层负责用物理 dt 多次调用 step。

### 2.2 `uwm/dynamics/thrusters.py`

```python
class ThrusterClip:
    """v0：直接输出 3 维广义力，逐分量 clip 到 ±tau_max。"""
    def __init__(self, tau_max: Sequence[float]): ...
    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        """action ∈ [−1, 1]^3（环境动作空间），返回 τ ∈ R^3。"""
```

### 2.3 `uwm/envs/base.py` 与 `station_keeping.py`

```python
class UWEnvBase(gymnasium.Env):
    """统一基类。持有 Fossen3DOF + ThrusterClip。
    __init__(self, vehicle_cfg: dict, task_cfg: dict)
    物理 dt 与 action_repeat 解耦：env.step = action_repeat 次 fossen.step。
    """
    metadata = {"render_modes": []}

class StationKeepingEnv(UWEnvBase):
    """目标：从随机初始偏移出发，回到原点并保持。
    observation: Box(7): [x, y, ψ, u, v, r, dist_to_goal]  （state 模式，≤10 维）
    action: Box(3) ∈ [−1,1]
    reward: −w_pos·symlog(||p−p_goal||) − w_ctrl·||a||²，w_pos=1.0, w_ctrl=0.01
    terminated: False（v0 无终止）；truncated: t >= max_episode_steps
    reset: 初始位置在半径 init_radius 内均匀采样，ψ 均匀，ν=0
    """
```

symlog(x) = sign(x)·log1p(|x|)。

### 2.4 `configs/vehicle/bluerov2.yaml`

字段（全部 SI 单位）：
```yaml
name: bluerov2
dof: 3
mass: <kg>
added_mass: [X_udot, Y_vdot, N_rdot]   # 对角
inertia_z: <kg·m²>
linear_damping: [X_u, Y_v, N_r]        # 对角，正数
tau_max: [Fx, Fy, Mz]
current: [u_c, v_c]                    # 常值洋流，v0 可为 [0,0]
dt: 0.02
action_repeat: 5
```
BlueROV2 参数来源：gokulp01/bluerov2_gym 仓库的水动力参数；remus100.yaml 的 3-DOF 子集参数来源：cybergalactic/PythonVehicleSimulator 的 remus100.py（M、D 的 3×3 水平面子块）。

## 3. 测试（全部必须实际运行通过）

### 3.1 `tests/test_dynamics.py` — 解析解验证
1. **指数衰减**：ν(0)=[u0,0,0]，τ=0，对角 M、D 下 u(t)=u0·exp(−X_u/M_11·t)。数值轨迹与解析解时间常数相对误差 < 1%。三个通道各测一次。
2. **定常推力收敛**：恒定 τ=[Fx,0,0] → 终速 u∞ = Fx/X_u，相对误差 < 1%。
3. **纯 yaw 转动**：η 的 ψ 通道积分正确（r 恒定 → ψ 线性增长，模 wrap 正确）。
4. **批量一致性**：(B,3) 批量输入与逐样本循环结果完全一致。

### 3.2 `tests/test_regression_remus100.py` — 与 PythonVehicleSimulator 对齐
- 构建期：clone `cybergalactic/PythonVehicleSimulator`，用其 remus100 模块生成回归基准：同一初始状态 + 同一控制序列（阶跃 + 正弦，200 步），numpy 参考轨迹存入 `tests/fixtures/remus100_reference.npz`（vendored 进 repo，测试本身不需要网络）。
- 测试：我们的 torch 3-DOF 模型（remus100.yaml）与参考轨迹逐点对齐，积分轨迹相对误差 < 5%；M、D 矩阵逐项相对误差 < 1%。
- 若 PythonVehicleSimulator 的 REMUS100 矩阵耦合项导致 3-DOF 子集无法直接对齐，允许在测试中以"<5% 轨迹误差"为准并注释说明取舍。

### 3.3 `tests/test_env.py`
1. gymnasium `check_env` 通过（gymnasium.utils.env_checker）。
2. 随机策略 1000 步不发散：|x|,|y| 有界（< 1e3），无 NaN/Inf。
3. reset 可复现：同 seed 两次 reset 观测相同。
4. action_repeat 正确：env.step 一次，内部物理时间前进 dt×action_repeat。

## 4. marimo 面板 D — `notebooks/dynamics_playground.py`

- `marimo` notebook（`marimo edit`/`marimo run` 可打开），不依赖任何训练产物。
- 滑块：线性阻尼三通道缩放（0.1×–10×）、附加质量缩放、洋流强度（0–1 m/s）、推力限幅缩放。
- 交互输出：阶跃推力下 u/v/r 的响应曲线 + 2D 俯视轨迹，标注解析时间常数 τ=m/d 参考线。
- 只读 `configs/vehicle/bluerov2.yaml`，不 import 训练代码。

## 5. 工程约定

- 所有动力学计算 torch.float64（M0 精度优先，M2 后可切 float32）。
- 每个公开函数写 docstring（中文，含物理量单位）。
- 提交信息格式：`M0: <内容>`。
- 不做未在 SPEC 中出现的抽象（不写 registry、不写 configurable factory）。

## 6. 完成判据（Gate）

1. `pytest tests/` 全绿（在干净 venv，`pip install -e .[dev]` 后）。
2. `marimo run notebooks/dynamics_playground.py` 能启动（冒烟验证：import 无错即可，不要求长驻）。
3. 零超出 §0 的依赖。
