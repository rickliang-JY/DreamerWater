# UWM — SPEC M2：DreamerV3 接入（世界模型）+ P1/P4 探针

单一事实来源（M2 阶段）。遵循 SPEC.md / SPEC_M1.md；冲突时以本文件为准。

## 0. 范围与算力前提

- 算力：3 核 CPU 沙箱，无 GPU。**小网络 + 50k 环境步验证收敛趋势**，全量后续再补
- 世界模型载体：`third_party/dreamerv3-torch`（vendored，commit 6ef8646，MIT）
- state 观测（dmc_proprio 路线，无 CNN）、station_keeping、与 M1 同任务同判据
- 探针：P1（imagination 开环误差）+ P4（imag_horizon 可配置，为扫描做准备）
- **不做**：pixel 观测（M7）、TD-MPC2、时变洋流（M4）、完整 P4 网格扫描（算力不够，先把单点跑通）

依赖变更：新增 `ruamel.yaml`、`einops`（dreamerv3-torch 必需）。mujoco/dm_control/crafter 等不装（vendored 代码懒加载，不 import 即可）。其余上限不变。

## 1. 架构：薄封装

不 fork 重写训练循环。三层：
1. **vendored 上游**（`third_party/dreamerv3-torch/`）：只新增一个文件 `envs/uwm.py`（环境适配器）+ 在 `dreamer.py` 的 `make_env` 里注册一个 task 前缀分支（≤10 行 patch，注释标记 `# UWM-PATCH`）。其余上游代码一个字节不改。
2. **驱动**（`scripts/train_dreamer.py`）：把我们的 exp yaml 翻译成上游 config 对象，调上游 main 流程。
3. **转接**（`scripts/convert_dv3_episodes.py` + `uwm/eval/wm_probes.py`）：上游 logdir 的 episode npz → 我们的 recorder schema（面板 B 直接回放）；P1 探针。

## 2. 接口合同

### 2.1 环境适配器 `third_party/dreamerv3-torch/envs/uwm.py`

模仿 `envs/dmc.py` 的 DeepMindControl 接口（旧 gym 4 元组 step）：

```python
class UWMStationKeeping:
    """把 uwm.envs.station_keeping.StationKeepingEnv 包成上游期望的接口。"""
    metadata = {}
    reward_range = [-np.inf, np.inf]
    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        # name 形如 "uwm_station_keeping"；内部 vehicle_cfg 从
        # configs/vehicle/bluerov2.yaml 读（相对 vendored 根的上溯两级 repo 根）
    @property
    def observation_space(self):  # gym.spaces.Dict({"state": Box(7)})
        # 只提供 "state" 键，不提供 "image"
    @property
    def action_space(self):       # gym.spaces.Box(-1, 1, (3,), float32)
    def step(self, action):       # (obs_dict, reward, done, info)；done = terminated|truncated
        # obs_dict: {"state": float32 vec, "is_terminal": False, "is_first": bool}
    def reset(self):
```

`make_env` 注册：`task.startswith("uwm_")` → `UWMStationKeeping(...)`，外包上游 TimeLimit(500) + NormalizeActions（参照 dmc 分支写法）。

### 2.2 `configs/exp/m2_dreamerv3_station_keeping.yaml`

基于上游 `dmc_proprio` preset 缩小（CPU 预算）：

```yaml
exp_id: m2_dreamerv3_station_keeping
seeds: [0]
dv3_overrides:
  task: uwm_station_keeping
  logdir: runs/m2_dreamerv3_station_keeping/seed0/dv3_logdir
  steps: 50000
  eval_every: 5000
  log_every: 5000
  eval_episode_num: 10
  prefill: 1000
  envs: 1
  action_repeat: 1          # 环境内部已 action_repeat=5（0.1s/env step），不再叠加
  time_limit: 500
  device: cpu
  compile: false
  precision: 32
  train_ratio: 128          # 上游默认 512，CPU 降档
  batch_size: 8
  batch_length: 32
  # 小网络
  dyn_hidden: 128
  dyn_deter: 128
  dyn_stoch: 16
  dyn_discrete: 16
  units: 128
  imag_horizon: 15          # P4：做成可配，扫描时改这里
  encoder: {mlp_keys: 'state', cnn_keys: '$^', mlp_layers: 2, mlp_units: 256}
  decoder: {mlp_keys: 'state', cnn_keys: '$^', mlp_layers: 2, mlp_units: 256}
  video_pred_log: false
  reward_EMA: true
seed: 0
```

### 2.3 `scripts/train_dreamer.py`

- `python scripts/train_dreamer.py <exp.yaml> [--seed N]`
- 读 exp yaml → 构造上游 config（加载 vendored configs.yaml defaults + dmc_proprio preset + dv3_overrides）→ `sys.path.insert(0, "third_party/dreamerv3-torch")` → 调上游 dreamer.main
- 上游 logdir 自带 checkpoint.pt 断点续训（确认其行为并文档化）；记录 git commit hash
- 训练结束自动跑 convert（§2.4）与 P1 探针（§2.5）

### 2.4 `scripts/convert_dv3_episodes.py`

- 上游 evaldir/traindir 的 episode npz（其 schema：state/action/reward/...）→ 我们的 `episodes/ep_XXXX.npz`（t, eta, nu, action, reward）
- state 前 6 维即 [x,y,ψ,u,v,r]（对应 eta/nu），action 直接映射；t 由 dt×action_repeat 生成
- 输出到 `runs/m2_.../seed0/episodes/`（recorder schema），面板 B 可回放
- 同步生成 metrics.csv（与 M1 同表头，便于对比）

### 2.5 `uwm/eval/wm_probes.py` — P1 探针

```python
def open_loop_imagination_error(agent, episodes, horizons=(1, 5, 10, 15)) -> dict:
    """从真实 episode 取上下文，RSSM 开环想象 H 步，与真实后续 state 对比。
    返回 {horizon: mean_l2_error}。输入 agent 为已加载 ckpt 的上游 Dreamer。"""
def load_dreamer(logdir, exp_cfg) -> agent: ...
```

- `python -m uwm.eval.wm_probes <run_dir>` 输出 P1 结果 + 存 `p1_imagination_error.png`
- P4 说明：imag_horizon 已在配置层可配（§2.2），完整扫描等算力；本阶段 README 写明物理时间账：H=15 × 0.1s = 1.5s 想象时长 vs 载体时间常数 τ≈2.9s（M_11/D_11 = 17/4.03），指出 H=30（3.0s）是下一轮扫描的首要候选

### 2.6 `tests/test_dreamer.py`

1. 适配器接口：obs dict 键/形状/dtype、action 界、is_first/is_terminal、4 元组 step
2. 冒烟训练：tiny 配置（steps=600、prefill=100、dyn_hidden/deter=32、batch_size=4、batch_length=8、train_ratio=32、log/eval_every=500）10 分钟内跑完，ckpt 生成
3. convert：冒烟 logdir → recorder schema，load_episode 读回，字段齐全形状正确
4. P1：冒烟 ckpt 上 open_loop_imagination_error 跑通，输出 dict 值 finite
5. 既有 22 项不得破坏

## 3. 工程约定

- vendored 上游除 §2.1 两处 patch 外零改动；patch 处加 `# UWM-PATCH` 注释
- runs/ 依旧 gitignore（runs/demo 白名单不变）；冒烟产物不入库
- 负结果照实记录：50k 步未收敛也出曲线与 P1 图
- 提交：`M2: <内容>`

## 4. 完成判据（Gate）

1. pytest 全绿（26 项）
2. 冒烟训练 + convert + P1 全链路实测通过
3. 正式训练（50k 步 seed0）由 orchestrator 启动（coder 只负责到冒烟）

## 勘误（2026-09-07，orchestrator）
沙箱后台进程无法跨工具调用存活，训练改为同步分段（`timeout` + ckpt 续训）。
正式配置相应调整：steps 50k→20k（目标仍是"验证收敛趋势"，SAC 在 40k 已收敛）、
train_ratio 128→64（梯度成本减半，速度约翻倍）、eval_episode_num 10→5、
eval_every 5000→4000、log_every→500（ckpt 粒度，降低分段损耗）。
样本效率对比时以 env_steps 为横轴，与 SAC 曲线在 0–20k 区间直接可比。

## 勘误 2（2026-09-07）
- 上游 ckpt 保存点为 simulate() 边界（=eval_every 周期），分段训练够不到 →
  驱动层新增 apply_chunked_ckpt_patch：simulate 拆 500 步段、段间存 latest.pt（含优化器）。
- 实测一次训练发散（RSSM stoch logits NaN，Categorical constraint 崩溃）→
  补丁加 ckpt 轮转（latest_prev.pt），崩溃可回滚一档。vendored 上游仍零改动。

## 勘误 3（2026-09-07）
两次独立运行均发散出 NaN（约 2k / 7.5k 步，RSSM stoch logits）。处置：
① config 级稳定化 grad_clip 1000→100、model_lr 1e-4→5e-5（如实记录，对比有效性不受影响——SAC 同样条件）；
② ckpt 补丁加 NaN 防护：参数非有限则跳过保存，latest_prev.pt 永远干净。
从头重训（早期数据废弃，保证最终曲线可复现）。
