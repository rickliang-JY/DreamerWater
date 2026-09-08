# UWM — SPEC M4：OU 时变洋流 + P2 隐状态探针 + GPU 全量配置

单一事实来源（M4 阶段）。遵循 SPEC.md / SPEC_M1.md / SPEC_M2.md（含勘误）。

## 0. 范围与分工

- 沙箱：实现 + 测试 + 冒烟。**不做全量训练**（用户在 GPU 上执行）
- GPU（用户）：M2 全量重训（200k×3seed）、M4 OU 训练、P4 网格、P2 正式结果
- 科学目标（提案 §6 P2）：检验 RSSM 隐状态是否编码了**不可观测**的时变洋流——
  这是 world model 相对 SAC 唯一的结构性优势，且不需要策略收敛即可测量

## 1. OU 时变洋流（动力学层）

### 1.1 `uwm/dynamics/fossen.py` 扩展

`cfg["current"]` 向后兼容两种形式：
- 列表 `[u_c, v_c]`：常值洋流（现状，不变）
- 字典：
```yaml
current:
  type: ou
  mean: [0.3, 0.0]      # 均值回归目标 (m/s, NED)
  theta: 0.1            # 回归速率 (1/s)，相关时间 = 1/theta = 10s
  sigma: 0.05           # 扩散强度 (m/s/√s)
  seed: 0               # OU 过程独立随机流
```

OU 更新（每个物理 dt，Euler–Maruyama）：
`C ← C + theta*(mean−C)*dt + sigma*√dt*ξ`，ξ~N(0,I₂)

- OU 状态存于 Fossen3DOF 实例（`self.ou_state`），`step()` 内部推进
- `reset_current(seed=None)`：重置到 mean 并重建随机流（env.reset 时调用）
- 随机流用独立 `torch.Generator`（不与全局 RNG 混杂，保证 seed 可复现）
- `current` 属性语义不变（返回当前 C 值），下游零改动

### 1.2 `configs/vehicle/bluerov2_ou.yaml`

bluerov2.yaml 的副本，仅 `current` 改为 §1.1 的 OU 字典（mean [0.3, 0.0]，θ=0.1，σ=0.05）。
参数依据：0.3 m/s 中层流典型量级；相关时间 10s ≈ 2.4× 载体时间常数——
快得需要时序推断、慢得可学习（提案 §8.5：常值洋流是假成功，必须时变）。

### 1.3 环境层

- `UWEnvBase.reset()` 调用 `fossen.reset_current()`；obs 不变（洋流不可观测，7 维）
- `step()` 的 info 增加 `info["current"] = [u_c, v_c]`（真值通道，仅供记录/探针）

### 1.4 dv3 适配器：真值走"暗通道"

- 适配器 obs dict 增加 `"current_gt"` 键（Box(2)，**不在** observation_space 声明中
  给 encoder 用——`mlp_keys: 'state'` 已保证模型永远看不到它）
- 目的：dv3 的 episode cache 会存全部 obs 键 → P2 探针从 logdir 直接拿到洋流真值，
  无需改动上游存储代码。observation_space 中声明它（dv3 需要键存在于 spaces），
  但 encoder/decoder 的 mlp_keys 严格 'state'。

## 2. P2 探针（`uwm/eval/wm_probes.py` 扩展）

```python
def hidden_state_current_probe(agent, episodes, test_ratio=0.3) -> dict:
    """P2：RSSM 确定性状态 h_t → 洋流 [u_c, v_c] 的线性探针。
    1) 重放 episode：obs_step 序列取 deter h_t（不用 stoch，deter 是信息载体）
    2) 按 episode 划分 train/test（防泄漏）
    3) Ridge 回归（闭式解），报告 {r2_u, r2_v, r2_mean}
    4) 对照基线：从瞬时 state obs（7维）做同样回归（应 ≈ 0——单帧不含洋流信息）
    返回 {"r2_u":..,"r2_v":..,"r2_mean":..,"baseline_r2_mean":..,"n_train":..,"n_test":..}"""
```

- CLI：`python -m uwm.eval.wm_probes <run_dir> --p2` 输出 P2 并画 `p2_probe.png`
  （真值 vs 预测散点，train/test 分色）
- 判读标准（写进 docstring 与 README）：R² > 0.5 = RSSM 明确编码洋流；
  0.1–0.5 = 弱编码；< 0.1 = 未编码（若策略已收敛则更说明问题）

## 3. GPU 全量配置（用户执行）

### 3.1 `configs/exp/m2_dreamerv3_gpu.yaml`（常值流，M2 全量复刻）
上游 dmc_proprio 标准档（不再降档）：`device: cuda:0`、`steps: 200000`、
`train_ratio: 512`、`batch_size: 16`、`batch_length: 64`、dyn_* 与 units 用上游默认（512 档）、
`encoder/decoder` 默认层（不再缩小）、`eval_episode_num: 10`、`eval_every: 10000`、
`prefill: 2500`、`compile: false`（兼容性优先）、seeds [0,1,2]。imag_horizon 15。

### 3.2 `configs/exp/m4_dreamerv3_ou_gpu.yaml`
同 §3.1，差异：`exp_id: m4_dreamerv3_ou`、`imag_horizon: 30`（P4 结论后的首选）、
环境变量 `UWM_VEHICLE_CFG=configs/vehicle/bluerov2_ou.yaml`（适配器读取，见 §4）。

### 3.3 `configs/exp/m1_sac_station_keeping_ou.yaml`
SAC 在 OU 环境的基线（vehicle: bluerov2_ou.yaml，120k×3seed）。

### 3.4 `scripts/gpu_run_all.sh`
顺序执行脚本（含注释说明预计时长与顺序理由）：
1. SAC × 3 seed（常值 + OU）
2. DreamerV3 常值 × 3 seed（M2 全量）
3. DreamerV3 OU × 3 seed（M4）
4. 每次 Dreamer 训练后自动 convert + P1 + P2（OU run）
附注：H 网格 {15,30,45} 为可选追加（改 imag_horizon 重跑 §3.2 即可）。

### 3.5 适配器读 `UWM_VEHICLE_CFG`
dv3 适配器 `envs/uwm.py` 从环境变量 `UWM_VEHICLE_CFG` 读载体 yaml 路径
（默认 `configs/vehicle/bluerov2.yaml`，向后兼容）；`train_dreamer.py` 从
exp yaml 的 `vehicle:` 字段设置该环境变量（子进程隔离，无副作用）。

## 4. 测试（`tests/test_ou_current.py` + P2 smoke）

1. **OU 统计性质**：10⁴ 步样本均值 ≈ mean（容差 3σ/√N·√(2/θ)）；经验 lag-1 自相关
   ≈ exp(−θ·dt)（容差 0.05）
2. **可复现**：同 seed 两次 reset_current + 100 步序列逐位相同；不同 seed 不同
3. **向后兼容**：列表形式 current 行为与 v0.1 完全一致（既有 30 项不破）
4. **env 集成**：OU 环境下 reset 可复现（同 seed 同初始洋流）、obs 仍 7 维无洋流泄漏、
   info["current"] 形状正确且随时间变化
5. **dv3 适配器**：obs dict 含 current_gt；tiny 冒烟训练（沿用 m2 smoke 配置 + OU）跑通
6. **P2 smoke**：冒烟 ckpt 上 hidden_state_current_probe 跑通，返回字段齐全、
   R² 为有限值、基线同算（不断言 R² 大小，只验证管线）

## 5. 工程约定

- OU 默认关闭（列表即常值），M0–M2 全部既有结果不受影响、无需重跑
- GPU 配置文件头部注释写明预期显存/时长（按 dv3 官方 dmc_proprio 量级估算）
- README 增 "GPU 全量复现实验指南" 一节：环境安装、三条命令、预期产出清单
- 提交：`M4: <内容>`

## 6. 完成判据（Gate）

1. pytest 全绿（既有 30 + 新增 ~6）
2. OU 冒烟训练（tiny，≤10 min）+ P2 管线冒烟通过
3. `bash -n scripts/gpu_run_all.sh` 语法检查 + 三个 GPU 配置能被 train_dreamer.py 正确解析（dry-run 到 config 构建为止）
