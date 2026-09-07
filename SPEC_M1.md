# UWM — SPEC M1：SAC 基线 + state 观测 + station_keeping

单一事实来源（M1 阶段）。遵循 SPEC.md（M0）全部约定；冲突时以本文件为准。

## 0. 范围

- 手写单文件 SAC（cleanrl 风格，只依赖 torch），不接 Stable-Baselines3（避免新依赖）
- 训练入口 + 实验配置 + EpisodeRecorder 集成 + 断点续训
- 确定性评估脚本，按 M1 通过判据出结论
- **不做**：DreamerV3、pixel 观测、新任务、新动力学项

依赖上限不变（SPEC §0 七项）。不引入 SB3/wandb/hydra。

## 1. 目录新增

```
uwm/
├── algos/
│   ├── __init__.py
│   └── sac.py              # 单文件 SAC（Actor + 双 Q + target net + 自适应 alpha）
├── configs/
│   └── exp/
│       └── m1_sac_station_keeping.yaml
├── scripts/
│   ├── train.py            # 训练入口：python scripts/train.py configs/exp/xxx.yaml [--resume]
│   └── eval_policy.py      # 确定性评估 + 误差统计 + 存 eval episodes 供面板 B 回放
└── tests/
    └── test_sac.py
```

## 2. 接口合同

### 2.1 `algos/sac.py`

cleanrl 单文件风格（参考其 sac_continuous_action.py 的结构，不复制其代码）：

```python
class SAC:
    """连续控制 SAC。Actor: tanh-squashed 高斯；Critic: 双 Q + target；alpha 自适应。"""
    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, device: str = "cpu"): ...
        # cfg 字段: hidden(=256), lr(=3e-4), gamma(=0.99), tau(=0.005),
        #           batch_size(=256), buffer_size(=200_000), warmup_steps(=2000),
        #           target_entropy(=-act_dim)
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """返回 [−1,1]^act_dim。warmup 期间由 train.py 用均匀随机，不走这里。"""
    def update(self, batch: dict) -> dict:
        """一次梯度步，返回 {q_loss, actor_loss, alpha, alpha_loss}（float）。"""
    def save(self, path) -> None: ...   # 含所有网络+优化器+alpha+rng 状态
    def load(self, path) -> None: ...

class ReplayBuffer:
    """numpy 环形缓冲。add(obs, act, rew, next_obs, done)；sample(batch_size)->dict of torch。"""
```

MLP：2 层 hidden=256，ReLU。全部 float32（训练侧；环境内部仍 float64，接口处转换）。

### 2.2 `configs/exp/m1_sac_station_keeping.yaml`

```yaml
exp_id: m1_sac_station_keeping
vehicle: configs/vehicle/bluerov2.yaml
task: {}                    # StationKeepingEnv 默认
seeds: [0, 1, 2]
total_steps: 120_000        # 环境步（= 12000 物理秒 ÷ 0.1s/env step）
eval_every: 10_000          # 每 N 步跑一次确定性评估（10 episode）
eval_episodes: 10
record_ep_every: 20         # 每 20 个 episode 存一份 npz
sac: {hidden: 256, lr: 0.0003, gamma: 0.99, tau: 0.005, batch_size: 256,
      buffer_size: 200000, warmup_steps: 2000}
```

### 2.3 `scripts/train.py`

- `python scripts/train.py <exp.yaml> [--seed N] [--resume <ckpt>] [--total-steps M]`
- 每个 (exp_id, seed) 一个 run 目录：`runs/<exp_id>/seed<N>/`，复用 EpisodeRecorder
- metrics.csv 每 episode 追加：episode, env_steps, return_, final_dist, ep_len, q_loss, actor_loss, alpha
- 每 eval_every 步：确定性评估 eval_episodes 个 episode，写 metrics.csv（is_eval=1 行）
- 定期存 ckpt（`ckpt/latest.pt`），`--resume` 可续训（沙箱超时友好：分段跑）
- 训练结束画样本效率曲线：`runs/<exp_id>/learning_curve_seed<N>.png`（return vs env_steps，eval 点 + 训练滑动平均）

### 2.4 `scripts/eval_policy.py`

- `python scripts/eval_policy.py <run_dir> [--episodes 20]`
- 确定性策略跑 N 个 episode，输出：final_dist 均值/中位数/max、最后 5 秒位置 std（振荡指标）、return 均值
- episodes 存到 `<run_dir>/eval/episodes/`（面板 B 可直接回放）
- **M1 通过判据**（写进脚本 docstring 与 README）：
  - final_dist 中位数 < 0.3 m
  - 最后 5 秒位置 std < 0.05 m（无持续振荡）

## 3. 测试 `tests/test_sac.py`

1. **冒烟**：tiny 配置（buffer 500、batch 32、hidden 32）训练 1500 步跑通，loss 全部 finite
2. **动作界**：select_action 输出恒 ∈ [−1,1]
3. **ckpt 一致性**：save→load 后同 obs 确定性动作逐位相同；resume 后 update 正常
4. **buffer 环形**：超过 capacity 后覆盖正确，sample 形状正确
5. 既有 18 项测试不得破坏

## 4. 工程约定

- 训练侧 float32；与环境交互处 obs.astype(np.float32)
- 每个 run 目录写 config.yaml（recorder 已支持），含 git commit hash（若可取）
- 负结果同样记录：若 120k 步未收敛，metrics 与曲线照存，报告如实写
- 提交信息：`M1: <内容>`

## 5. 完成判据（Gate）

1. `pytest tests/ -p no:cacheprovider` 全绿（含新增 4 项）
2. 冒烟训练（tiny 配置）在干净 venv 跑通
3. eval_policy.py 能对冒烟 ckpt 输出判据指标
4. 零新依赖
