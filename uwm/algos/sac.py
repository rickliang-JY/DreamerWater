"""单文件 SAC（Soft Actor-Critic），cleanrl 风格，只依赖 torch + numpy。

连续控制 SAC 要点（Haarnoja et al., 2018；实现结构参考 cleanrl
sac_continuous_action.py 的组织方式，代码为本项目手写）：

- Actor: tanh-squashed 高斯策略。采样 a = tanh(u)，u ~ N(μ, σ)；
  log_prob 必须带 tanh 修正项：log π(a|s) = log N(u|μ,σ) − Σ log(1 − tanh(u)²)。
  修正项用数值稳定形式 2·(log2 − u − softplus(−2u))（等价于 log(1−tanh²u)）。
- Critic: 双 Q（QF1/QF2）+ 对应 target 网络，actor loss 与 target 值都取 min。
- target 网络软更新：θ_target ← τ·θ + (1−τ)·θ_target，每次 update 一次。
- 自适应温度 alpha：优化 log_alpha 使熵逼近 target_entropy = −act_dim。
- done 语义：buffer 里存 terminated；TimeLimit 截断（truncated）的 bootstrap
  按未终止处理（即不把 truncated 当 done），由 train.py 保证传入的 done=terminated。

训练侧全部 float32；环境 obs 是 float64，由 train.py 在接口处 astype(np.float32)。
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOG_STD_MIN = -5.0  # log σ 下界
LOG_STD_MAX = 2.0  # log σ 上界


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """2 层 hidden（ReLU）MLP，SPEC_M1 §2.1。"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """tanh-squashed 高斯策略：输出 (μ, log σ)，动作界恒 ∈ (−1, 1)^act_dim。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden, act_dim)
        self.fc_logstd = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (mean, log_std)；log_std 裁剪到 [LOG_STD_MIN, LOG_STD_MAX]。"""
        h = self.trunk(obs)
        mean = self.fc_mean(h)
        log_std = torch.clamp(self.fc_logstd(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """采样动作。返回 (action, log_prob)；deterministic 时 action=tanh(μ)、log_prob=None。

        log_prob 含 tanh 修正（数值稳定形式）：
            log(1 − tanh(u)²) = 2·(log 2 − u − softplus(−2u))
        """
        mean, log_std = self(obs)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()  # reparameterization trick
        action = torch.tanh(u)
        # tanh 修正：log π(a) = log N(u) − Σ log(1 − tanh²u)，
        # 其中 log(1−tanh²u) = 2·(log2 − u − softplus(−2u))（数值稳定写法，
        # 避免 |u| 大时 log(1−tanh²) 下溢）。注意修正项是【减】。
        log_tanh_det = 2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))
        log_prob = (normal.log_prob(u) - log_tanh_det).sum(dim=-1)
        return action, log_prob


class SoftQNetwork(nn.Module):
    """Q(s, a) 网络：输入 obs‖act，输出标量 Q 值。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """返回 Q 值，形状 (batch,)。"""
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class ReplayBuffer:
    """numpy 环形缓冲（float32）。

    add(obs, act, rew, next_obs, done)；sample(batch_size) -> dict of torch。
    超过 capacity 后从头部覆盖（环形）。
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self._ptr = 0  # 下一条写入位置
        self._size = 0  # 当前有效条数

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """写入一条转移；done 只应取 terminated（truncated 由调用方按 False 传入）。"""
        i = self._ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.act[i] = act
        self.rew[i] = rew
        self.done[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """均匀采样一批，返回 torch 张量字典（已搬到 device）。"""
        idx = np.random.randint(0, self._size, size=batch_size)
        to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        return {
            "obs": to(self.obs[idx]),
            "act": to(self.act[idx]),
            "rew": to(self.rew[idx]),
            "next_obs": to(self.next_obs[idx]),
            "done": to(self.done[idx]),
        }

    def state_dict(self) -> dict:
        """序列化 buffer 全部状态（断点续训用）。"""
        return {
            "capacity": self.capacity,
            "obs": self.obs,
            "next_obs": self.next_obs,
            "act": self.act,
            "rew": self.rew,
            "done": self.done,
            "ptr": self._ptr,
            "size": self._size,
        }

    def load_state_dict(self, state: dict) -> None:
        """从 state_dict 恢复（capacity 必须一致）。"""
        if int(state["capacity"]) != self.capacity:
            raise ValueError(f"buffer capacity 不一致: ckpt={state['capacity']} vs {self.capacity}")
        self.obs[...] = state["obs"]
        self.next_obs[...] = state["next_obs"]
        self.act[...] = state["act"]
        self.rew[...] = state["rew"]
        self.done[...] = state["done"]
        self._ptr = int(state["ptr"])
        self._size = int(state["size"])


class SAC:
    """连续控制 SAC。Actor: tanh-squashed 高斯；Critic: 双 Q + target；alpha 自适应。

    cfg 字段（均有默认值）:
        hidden(=256), lr(=3e-4), gamma(=0.99), tau(=0.005),
        batch_size(=256), buffer_size(=200_000), warmup_steps(=2000),
        target_entropy(=-act_dim), alpha_init(=0.2)
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, device: str = "cpu"):
        cfg = dict(cfg or {})
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)
        self.hidden = int(cfg.get("hidden", 256))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.batch_size = int(cfg.get("batch_size", 256))
        self.buffer_size = int(cfg.get("buffer_size", 200_000))
        self.warmup_steps = int(cfg.get("warmup_steps", 2000))
        lr = float(cfg.get("lr", 3e-4))
        self.target_entropy = float(cfg.get("target_entropy", -float(act_dim)))

        self.actor = Actor(obs_dim, act_dim, self.hidden).to(self.device)
        self.qf1 = SoftQNetwork(obs_dim, act_dim, self.hidden).to(self.device)
        self.qf2 = SoftQNetwork(obs_dim, act_dim, self.hidden).to(self.device)
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        for p in self.qf1_target.parameters():
            p.requires_grad_(False)
        for p in self.qf2_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr
        )
        # 自适应温度：优化 log_alpha 保证 alpha > 0
        self.log_alpha = torch.tensor(
            np.log(float(cfg.get("alpha_init", 0.2))),
            dtype=torch.float32, device=self.device, requires_grad=True,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> float:
        """当前温度 alpha = exp(log_alpha)（float）。"""
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """返回动作 ∈ [−1,1]^act_dim（numpy float32）。warmup 随机动作不走这里。"""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """一次梯度步，返回 {q_loss, actor_loss, alpha, alpha_loss}（float）。"""
        obs, act = batch["obs"], batch["act"]
        rew, next_obs, done = batch["rew"], batch["next_obs"], batch["done"]

        # ---- Critic：target = r + γ·(1−done)·(min Q_target(s',a') − α·log π(a'|s')) ----
        with torch.no_grad():
            next_act, next_log_pi = self.actor.sample(next_obs)
            q_next = torch.min(
                self.qf1_target(next_obs, next_act), self.qf2_target(next_obs, next_act)
            ) - self.log_alpha.exp() * next_log_pi
            target_q = rew + (1.0 - done) * self.gamma * q_next
        q1 = self.qf1(obs, act)
        q2 = self.qf2(obs, act)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        # ---- Actor：最大化 E[min Q(s,a) − α·log π]（冻结 Q 避免无谓反传）----
        for p in self.qf1.parameters():
            p.requires_grad_(False)
        for p in self.qf2.parameters():
            p.requires_grad_(False)
        pi, log_pi = self.actor.sample(obs)
        q_pi = torch.min(self.qf1(obs, pi), self.qf2(obs, pi))
        actor_loss = (self.log_alpha.exp().detach() * log_pi - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.qf1.parameters():
            p.requires_grad_(True)
        for p in self.qf2.parameters():
            p.requires_grad_(True)

        # ---- 自适应 alpha：J(α) = E[−α·(log π + target_entropy)] ----
        with torch.no_grad():
            _, log_pi = self.actor.sample(obs)
        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---- target 网络软更新：θ' ← τ·θ + (1−τ)·θ' ----
        with torch.no_grad():
            for net, tgt in ((self.qf1, self.qf1_target), (self.qf2, self.qf2_target)):
                for p, p_t in zip(net.parameters(), tgt.parameters()):
                    p_t.mul_(1.0 - self.tau).add_(self.tau * p)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": self.alpha,
            "alpha_loss": float(alpha_loss.item()),
        }

    def _state_dict(self) -> dict:
        """打包全部状态（网络 + 优化器 + log_alpha + RNG），供 save/组合 ckpt 用。"""
        return {
            "cfg": {
                "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                "hidden": self.hidden, "gamma": self.gamma, "tau": self.tau,
                "batch_size": self.batch_size, "buffer_size": self.buffer_size,
                "warmup_steps": self.warmup_steps,
                "target_entropy": self.target_entropy,
            },
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "q_opt": self.q_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

    def _load_state_dict(self, ckpt: dict) -> None:
        """从 _state_dict 打包的字典恢复全部状态。"""
        self.actor.load_state_dict(ckpt["actor"])
        self.qf1.load_state_dict(ckpt["qf1"])
        self.qf2.load_state_dict(ckpt["qf2"])
        self.qf1_target.load_state_dict(ckpt["qf1_target"])
        self.qf2_target.load_state_dict(ckpt["qf2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q_opt.load_state_dict(ckpt["q_opt"])
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    def save(self, path: str | Path) -> None:
        """保存全部状态：网络 + 优化器 + log_alpha + RNG（torch/numpy/python）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_dict(), path)

    def load(self, path: str | Path) -> None:
        """从 save 的 ckpt 恢复；也兼容 train.py 的组合 ckpt（含 "sac" 键）。"""
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        if "sac" in ckpt:  # train.py 组合 ckpt: {"sac": ..., "buffer": ..., ...}
            ckpt = ckpt["sac"]
        self._load_state_dict(ckpt)
