"""vendored dreamerv3-torch 的兼容层（SPEC_M2 §1 / §3）。

上游（commit 6ef8646）依赖两个本仓库主栈没有的包：

1. 旧版 ``gym``：上游只用了 ``gym.Wrapper`` / ``gym.spaces.Box`` /
   ``gym.spaces.Dict`` 三个基础 API（envs/wrappers.py、envs/dmc.py）。
   gym 0.22 在 py3.12 上装不上，且 SPEC_M2 §0 只允许新增 ruamel.yaml /
   einops 两个依赖，因此这里提供一个 ~50 行的 gym shim，在 ``import gym``
   失败时注入 ``sys.modules``。若环境里有真 gym 则原样使用，不注入。
2. ``tensorboard``：上游 tools.Logger 只用 ``SummaryWriter`` 写标量/视频，
   我们消费的是 logdir 里的 ``metrics.jsonl``，tensorboard 事件文件无用。
   ``import tensorboard`` 失败时给 ``torch.utils.tensorboard`` 注入一个
   no-op SummaryWriter shim。

入口（scripts/train_dreamer.py、uwm/eval/wm_probes.py、tests）在 import 上游
模块之前调用 :func:`ensure_upstream_compat` 即可，上游代码零改动。
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def _build_gym_shim() -> types.ModuleType:
    """构造最小 gym shim：Wrapper + spaces.Box / spaces.Dict。"""
    gym = types.ModuleType("gym")
    spaces = types.ModuleType("gym.spaces")

    class Box:
        """对标 gym.spaces.Box 的最小实现（low/high/shape/dtype/sample）。"""

        def __init__(self, low, high, shape=None, dtype=np.float32):
            dtype = np.dtype(dtype)
            low = np.asarray(low, dtype=dtype)
            high = np.asarray(high, dtype=dtype)
            if shape is None:
                shape = np.broadcast(low, high).shape
            self.shape = tuple(shape)
            self.low = np.broadcast_to(low, self.shape).astype(dtype).copy()
            self.high = np.broadcast_to(high, self.shape).astype(dtype).copy()
            self.dtype = dtype

        def sample(self):
            lo, hi = self.low, self.high
            finite = np.isfinite(lo) & np.isfinite(hi)
            out = np.where(finite, lo, 0.0).astype(np.float64)
            out = out + np.where(
                finite,
                np.random.random(self.shape) * np.where(finite, hi - lo, 1.0),
                np.random.standard_normal(self.shape),
            )
            return out.astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return (
                x.shape == self.shape
                and bool(np.all(x >= self.low))
                and bool(np.all(x <= self.high))
            )

        def __repr__(self):
            return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})"

    class Dict:
        """对标 gym.spaces.Dict 的最小实现（.spaces 字典）。"""

        def __init__(self, spaces_dict):
            self.spaces = dict(spaces_dict)

        def __getitem__(self, key):
            return self.spaces[key]

        def __repr__(self):
            return f"Dict({list(self.spaces)})"

    class Wrapper:
        """对标 gym.Wrapper：属性转发到底层 env。"""

        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            return getattr(self.env, name)

        def step(self, action):
            return self.env.step(action)

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

    spaces.Box = Box
    spaces.Dict = Dict
    gym.spaces = spaces
    gym.Wrapper = Wrapper
    return gym


def ensure_upstream_compat() -> None:
    """按需注入 gym / tensorboard shim（幂等）。"""
    try:
        importlib.import_module("gym")
    except ImportError:
        gym = _build_gym_shim()
        sys.modules["gym"] = gym
        sys.modules["gym.spaces"] = gym.spaces

    try:
        importlib.import_module("tensorboard")
    except ImportError:
        tb = types.ModuleType("torch.utils.tensorboard")

        class SummaryWriter:  # noqa: D101 — no-op shim
            def __init__(self, *args, **kwargs):
                pass

            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        tb.SummaryWriter = SummaryWriter
        sys.modules["torch.utils.tensorboard"] = tb
