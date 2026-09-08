"""UWM 环境适配器（SPEC_M2 §2.1，本文件为 vendored 上游中唯一新增的 env 文件）。

把 uwm.envs.station_keeping.StationKeepingEnv（gymnasium，5 元组 step）包成
上游 dreamerv3-torch 期望的旧 gym 接口（4 元组 step + gym.spaces），
模仿 envs/dmc.py 的 DeepMindControl。

关于 "image" 占位键：上游 tools.simulate 在 episode 结束时无条件读取
cache[id]["image"]，WorldModel.preprocess 也无条件访问 obs["image"]，而
SPEC_M2 又规定 observation_space 只声明 "state"。因此 observation_space
只暴露 state（encoder/decoder 配置 mlp_keys='state', cnn_keys='$^'，编解码
均不触碰 image），step/reset 返回的 obs dict 额外携带一个小的全零
"image" 键满足上游数据管线；它是哑数据，不参与编解码与损失。

M4（SPEC_M4 §1.4/§3.5）：
- 载体 yaml 路径从环境变量 UWM_VEHICLE_CFG 读取（默认
  configs/vehicle/bluerov2.yaml，向后兼容）；train_dreamer.py 会按 exp
  yaml 的 vehicle 字段设置该变量（进程内设置，子进程隔离，无副作用）。
- obs dict 增加 "current_gt" 暗通道（洋流真值 Box(2)）：在
  observation_space 中声明（dv3 的 episode cache 需要键存在于 spaces 才会
  落盘），但 encoder/decoder 的 mlp_keys 严格保持 'state'——正则不匹配
  "current_gt"，模型在结构上永远看不到它。用途：dv3 episode cache 会存
  全部 obs 键 → P2 探针（hidden_state_current_probe）直接从 logdir 的
  npz 拿到洋流真值，无需改动上游存储代码。
"""

import os
import pathlib

import numpy as np
import yaml

try:
    import gym
except ImportError:
    # UWM 仓库环境无旧 gym：注入 compat shim 后重试（见 uwm/compat.py）
    from uwm.compat import ensure_upstream_compat

    ensure_upstream_compat()
    import gym

# vendored 根（third_party/dreamerv3-torch）上溯两级 = 仓库根
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


class UWMStationKeeping:
    """StationKeepingEnv → 上游接口。name 形如 "uwm_station_keeping"。"""

    metadata = {}
    reward_range = [-np.inf, np.inf]

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        from uwm.envs.station_keeping import StationKeepingEnv

        task = name[len("uwm_"):] if name.startswith("uwm_") else name
        if task != "station_keeping":
            raise NotImplementedError(f"未知 uwm 任务: {name!r}（仅支持 uwm_station_keeping）")
        # SPEC_M4 §3.5：载体配置走 UWM_VEHICLE_CFG（默认常值流 bluerov2.yaml）
        vehicle_path = pathlib.Path(
            os.environ.get("UWM_VEHICLE_CFG", "configs/vehicle/bluerov2.yaml")
        )
        if not vehicle_path.is_absolute():
            vehicle_path = _REPO_ROOT / vehicle_path
        with open(vehicle_path, encoding="utf-8") as f:
            vehicle_cfg = yaml.safe_load(f)
        # 环境内部已 action_repeat=5（0.1s/env step）；dv3 配置 action_repeat=1，
        # 这里仍按 dmc 写法支持额外的外部 action_repeat（配置层保持 1）。
        self._env = StationKeepingEnv(vehicle_cfg, {})
        self._action_repeat = int(action_repeat)
        self._size = tuple(size)
        self._seed = int(seed)
        self._needs_seed = True
        self._image = np.zeros(self._size + (3,), dtype=np.uint8)  # 占位哑数据

    @property
    def observation_space(self):
        # "state" 是唯一的模型输入（encoder/decoder mlp_keys 严格 'state'）；
        # "current_gt" 为 M4 暗通道（SPEC_M4 §1.4）：声明在此仅为了让 dv3 的
        # episode cache 落盘洋流真值，encoder/decoder 的正则键永不匹配它。
        # 不声明 "image"（哑数据，满足上游数据管线即可）。
        return gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
                "current_gt": gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    def _obs(self, obs, is_first):
        return {
            "state": np.asarray(obs, dtype=np.float32),
            # 洋流真值暗通道（仅供 P2 探针，模型不可见，见 observation_space 注释）
            "current_gt": np.asarray(self._env.current_truth(), dtype=np.float32),
            "image": self._image,
            "is_terminal": False,
            "is_first": bool(is_first),
        }

    def step(self, action):
        assert np.isfinite(action).all(), action
        action = np.asarray(action, dtype=np.float64)
        reward = 0.0
        for _ in range(self._action_repeat):
            obs, r, terminated, truncated, info = self._env.step(action)
            reward += float(r)
            if terminated or truncated:
                break
        done = bool(terminated or truncated)
        out = self._obs(obs, is_first=False)
        out["is_terminal"] = bool(terminated)
        # truncated（TimeLimit 截断）不是终止：discount=1.0 让上游照常 bootstrap
        info = {"discount": np.array(1.0 - float(terminated), np.float32)}
        return out, reward, done, info

    def reset(self):
        if self._needs_seed:  # 首次 reset 播种，之后沿用环境内部 np_random 序列
            obs, _ = self._env.reset(seed=self._seed)
            self._needs_seed = False
        else:
            obs, _ = self._env.reset()
        return self._obs(obs, is_first=True)

    def render(self, *args, **kwargs):
        return self._image.copy()
