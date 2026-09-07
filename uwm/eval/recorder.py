"""Episode 录制器（提案 §7.1 文件契约）。

目录结构：

    runs/<exp_id>/
    ├── config.yaml
    ├── metrics.csv            # 追加写
    └── episodes/ep_0100.npz   # t, eta, nu, action, reward

只用 numpy / 标准库 / pyyaml，不引入新依赖（SPEC §0）。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml


class EpisodeRecorder:
    """把训练/评测过程落盘：配置、标量指标、逐 episode 轨迹。

    run_dir: 运行目录（如 runs/demo），不存在则创建。
    config:  任意可 yaml 序列化的配置字典，写入 run_dir/config.yaml。
    """

    def __init__(self, run_dir: str | Path, config: dict):
        self.run_dir = Path(run_dir)
        self.episodes_dir = self.run_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"

        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

        self._metric_fields: list[str] = ["step"]
        self._metrics_file = open(  # noqa: SIM115 — 生命周期随对象
            self.metrics_path, "a", newline="", encoding="utf-8"
        )
        self._metrics_writer: csv.DictWriter | None = None

    def log_metrics(self, step: int, **kv) -> None:
        """追加写一行 metrics.csv（立即 flush）。

        首次调用时按传入键确定表头（step 列在前）；后续调用的键集合
        必须与首次一致，否则抛 ValueError（CSV 无法动态加列）。
        """
        row = {"step": step, **kv}
        if self._metrics_writer is None:
            if self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
                raise ValueError("metrics.csv 已存在内容，无法对齐表头，请换新 run_dir")
            self._metric_fields = list(row.keys())
            self._metrics_writer = csv.DictWriter(
                self._metrics_file, fieldnames=self._metric_fields
            )
            self._metrics_writer.writeheader()
        if set(row.keys()) != set(self._metric_fields):
            raise ValueError(
                f"指标键 {sorted(row.keys())} 与表头 {self._metric_fields} 不一致"
            )
        self._metrics_writer.writerow(row)
        self._metrics_file.flush()

    def save_episode(self, ep_idx: int, t, eta, nu, action, reward) -> Path:
        """保存一条 episode 轨迹到 episodes/ep_{ep_idx:04d}.npz。

        只存关键量（float64→float32 压缩体积，几十 KB 级）：
            t:      (T,)      物理时间（s）
            eta:    (T, 3)    [x, y, ψ]，单位 [m, m, rad]
            nu:     (T, 3)    [u, v, r]，单位 [m/s, m/s, rad/s]
            action: (T, 3)    施加的动作 ∈ [-1, 1]
            reward: (T,)      每步奖励
        """
        path = self.episodes_dir / f"ep_{ep_idx:04d}.npz"
        np.savez(
            path,
            t=np.asarray(t, dtype=np.float64),
            eta=np.asarray(eta, dtype=np.float32),
            nu=np.asarray(nu, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            reward=np.asarray(reward, dtype=np.float64),
        )
        return path

    def close(self) -> None:
        """关闭 metrics.csv 文件句柄。"""
        if not self._metrics_file.closed:
            self._metrics_file.flush()
            self._metrics_file.close()

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    """读取 save_episode 保存的 npz，返回 {t, eta, nu, action, reward} 字典。"""
    with np.load(Path(path)) as z:
        return {k: z[k] for k in ("t", "eta", "nu", "action", "reward")}
