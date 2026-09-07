"""dreamerv3-torch 配置构造（train_dreamer 与 wm_probes 共用，SPEC_M2 §2.3）。

把我们的 exp yaml 里的 ``dv3_overrides`` 叠加到 vendored ``configs.yaml`` 的
``defaults`` + ``dmc_proprio`` preset 上，产出上游 ``dreamer.main`` 接受的
Namespace 配置对象。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import ruamel.yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDORED_ROOT = REPO_ROOT / "third_party" / "dreamerv3-torch"


def add_vendored_path() -> None:
    """把 vendored 根插到 sys.path 最前（幂等）。"""
    path = str(VENDORED_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _recursive_update(base: dict, update: dict) -> None:
    """与上游 dreamer.py __main__ 的 recursive_update 同语义。"""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value


def apply_cpu_fallback() -> None:
    """无 CUDA 时把 networks.MLP 缺省 device="cuda" 运行时回落到 "cpu"。

    上游 networks.MLP.__init__ 的 device 形参缺省 "cuda"（networks.py:606，
    vendored 唯一硬编码），MultiEncoder/MultiDecoder 构造 MLP 时不传 device，
    无 GPU 机器上创建 self._std 张量即触发 cuda init 报错。此处运行时包装
    MLP.__init__（幂等），vendored 文件零改动。所有调用点（models.py 等处）
    都以关键字传 device，无缺省位置实参，故只需处理 kwargs。
    """
    import torch

    if torch.cuda.is_available():
        return
    add_vendored_path()
    import networks

    if getattr(networks.MLP, "_uwm_cpu_fallback", False):
        return
    orig_init = networks.MLP.__init__

    def _init_with_cpu_fallback(self, *args, **kwargs):
        if kwargs.get("device", "cuda") == "cuda":
            kwargs["device"] = "cpu"
        orig_init(self, *args, **kwargs)

    networks.MLP.__init__ = _init_with_cpu_fallback
    networks.MLP._uwm_cpu_fallback = True


def build_dv3_config(dv3_overrides: dict, preset: str = "dmc_proprio") -> argparse.Namespace:
    """defaults + preset + dv3_overrides → argparse.Namespace。"""
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    configs = yaml.load((VENDORED_ROOT / "configs.yaml").read_text())
    merged: dict = {}
    for name in ("defaults", preset):
        _recursive_update(merged, configs[name])
    _recursive_update(merged, dict(dv3_overrides))
    return argparse.Namespace(**merged)


def apply_chunked_ckpt_patch(ckpt_every: int = 500) -> None:
    """把 tools.simulate 运行时包装为按 ckpt_every 分段执行、段间保存 latest.pt（幂等）。

    动机：上游 dreamer.main 只在一次 simulate() 完整返回后才保存 checkpoint，
    即每 eval_every 步（4000）存一次。沙箱分段训练单段约 250s（≈1000 步），
    跑不完一个保存周期，导致 latest.pt 长期停留在早期版本，
    每次重启都从陈旧 ckpt 回滚（实测：update_count 在重启后回到 100，
    模型损失回升，段内学习成果全部丢失）。

    本补丁在驱动层运行时包装 tools.simulate：训练调用（非 eval、按步数）
    拆成 ckpt_every 步的小段，每段结束后保存 latest.pt（含优化器状态，
    与上游保存内容一致）。vendored 上游文件零改动。
    """
    add_vendored_path()
    import tools

    if getattr(tools, "_uwm_chunked_ckpt", False):
        return
    import torch

    orig_simulate = tools.simulate

    def simulate_chunked(
        agent, envs, cache, directory, logger,
        is_eval=False, limit=None, steps=0, episodes=0, state=None,
    ):
        # eval、非按步数调用、以及 prefill 的随机策略（callable 而非 nn.Module）
        # 走原逻辑
        if is_eval or not steps or not hasattr(agent, "state_dict"):
            return orig_simulate(
                agent, envs, cache, directory, logger,
                is_eval=is_eval, limit=limit, steps=steps,
                episodes=episodes, state=state,
            )
        # directory = config.traindir = logdir/train_eps → 上级即 logdir
        logdir = pathlib.Path(directory).parent
        remaining = steps
        while remaining > 0:
            n = min(ckpt_every, remaining)
            state = orig_simulate(
                agent, envs, cache, directory, logger,
                limit=limit, steps=n, state=state,
            )
            remaining -= n
            items_to_save = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }
            # NaN 防护：参数非有限则跳过保存（绝不落毒 ckpt）
            finite = all(
                bool(torch.isfinite(v).all())
                for v in items_to_save["agent_state_dict"].values()
                if torch.is_tensor(v)
            )
            if not finite:
                print("[uwm] NaN/Inf in agent params, skip ckpt save", flush=True)
                continue
            # 轮转备份：崩溃/NaN 时可回滚到上一个 ckpt（训练发散有实测前科）
            prev = logdir / "latest.pt"
            if prev.exists():
                import shutil

                shutil.copy2(prev, logdir / "latest_prev.pt")
            torch.save(items_to_save, prev)
        return state

    tools.simulate = simulate_chunked
    tools._uwm_chunked_ckpt = True


def apply_nan_canary(dump_path: str | pathlib.Path) -> None:
    """NaN 金丝雀（幂等）：包装 networks.RSSM.obs_step/img_step。

    崩溃前把完整上下文（输入 embed/action/latent 的有限性与 absmax、
    全部参数的 absmax -top 列表）dump 到 dump_path，再原样抛出。
    用于定位间歇性 NaN（三次崩溃分别在 ~7.5k/2k/2k 步，ckpt 参数均有限）。
    vendored 上游零改动。
    """
    add_vendored_path()
    import networks
    import torch

    if getattr(networks.RSSM, "_uwm_nan_canary", False):
        return
    orig_obs_step = networks.RSSM.obs_step

    def _stats(t):
        if not torch.is_tensor(t):
            return None
        return {
            "finite": bool(torch.isfinite(t).all()),
            "absmax": float(t.abs().max()) if t.numel() else 0.0,
        }

    def obs_step_canary(self, prev_state, prev_action, embed, is_first, sample=True):
        try:
            post, prior = orig_obs_step(
                self, prev_state, prev_action, embed, is_first, sample
            )
        except Exception:
            _dump(self, prev_state, prev_action, embed, torch, dump_path)
            raise
        for name, lat in (("post", post), ("prior", prior)):
            for k, v in lat.items():
                if torch.is_tensor(v) and not torch.isfinite(v).all():
                    print(f"[uwm canary] NaN in {name}[{k}]", flush=True)
                    _dump(self, prev_state, prev_action, embed, torch, dump_path)
                    raise FloatingPointError(f"NaN in {name}[{k}]")
        return post, prior

    def _dump(self, prev_state, prev_action, embed, torch, dump_path):
        import numpy as np

        payload = {"embed": _stats(embed), "prev_action": _stats(prev_action)}
        if prev_state:
            for k, v in prev_state.items():
                payload[f"prev_state.{k}"] = _stats(v)
        bad_params = []
        for k, v in self.state_dict().items():
            if torch.is_tensor(v):
                if not torch.isfinite(v).all():
                    bad_params.append((k, "NONFINITE"))
        payload["nonfinite_params"] = bad_params
        # 输入原始值（供复算）
        for name, t in (("embed", embed), ("prev_action", prev_action)):
            if torch.is_tensor(t):
                payload[f"raw.{name}"] = t.detach().cpu().numpy()
        if prev_state:
            for k, v in prev_state.items():
                if torch.is_tensor(v):
                    payload[f"raw.prev_state.{k}"] = v.detach().cpu().numpy()
        np.savez(str(dump_path), **{k: v for k, v in payload.items() if v is not None})
        print(f"[uwm canary] dump -> {dump_path}", flush=True)

    networks.RSSM.obs_step = obs_step_canary
    networks.RSSM._uwm_nan_canary = True


def apply_encoder_canary(dump_path: str | pathlib.Path) -> None:
    """encoder 输入/输出金丝雀（幂等）：embed 出现 NaN 时 dump 原始 obs。

    第一次金丝雀确认 NaN 来自 embed（encoder 输出全 256 维 NaN），
    但 RSSM 参数有限、未覆盖 obs 本身与 encoder 参数。本补丁包住
    MultiEncoder.forward：输出非有限时把输入 obs 各键的有限性/极值
    与原始值、encoder 参数有限性一并 dump。vendored 上游零改动。
    """
    add_vendored_path()
    import networks
    import torch

    if getattr(networks.MultiEncoder, "_uwm_enc_canary", False):
        return
    orig_forward = networks.MultiEncoder.forward

    def forward_canary(self, data):
        out = orig_forward(self, data)
        if torch.is_tensor(out) and not torch.isfinite(out).all():
            import numpy as np

            payload = {}
            for k, v in data.items():
                if torch.is_tensor(v):
                    vv = v.detach().cpu().numpy()
                    payload[f"obs.{k}"] = vv
                    print(
                        f"[uwm enc-canary] obs[{k}] finite={np.isfinite(vv).all()} "
                        f"absmax={np.abs(vv).max() if vv.size else 0:.3g}",
                        flush=True,
                    )
            bad = [
                k
                for k, p in self.state_dict().items()
                if torch.is_tensor(p) and not torch.isfinite(p).all()
            ]
            payload["nonfinite_encoder_params"] = np.array(bad, dtype=object)
            print(f"[uwm enc-canary] nonfinite encoder params: {bad}", flush=True)
            np.savez(str(dump_path), **payload)
            print(f"[uwm enc-canary] dump -> {dump_path}", flush=True)
        return out

    networks.MultiEncoder.forward = forward_canary
    networks.MultiEncoder._uwm_enc_canary = True
