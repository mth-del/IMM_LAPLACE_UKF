"""
仿真/验证总控：注册表 + 命令行调度。

新增一种验证时：
1. 在 sims/ 下实现 `run_from_ns(ns: argparse.Namespace) -> None`（或包一层 thin wrapper）；
2. 在本文件用 `@register_sim("名称", "一句话说明")` 注册；
3. 若需新参数，在 `build_argparser()` 里补充，并在对应 `run_from_ns` 里 `getattr(ns, ...)` 读取。

用法：
  python main.py --list-sims
  python main.py --sim baseline
  python main.py --sim marine --marine-ini configs/marine.ini --no-plots
  python main.py --sim all --gui
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections import OrderedDict
from typing import Callable

SimHandler = Callable[[Namespace], None]

_SIM_REGISTRY: OrderedDict[str, tuple[str, SimHandler]] = OrderedDict()


def register_sim(name: str, description: str = "") -> Callable[[SimHandler], SimHandler]:
    """注册一个可在 `--sim` 中调用的验证流水线（名称小写）。"""

    key = name.strip().lower()
    if not key:
        raise ValueError("sim name must be non-empty")

    def decorator(fn: SimHandler) -> SimHandler:
        if key in _SIM_REGISTRY:
            raise ValueError(f"sim 名称重复: {key}")
        _SIM_REGISTRY[key] = (description.strip(), fn)
        return fn

    return decorator


def list_registered_sims() -> list[tuple[str, str]]:
    return [(name, desc) for name, (desc, _) in _SIM_REGISTRY.items()]


def resolve_sim_names(requested: list[str]) -> list[str]:
    req = [r.strip().lower() for r in requested if r.strip()]
    if not req:
        return ["baseline"]
    if "all" in req:
        return list(_SIM_REGISTRY.keys())
    unknown = [r for r in req if r not in _SIM_REGISTRY]
    if unknown:
        avail = ", ".join(_SIM_REGISTRY.keys())
        raise SystemExit(f"未知的 --sim 名称: {unknown}。已注册: {avail}")
    return req


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="USBL 融合 — 多仿真总控")
    p.add_argument(
        "--sim",
        nargs="+",
        default=["baseline"],
        metavar="NAME",
        help="要运行的注册名：baseline | marine | dataset | all",
    )
    p.add_argument(
        "--list-sims",
        action="store_true",
        help="列出已注册的 sim 并退出",
    )
    p.add_argument("--gui", action="store_true", help="需要出图时弹窗（否则保存 PNG）")

    # 与 baseline / marine 共用：输出目录（空则各 sim 使用自己的默认）
    p.add_argument("--plots-dir", default="", help="出图目录（baseline/marine；空则用各自默认）")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="baseline/marine/dataset：跳过保存或显示图",
    )

    # baseline / marine 共用：合成仿真 ini
    p.add_argument("--sim-ini", default="", help="sim+noise 配置路径（空则 configs/sim_noise.ini）")

    # marine
    p.add_argument("--marine-ini", default="", help="configs/marine.ini")
    p.add_argument("--plot-prefix", default="marine_sim", help="marine 保存图文件名前缀")
    p.add_argument("--marine-seed", type=int, default=-1, help="海洋扰动 RNG；-1 为自动")

    # dataset（CSV）
    p.add_argument("--dataset-dir", default="", help="含 imu/dvl/gt 的目录")
    p.add_argument("--out-csv", default="", help="dataset 输出 fused CSV 路径")
    p.add_argument(
        "--dataset-plots-dir",
        default="",
        help="dataset 出图目录（空则 outputs/plots_dataset）",
    )
    p.add_argument("--plot-only", action="store_true", help="dataset：只读 fused CSV 重画图")
    p.add_argument("--acc-noise-std", type=float, default=0.05)
    p.add_argument("--dvl-noise-std", type=float, default=0.02)
    p.add_argument("--usbl-dummy-std", type=float, default=1.0)
    return p


def run_sims(ns: Namespace, names: list[str]) -> None:
    from tools.logging_setup import logger

    for name in names:
        desc, fn = _SIM_REGISTRY[name]
        logger.info("======== 运行 sim: {} — {} ========", name, desc or "(无说明)")
        fn(ns)


def cli_main(argv: list[str] | None = None) -> None:
    from tools.logging_setup import logger, setup_logger

    parser = build_argparser()
    ns = parser.parse_args(argv)
    setup_logger()

    if ns.list_sims:
        for name, desc in list_registered_sims():
            print(f"  {name}\t{desc}")
        return

    names = resolve_sim_names(ns.sim)
    logger.info("总控调度：{}", " -> ".join(names))
    run_sims(ns, names)


# --- 在此处注册各验证（新增 sim 只改本段 + sims/ 实现）---


@register_sim("baseline", "合成真值 + 高斯传感器 + INS/EKF 融合")
def _sim_baseline(ns: Namespace) -> None:
    from sims.baseline import run_from_ns

    run_from_ns(ns)


@register_sim("marine", "海洋环境扰动层 + 与 baseline 相同 EKF 流水线")
def _sim_marine(ns: Namespace) -> None:
    from sims.marine import run_from_ns

    run_from_ns(ns)


@register_sim("dataset", "真实 CSV（IMU+DVL+GT）二维 EKF 融合")
def _sim_dataset(ns: Namespace) -> None:
    from sims.dataset import run_from_ns

    run_from_ns(ns)
