#!/usr/bin/env python3
"""
在传感器仿真之后叠加海洋环境扰动，再跑 EKF。

命令行参数与历史版本兼容；实现位于 sims/marine.py，亦可经总控运行：
  python main.py --sim marine
"""

from __future__ import annotations

import argparse

from sims.marine import run


def main() -> None:
    parser = argparse.ArgumentParser(description="海洋环境扰动 + USBL/DVL/INS EKF 仿真")
    parser.add_argument("--gui", action="store_true", help="使用 GUI 弹窗显示图")
    parser.add_argument(
        "--sim-ini",
        type=str,
        default="",
        help="sim+noise 配置 ini（默认 usbl_fusion/configs/sim_noise.ini）",
    )
    parser.add_argument(
        "--marine-ini",
        type=str,
        default="",
        help="海洋扰动 [marine] 配置（默认 usbl_fusion/configs/marine.ini）",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="outputs/plots_marine",
        help="非 GUI 下保存图片目录",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="marine_sim",
        help="保存图片文件名前缀",
    )
    parser.add_argument("--no-plots", action="store_true", help="不绘图")
    parser.add_argument(
        "--marine-seed",
        type=int,
        default=-1,
        help="海洋扰动 RNG 种子；-1 表示 sim_cfg.seed+20250412",
    )
    args = parser.parse_args()

    from tools.logging_setup import setup_logger

    setup_logger()

    run(
        gui=bool(args.gui),
        sim_ini=args.sim_ini,
        marine_ini=args.marine_ini,
        plots_dir=args.plots_dir,
        plot_prefix=args.plot_prefix,
        no_plots=bool(args.no_plots),
        marine_seed=int(args.marine_seed),
    )


if __name__ == "__main__":
    main()
