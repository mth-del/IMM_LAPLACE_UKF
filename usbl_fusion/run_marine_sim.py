#!/usr/bin/env python3
"""
在传感器仿真之后叠加海洋环境扰动（洋流/湍流/USBL 慢偏与尖峰），再跑与 main 相同的 EKF 流水线。

示例：
    python run_marine_sim.py --marine-ini configs/marine.ini
    python run_marine_sim.py --sim-ini configs/sim_noise.ini --marine-ini configs/marine.ini --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.marine_environment import apply_marine_disturbances
from data.simulator import SimConfig, NoiseConfig, generate_truth, simulate_sensors
from fusion.run_pipeline import FusionTrackState, planar_pos_err_norm, run_ekf_tracks
from tools.config_loader import load_marine_config, load_sim_and_noise_config
from tools.logging_setup import logger, setup_logger
from tools.mpl_setup import configure_matplotlib, save_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="海洋环境扰动 + USBL/DVL/INS EKF 仿真")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="使用 GUI 弹窗显示图",
    )
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

    root = Path(__file__).resolve().parent
    sim_path = Path(args.sim_ini) if args.sim_ini else root / "configs" / "sim_noise.ini"
    marine_path = Path(args.marine_ini) if args.marine_ini else root / "configs" / "marine.ini"

    setup_logger()
    logger.info("海洋扰动仿真：sim_ini={} marine_ini={}", str(sim_path), str(marine_path))

    backend = configure_matplotlib(gui=bool(args.gui))
    logger.info("matplotlib backend = {}", backend)

    if not args.no_plots:
        import matplotlib.pyplot as plt
        from viz.plots import plot_errors, plot_trajectories
    else:
        plt = None  # type: ignore[assignment]

    if sim_path.exists():
        sim_cfg, noise_cfg = load_sim_and_noise_config(sim_path)
        logger.info("已加载 sim/noise：{}", str(sim_path))
    else:
        sim_cfg = SimConfig(dt=0.1, T=600.0)
        noise_cfg = NoiseConfig()
        logger.warning("未找到 sim 配置 {}，使用默认", str(sim_path))

    marine_cfg = load_marine_config(marine_path)
    logger.info(
        "海洋扰动参数：mean_N={} mean_E={} ou_sigma={} dvl_turb={} usbl_ou_sigma={} burst_p={}",
        marine_cfg.current_mean_N,
        marine_cfg.current_mean_E,
        marine_cfg.current_ou_sigma,
        marine_cfg.dvl_turbulence_std,
        marine_cfg.usbl_bias_ou_sigma,
        marine_cfg.usbl_burst_prob,
    )

    t, x_true, a_true = generate_truth(sim_cfg)
    a_meas, dvl_meas, usbl_meas, usbl_mask = simulate_sensors(
        x_true, a_true, sim_cfg, noise_cfg
    )

    mseed = None if int(args.marine_seed) < 0 else int(args.marine_seed)
    a2, d2, u2 = apply_marine_disturbances(
        t,
        a_meas,
        dvl_meas,
        usbl_meas,
        usbl_mask,
        sim_cfg,
        marine_cfg,
        seed=mseed,
    )

    x_ins, x_fused, x_usbl_ins = run_ekf_tracks(
        x_true,
        a2,
        d2,
        u2,
        usbl_mask,
        sim_cfg,
        noise_cfg,
        track_state=FusionTrackState(),
    )

    e_ins = planar_pos_err_norm(x_true, x_ins)
    e_fused = planar_pos_err_norm(x_true, x_fused)
    e_usbl = planar_pos_err_norm(x_true, x_usbl_ins)
    logger.info(
        "位置误差（平面 ‖Δp‖₂ m）：INS mean={:.6f} max={:.6f} | "
        "INS+DVL+USBL EKF mean={:.6f} max={:.6f} | INS+USBL EKF mean={:.6f} max={:.6f}",
        float(np.mean(e_ins)),
        float(np.max(e_ins)),
        float(np.mean(e_fused)),
        float(np.max(e_fused)),
        float(np.mean(e_usbl)),
        float(np.max(e_usbl)),
    )

    if not args.no_plots and plt is not None:
        plot_trajectories(t, x_true, x_ins, x_fused, x_usbl_ins)
        plot_errors(t, x_true, x_ins, x_fused, x_usbl_ins)
        if args.gui:
            plt.show()
        else:
            saved = save_all_figures(args.plots_dir, prefix=args.plot_prefix)
            logger.info("已保存图像 {} 张 -> {}", len(saved), str(Path(args.plots_dir).resolve()))


if __name__ == "__main__":
    main()
