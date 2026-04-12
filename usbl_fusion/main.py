'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:18:34
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:59:20
FilePath: /usbl_fusion/mian.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# main.py
from __future__ import annotations

import argparse
import numpy as np

from data.simulator import SimConfig, NoiseConfig, generate_truth, simulate_sensors
from fusion.run_pipeline import FusionTrackState, planar_pos_err_norm, run_ekf_tracks
from tools.logging_setup import setup_logger, logger
from tools.config_loader import load_sim_and_noise_config
from tools.mpl_setup import configure_matplotlib, save_all_figures
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="USBL+DVL+INS EKF 融合仿真")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="使用 GUI 弹窗显示图（自动选择 QtAgg/TkAgg；否则回退 Agg）",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="outputs/plots",
        help="非 GUI 模式下保存图片的目录（默认 outputs/plots）",
    )
    args = parser.parse_args()

    setup_logger()
    logger.info("启动仿真与融合流程")

    backend = configure_matplotlib(gui=bool(args.gui))
    logger.info("matplotlib backend = {}", backend)

    # 注意：必须在 configure_matplotlib 之后再 import pyplot / viz.plots
    import matplotlib.pyplot as plt
    from viz.plots import plot_trajectories, plot_errors

    # 1. 配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sim_noise.ini"
    if cfg_path.exists():
        sim_cfg, noise_cfg = load_sim_and_noise_config(cfg_path)
        logger.info("已加载配置文件：{}", str(cfg_path))
    else:
        sim_cfg = SimConfig(dt=0.1, T=600.0)
        noise_cfg = NoiseConfig()
        logger.warning("未找到配置文件：{}，将使用代码默认参数", str(cfg_path))

    # 2. 生成真值轨迹
    t, x_true, a_true = generate_truth(sim_cfg)
    logger.info("真值轨迹生成完成：N={}, dt={}", x_true.shape[0], sim_cfg.dt)

    # 3. 生成传感器数据
    a_meas, dvl_meas, usbl_meas, usbl_mask = simulate_sensors(
        x_true, a_true, sim_cfg, noise_cfg
    )
    logger.info("传感器数据生成完成：DVL@每拍，USBL 有效点数={}", int(np.sum(usbl_mask)))

    # 4–5. INS DR + EKF 融合
    x_ins, x_fused, x_usbl_ins = run_ekf_tracks(
        x_true,
        a_meas,
        dvl_meas,
        usbl_meas,
        usbl_mask,
        sim_cfg,
        noise_cfg,
        track_state=FusionTrackState(),
    )

    e_ins = planar_pos_err_norm(x_true, x_ins)
    e_fused = planar_pos_err_norm(x_true, x_fused)
    e_usbl = planar_pos_err_norm(x_true, x_usbl_ins)
    logger.info(
        "位置误差统计（相对真值，平面 ‖Δp‖₂，单位 m）："
        " INS only  mean={:.6f} max={:.6f} |"
        " INS+DVL+USBL EKF mean={:.6f} max={:.6f} |"
        " INS+USBL EKF mean={:.6f} max={:.6f}",
        float(np.mean(e_ins)),
        float(np.max(e_ins)),
        float(np.mean(e_fused)),
        float(np.max(e_fused)),
        float(np.mean(e_usbl)),
        float(np.max(e_usbl)),
    )

    # 6. 绘图
    logger.info("融合完成，开始绘图")
    plot_trajectories(t, x_true, x_ins, x_fused, x_usbl_ins)
    plot_errors(t, x_true, x_ins, x_fused, x_usbl_ins)
    if args.gui:
        plt.show()
    else:
        saved = save_all_figures(args.plots_dir, prefix="usbl_fusion")
        logger.info("已保存图像：{} 张 -> {}", len(saved), str(Path(args.plots_dir).resolve()))


if __name__ == "__main__":
    main()
