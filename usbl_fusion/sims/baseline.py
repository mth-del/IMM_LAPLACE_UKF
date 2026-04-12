"""合成真值 + 高斯传感器仿真 + EKF（原 main 主体）。"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from data.simulator import NoiseConfig, SimConfig, generate_truth, simulate_sensors
from fusion.run_pipeline import FusionTrackState, planar_pos_err_norm, run_ekf_tracks
from tools.config_loader import load_sim_and_noise_config
from tools.logging_setup import logger
from tools.mpl_setup import configure_matplotlib, save_all_figures


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def run(
    *,
    gui: bool = False,
    plots_dir: str = "outputs/plots",
    sim_ini: str = "",
    plot_prefix: str = "usbl_fusion",
    no_plots: bool = False,
) -> None:
    backend = configure_matplotlib(gui=bool(gui))
    logger.info("matplotlib backend = {}", backend)

    plt = None
    if not no_plots:
        import matplotlib.pyplot as plt
        from viz.plots import plot_errors, plot_trajectories

    cfg_path = Path(sim_ini) if sim_ini else _root() / "configs" / "sim_noise.ini"
    if cfg_path.exists():
        sim_cfg, noise_cfg = load_sim_and_noise_config(cfg_path)
        logger.info("已加载配置文件：{}", str(cfg_path))
    else:
        sim_cfg = SimConfig(dt=0.1, T=600.0)
        noise_cfg = NoiseConfig()
        logger.warning("未找到配置文件：{}，将使用代码默认参数", str(cfg_path))

    t, x_true, a_true = generate_truth(sim_cfg)
    logger.info("真值轨迹生成完成：N={}, dt={}", x_true.shape[0], sim_cfg.dt)

    a_meas, dvl_meas, usbl_meas, usbl_mask = simulate_sensors(
        x_true, a_true, sim_cfg, noise_cfg
    )
    logger.info("传感器数据生成完成：DVL@每拍，USBL 有效点数={}", int(np.sum(usbl_mask)))

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

    if plt is not None:
        logger.info("融合完成，开始绘图")
        plot_trajectories(t, x_true, x_ins, x_fused, x_usbl_ins)
        plot_errors(t, x_true, x_ins, x_fused, x_usbl_ins)
        if gui:
            plt.show()
        else:
            saved = save_all_figures(plots_dir, prefix=plot_prefix)
            logger.info("已保存图像：{} 张 -> {}", len(saved), str(Path(plots_dir).resolve()))


def run_from_ns(ns: Namespace) -> None:
    plots = (getattr(ns, "plots_dir", "") or "").strip() or "outputs/plots"
    sim_ini = (getattr(ns, "sim_ini", "") or "").strip()
    run(
        gui=bool(ns.gui),
        plots_dir=plots,
        sim_ini=sim_ini,
        plot_prefix="usbl_fusion",
        no_plots=bool(getattr(ns, "no_plots", False)),
    )
