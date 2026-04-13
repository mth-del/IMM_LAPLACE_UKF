"""海洋环境扰动层 + 与 baseline 相同的 EKF 流水线（原 run_marine_sim 主体）。"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from data.marine_environment import apply_marine_disturbances
from data.simulator import NoiseConfig, SimConfig, generate_truth, simulate_sensors
from fusion.run_pipeline import FusionTrackState, planar_pos_err_norm, run_ekf_tracks
from tools.config_loader import load_marine_config, load_sim_and_noise_config
from tools.logging_setup import logger
from tools.mpl_setup import configure_matplotlib, save_all_figures


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def run(
    *,
    gui: bool = False,
    sim_ini: str = "",
    marine_ini: str = "",
    plots_dir: str = "outputs/plots_marine",
    plot_prefix: str = "marine_sim",
    no_plots: bool = False,
    marine_seed: int = -1,
) -> None:
    root = _root()
    sim_path = Path(sim_ini) if sim_ini else root / "configs" / "sim_noise.ini"
    marine_path = Path(marine_ini) if marine_ini else root / "configs" / "marine.ini"

    logger.info("海洋扰动仿真：sim_ini={} marine_ini={}", str(sim_path), str(marine_path))

    backend = configure_matplotlib(gui=bool(gui))
    logger.info("matplotlib backend = {}", backend)

    plt = None
    if not no_plots:
        import matplotlib.pyplot as plt
        from viz.plots import plot_errors, plot_trajectories

    if sim_path.exists():
        sim_cfg, noise_cfg = load_sim_and_noise_config(sim_path)
        logger.info("已加载 sim/noise：{}", str(sim_path))
    else:
        sim_cfg = SimConfig(dt=0.1, T=600.0)
        noise_cfg = NoiseConfig()
        logger.warning("未找到 sim 配置 {}，使用默认", str(sim_path))

    marine_cfg = load_marine_config(marine_path)
    logger.info(
        "海洋扰动参数：mean_N={} mean_E={} ou_sigma={} dvl_turb={} usbl_ou_sigma={} "
        "burst_p={} burst_scale={} burst_dist={}",
        marine_cfg.current_mean_N,
        marine_cfg.current_mean_E,
        marine_cfg.current_ou_sigma,
        marine_cfg.dvl_turbulence_std,
        marine_cfg.usbl_bias_ou_sigma,
        marine_cfg.usbl_burst_prob,
        marine_cfg.usbl_burst_scale,
        getattr(marine_cfg, "usbl_burst_dist", "gaussian"),
    )

    t, x_true, a_true = generate_truth(sim_cfg)
    a_meas, dvl_meas, usbl_meas, usbl_mask = simulate_sensors(
        x_true, a_true, sim_cfg, noise_cfg
    )

    mseed = None if int(marine_seed) < 0 else int(marine_seed)
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

    if plt is not None:
        plot_trajectories(t, x_true, x_ins, x_fused, x_usbl_ins)
        plot_errors(t, x_true, x_ins, x_fused, x_usbl_ins)
        if gui:
            plt.show()
        else:
            saved = save_all_figures(plots_dir, prefix=plot_prefix)
            logger.info("已保存图像 {} 张 -> {}", len(saved), str(Path(plots_dir).resolve()))


def run_from_ns(ns: Namespace) -> None:
    plots = (getattr(ns, "plots_dir", "") or "").strip() or "outputs/plots_marine"
    sim_ini = (getattr(ns, "sim_ini", "") or "").strip()
    marine_ini = (getattr(ns, "marine_ini", "") or "").strip()
    run(
        gui=bool(ns.gui),
        sim_ini=sim_ini,
        marine_ini=marine_ini,
        plots_dir=plots,
        plot_prefix=str(getattr(ns, "plot_prefix", "marine_sim") or "marine_sim"),
        no_plots=bool(getattr(ns, "no_plots", False)),
        marine_seed=int(getattr(ns, "marine_seed", -1)),
    )
