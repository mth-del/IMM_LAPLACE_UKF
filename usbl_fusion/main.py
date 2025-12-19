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
from models.dynamics import propagate_state
from fusion.ekf import EKFConfig, EkfUsblDvlIns
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
    # 记录“上一次 USBL 对应的真值时刻”（用于动态R），避免放全局变量污染模块命名空间
    main._last_usbl_k_true = None  # type: ignore[attr-defined]

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

    N = x_true.shape[0]

    # 4. 纯 INS 死 Reckoning（用于对比）
    x_ins = np.zeros_like(x_true)
    x_ins[0, :] = x_true[0, :]  # 假设初始对准正确
    for k in range(N - 1):
        x_ins[k + 1, :] = propagate_state(
            x_ins[k, :], a_meas[k, :], sim_cfg.dt
        )

    # 5. EKF 融合
    ekf_cfg = EKFConfig(
        dt=sim_cfg.dt,
        acc_noise_std=noise_cfg.acc_noise_std,
        dvl_noise_std=noise_cfg.dvl_noise_std,
        usbl_noise_std=noise_cfg.usbl_noise_std,
    )
    ekf = EkfUsblDvlIns(ekf_cfg)
    ekf_usbl_ins = EkfUsblDvlIns(ekf_cfg)  # 仅用 INS 预测 + USBL 更新（不使用 DVL）

    x_fused = np.zeros_like(x_true)
    x_fused[0, :] = ekf.x

    x_usbl_ins = np.zeros_like(x_true)
    x_usbl_ins[0, :] = ekf_usbl_ins.x

    for k in range(N - 1):
        # 预测
        ekf.predict(a_meas[k, :])
        ekf_usbl_ins.predict(a_meas[k, :])

        # DVL 更新（每拍）
        ekf.update_dvl(dvl_meas[k, :])

        # USBL 更新（低频，有就更新）
        if usbl_mask[k] and not np.any(np.isnan(usbl_meas[k, :])):
            # 若选择 USBL 噪声随位移变化（sigma = factor * Δs），这里同步使用动态 R
            mode = str(getattr(noise_cfg, "usbl_noise_mode", "constant")).strip().lower()
            if mode in ("distance", "dist", "range"):
                latency_steps = int(max(0.0, float(noise_cfg.usbl_latency)) / float(sim_cfg.dt))
                k_true = max(0, k - latency_steps)

                # 找到上一次有效 USBL 的真值时刻
                if not hasattr(main, "_last_usbl_k_true"):
                    main._last_usbl_k_true = None  # type: ignore[attr-defined]
                last_k_true = getattr(main, "_last_usbl_k_true")  # type: ignore[attr-defined]

                if last_k_true is None:
                    std = float(noise_cfg.usbl_noise_std)
                else:
                    ds = float(np.linalg.norm(x_true[k_true, 0:2] - x_true[int(last_k_true), 0:2]))
                    factor = float(getattr(noise_cfg, "usbl_noise_factor", 0.01))
                    min_std = float(getattr(noise_cfg, "usbl_noise_min_std", 0.0))
                    max_std = float(getattr(noise_cfg, "usbl_noise_max_std", 1e9))
                    std = float(np.clip(factor * ds, min_std, max_std))

                rho = float(np.clip(float(getattr(noise_cfg, "usbl_rho", 0.0)), -1.0, 1.0))
                R = (std ** 2) * np.array([[1.0, rho], [rho, 1.0]])

                ekf.update_usbl(usbl_meas[k, :], R=R)
                ekf_usbl_ins.update_usbl(usbl_meas[k, :], R=R)
                main._last_usbl_k_true = k_true  # type: ignore[attr-defined]
            else:
                ekf.update_usbl(usbl_meas[k, :])
                ekf_usbl_ins.update_usbl(usbl_meas[k, :])

        x_fused[k + 1, :] = ekf.x
        x_usbl_ins[k + 1, :] = ekf_usbl_ins.x

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
