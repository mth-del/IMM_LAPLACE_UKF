'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:18:34
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:59:20
FilePath: /usbl_fusion/mian.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# main.py
import numpy as np

from data.simulator import SimConfig, NoiseConfig, generate_truth, simulate_sensors
from models.dynamics import propagate_state
from fusion.ekf import EKFConfig, EkfUsblDvlIns
from viz.plots import plot_trajectories, plot_errors
import matplotlib.pyplot as plt


def main():
    # 1. 配置
    sim_cfg = SimConfig(dt=0.1, T=600.0)
    noise_cfg = NoiseConfig()

    # 2. 生成真值轨迹
    t, x_true, a_true = generate_truth(sim_cfg)

    # 3. 生成传感器数据
    a_meas, dvl_meas, usbl_meas, usbl_mask = simulate_sensors(
        x_true, a_true, sim_cfg, noise_cfg
    )

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

    x_fused = np.zeros_like(x_true)
    x_fused[0, :] = ekf.x

    for k in range(N - 1):
        # 预测
        ekf.predict(a_meas[k, :])

        # DVL 更新（每拍）
        ekf.update_dvl(dvl_meas[k, :])

        # USBL 更新（低频，有就更新）
        if usbl_mask[k] and not np.any(np.isnan(usbl_meas[k, :])):
            ekf.update_usbl(usbl_meas[k, :])

        x_fused[k + 1, :] = ekf.x

    # 6. 绘图
    plot_trajectories(t, x_true, x_ins, x_fused)
    plot_errors(t, x_true, x_ins, x_fused)
    plt.show()


if __name__ == "__main__":
    main()
