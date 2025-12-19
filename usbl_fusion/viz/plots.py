'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:17:44
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:17:51
FilePath: /usbl_fusion/viz/plots.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# viz/plots.py
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(t, x_true, x_ins, x_fused):
    plt.figure()
    plt.plot(x_true[:, 1], x_true[:, 0], label="truth")
    plt.plot(x_ins[:, 1], x_ins[:, 0], "--", label="INS only")
    plt.plot(x_fused[:, 1], x_fused[:, 0], label="INS+DVL+USBL EKF")
    plt.xlabel("E (m)")
    plt.ylabel("N (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Horizontal trajectory")


def plot_errors(t, x_true, x_ins, x_fused):
    err_ins = x_ins[:, 0:2] - x_true[:, 0:2]
    err_fused = x_fused[:, 0:2] - x_true[:, 0:2]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, np.linalg.norm(err_ins, axis=1), "--", label="INS only")
    plt.plot(t, np.linalg.norm(err_fused, axis=1), label="Fusion")
    plt.ylabel("Position error (m)")
    plt.grid(True)
    plt.legend()
    plt.title("Position error norm vs time")

    plt.subplot(2, 1, 2)
    plt.plot(t, err_fused[:, 0], label="N error")  # 融合结果的北向位置误差：pN_est - pN_true
    plt.plot(t, err_fused[:, 1], label="E error")  # 融合结果的东向位置误差：pE_est - pE_true
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.grid(True)
    plt.legend()
