'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:15:05
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:15:29
FilePath: /usbl_fusion/data/simlator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# data/simulator.py
from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    dt: float = 0.1        # 采样周期 (s)
    T: float = 600.0       # 仿真时长 (s)
    v0_N: float = 1.0      # 初始速度 (N方向)
    v0_E: float = 0.2      # 初始速度 (E方向)


@dataclass
class NoiseConfig:
    acc_bias_N: float = 0.003
    acc_bias_E: float = -0.002
    acc_noise_std: float = 0.02   # INS 加速度噪声 (m/s^2)
    dvl_noise_std: float = 0.02   # DVL 速度噪声 (m/s)
    usbl_noise_std: float = 0.5   # USBL 位置噪声 (m)
    usbl_period: float = 5.0      # USBL 观测周期 (s)


def generate_truth(config: SimConfig):
    """生成真实轨迹: x = [pN, pE, vN, vE]^T"""
    dt = config.dt
    N = int(config.T / dt) + 1
    t = np.arange(N) * dt

    x_true = np.zeros((N, 4))
    x_true[0, :] = [0.0, 0.0, config.v0_N, config.v0_E]

    a_true = np.zeros((N, 2))

    for k in range(N - 1):
        # 自己造一个缓慢变化的加速度轨迹
        a_N = 0.01 * np.sin(0.01 * t[k])
        a_E = 0.01 * np.cos(0.005 * t[k])
        a_true[k, :] = [a_N, a_E]

        pN, pE, vN, vE = x_true[k, :]
        v_next = np.array([vN, vE]) + a_true[k, :] * dt
        p_next = np.array([pN, pE]) + np.array([vN, vE]) * dt + 0.5 * a_true[k, :] * dt**2
        x_true[k + 1, :] = [p_next[0], p_next[1], v_next[0], v_next[1]]

    a_true[-1, :] = a_true[-2, :]
    return t, x_true, a_true


def simulate_sensors(x_true: np.ndarray, a_true: np.ndarray,
                     cfg: SimConfig, noise: NoiseConfig):
    """
    根据真值轨迹生成 INS / DVL / USBL 观测
    返回:
        a_meas:   INS 加速度 (N×2)
        dvl_meas: DVL 速度 (N×2)
        usbl_meas: USBL 位置 (N×2, 没测到的地方用 np.nan)
        usbl_mask: bool 数组, True 表示这一拍有 USBL 观测
    """
    dt = cfg.dt
    N = x_true.shape[0]

    # --- INS ---
    rng = np.random.default_rng(1)
    acc_bias = np.array([noise.acc_bias_N, noise.acc_bias_E])
    a_meas = a_true + acc_bias + noise.acc_noise_std * rng.standard_normal((N, 2))

    # --- DVL ---
    dvl_meas = x_true[:, 2:4] + noise.dvl_noise_std * rng.standard_normal((N, 2))

    # --- USBL ---
    usbl_meas = np.full((N, 2), np.nan)
    usbl_mask = np.zeros(N, dtype=bool)
    usbl_step = max(1, int(noise.usbl_period / dt))

    for k in range(0, N, usbl_step):
        usbl_mask[k] = True
        usbl_meas[k, :] = x_true[k, 0:2] + noise.usbl_noise_std * \
            rng.standard_normal(2)

    return a_meas, dvl_meas, usbl_meas, usbl_mask
