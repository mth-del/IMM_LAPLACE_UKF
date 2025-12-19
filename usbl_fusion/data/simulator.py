'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:15:05
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:15:29
FilePath: /usbl_fusion/data/simlator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
更贴近真实传感器的仿真器（INS / DVL / USBL）

说明（工程 & 近年文献中常见的误差建模思路，非严格对应某一篇论文）：
- 纯白噪声模型在长期仿真中往往过于“理想”，会低估系统误差的相关性与极端情况。
- 实际设备误差通常包含：
  - **偏置(bias)**：随时间缓慢漂移，常用一阶高斯-马尔可夫（Gauss-Markov / OU）或随机游走建模
  - **有色噪声(colored noise)**：短期相关性，常用 GM/OU 过程
  - **比例因子(scale factor)** 与 **安装失准(misalignment)**：对加速度/速度量测产生系统性误差
  - **掉锁/丢包(dropout)**：DVL 底锁丢失、USBL 丢测
  - **离群点(outlier)** / **重尾噪声(heavy-tail)**：多路径、异常回波导致的偶发大误差
  - **时延(latency)**：USBL 低频且可能有处理/通信延迟

本文件在保持原接口输出 (a_meas, dvl_meas, usbl_meas, usbl_mask) 不变的前提下，
引入上述误差项，使仿真更接近实际。
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    dt: float = 0.1        # 采样周期 (s)
    T: float = 600.0       # 仿真时长 (s)
    v0_N: float = 1.0      # 初始速度 (N方向)
    v0_E: float = 0.2      # 初始速度 (E方向)
    seed: int = 1          # 随机种子（保证可复现）

    # ---------------- 真值轨迹（AUV 运动）----------------
    # 分段：加速 -> 匀速直线 -> 圆周/圆弧 -> （剩余时间匀速）
    # 说明：
    # - 状态 x = [pN, pE, vN, vE]
    # - 加速度 a_true = [aN, aE] 是“导航系 N/E”下的真实加速度（不含噪声）
    #
    # Phase 1: 常值加速度（可产生“加速段”）
    accel_duration: float = 60.0
    accel_a_N: float = 0.02
    accel_a_E: float = 0.0

    # Phase 2: 匀速直线（a = 0）
    cruise_duration: float = 120.0

    # Phase 3: 圆周/圆弧（向心加速度 a = ω * J v，|ω|=|v|/R）
    turn_duration: float = 180.0
    turn_radius: float = 200.0
    turn_direction: int = 1  # +1: 逆时针(CCW)  -1: 顺时针(CW)


@dataclass
class NoiseConfig:
    """
    误差模型参数（默认值尽量兼容旧版本的“白噪声 + 常值偏置”风格）

    你可以按设备指标来调参：
    - INS：加速度噪声密度/带宽、偏置稳定性、标定误差
    - DVL：速度精度、底锁概率、异常回波概率
    - USBL：定位精度、丢测率、多路径离群、时延
    """

    # ---------------- INS（加速度）----------------
    acc_noise_std: float = 0.02   # 白噪声标准差 (m/s^2)，每个采样点独立

    # 初始常值偏置（用于给偏置过程一个合理初值，兼容旧参数）
    acc_bias_N: float = 0.003
    acc_bias_E: float = -0.002

    # 偏置漂移：一阶高斯-马尔可夫(OU)过程
    # b[k+1] = exp(-dt/tau) b[k] + sigma_b * sqrt(1-exp(-2dt/tau)) * w
    acc_bias_tau: float = 300.0      # 偏置相关时间(s)，越大越接近随机游走
    acc_bias_sigma: float = 0.0005   # 偏置稳态标准差(m/s^2)

    # 有色噪声（可选）：同样用 OU 建模，模拟带宽受限噪声/滤波残差
    acc_colored_tau: float = 5.0
    acc_colored_sigma: float = 0.005

    # 标定误差：比例因子误差 & 安装失准（2D 下用一个小角度近似）
    acc_scale_sigma: float = 0.001      # 比例因子误差标准差（相对量，无量纲）
    acc_misalignment_std: float = 0.2 * np.pi / 180.0  # 安装失准(弧度)，默认 0.2deg

    # ---------------- DVL（速度）----------------
    dvl_noise_std: float = 0.02   # 速度白噪声标准差 (m/s)
    dvl_bias_sigma: float = 0.01  # 速度偏置稳态标准差 (m/s)
    dvl_bias_tau: float = 200.0   # 偏置相关时间(s)
    dvl_dropout_prob: float = 0.0 # 掉锁/丢测概率（0~1），丢测时输出 NaN
    dvl_outlier_prob: float = 0.0 # 离群点概率（0~1）
    dvl_outlier_scale: float = 0.3  # 离群点尺度(m/s)，越大越“炸”

    # ---------------- USBL（位置）----------------
    usbl_noise_std: float = 0.5   # 位置白噪声标准差 (m)（若未启用相关噪声）
    usbl_period: float = 5.0      # 观测周期 (s)
    usbl_dropout_prob: float = 0.0  # 丢测概率（0~1）
    usbl_latency: float = 0.0       # 时延(s)：量测对应的是过去的真值

    # 相关噪声：如果你希望 N/E 有相关性，可以用相关系数 rho
    usbl_rho: float = 0.0           # [-1,1]

    # 多路径/离群：用重尾分布（拉普拉斯/学生t风格）近似
    usbl_outlier_prob: float = 0.0
    usbl_outlier_scale: float = 5.0  # 离群尺度(m)

    # USBL 噪声模式：
    # - "constant": std = usbl_noise_std（默认，兼容旧行为）
    # - "distance": std = usbl_noise_factor * Δs，其中 Δs 为相邻两次 USBL 真值位置间距
    usbl_noise_mode: str = "constant"
    usbl_noise_factor: float = 0.01
    usbl_noise_min_std: float = 0.0
    usbl_noise_max_std: float = 1e9


def generate_truth(config: SimConfig):
    """生成真实轨迹: x = [pN, pE, vN, vE]^T"""
    dt = config.dt
    N = int(config.T / dt) + 1
    t = np.arange(N) * dt

    x_true = np.zeros((N, 4))
    x_true[0, :] = [0.0, 0.0, config.v0_N, config.v0_E]

    a_true = np.zeros((N, 2))

    # 分段时间点（若总时长不足，会自动截断；若有剩余时间，最终进入匀速段）
    t1 = max(0.0, float(config.accel_duration))
    t2 = t1 + max(0.0, float(config.cruise_duration))
    t3 = t2 + max(0.0, float(config.turn_duration))

    # 圆周段角速度（进入圆周段时根据当前速度模长计算一次，随后保持常值）
    omega_turn: float | None = None

    for k in range(N - 1):
        tk = float(t[k])
        pN, pE, vN, vE = x_true[k, :]

        # Phase 1: 加速段（常值加速度）
        if tk < t1:
            a_true[k, 0] = float(config.accel_a_N)
            a_true[k, 1] = float(config.accel_a_E)

        # Phase 2: 匀速段
        elif tk < t2:
            a_true[k, :] = 0.0

        # Phase 3: 圆周/圆弧段（向心加速度，速度方向持续旋转）
        elif tk < t3:
            v = np.array([vN, vE], dtype=float)
            v_norm = float(np.linalg.norm(v))
            r = float(max(1e-6, config.turn_radius))

            if omega_turn is None:
                # 用进入转弯段的速度确定 ω，保证“圆周半径约等于 turn_radius”
                # 若速度几乎为 0，则回退为 0（不会发生转弯）
                sgn = 1.0 if int(config.turn_direction) >= 0 else -1.0
                omega_turn = sgn * (v_norm / r if v_norm > 1e-6 else 0.0)

            # a = ω * J v，其中 J = [[0,-1],[1,0]]，保证 a ⟂ v（理想情况下速度模长不变）
            a_true[k, 0] = float(omega_turn * (-vE))
            a_true[k, 1] = float(omega_turn * (vN))

        # Phase 4: 剩余时间匀速
        else:
            a_true[k, :] = 0.0

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

    rng = np.random.default_rng(cfg.seed)

    def _ou_step(x: np.ndarray, tau: float, sigma: float) -> np.ndarray:
        """
        一阶高斯-马尔可夫(OU)离散化更新。
        - tau 越大：越“慢变”，趋近随机游走
        - sigma：稳态标准差
        """
        if tau <= 0:
            # 退化：无相关性，直接稳态采样（等价于白噪声）
            return sigma * rng.standard_normal(x.shape)

        alpha = float(np.exp(-dt / tau))
        q = float(np.sqrt(1.0 - np.exp(-2.0 * dt / tau)))
        return alpha * x + sigma * q * rng.standard_normal(x.shape)

    # --- INS（加速度）---
    # 1) 偏置漂移（OU）
    acc_bias = np.array([noise.acc_bias_N, noise.acc_bias_E], dtype=float)
    acc_bias_hist = np.zeros((N, 2))
    for k in range(N):
        acc_bias = _ou_step(acc_bias, noise.acc_bias_tau, noise.acc_bias_sigma)
        acc_bias_hist[k, :] = acc_bias

    # 2) 有色噪声（OU，可理解为带宽受限噪声）
    acc_col = np.zeros(2, dtype=float)
    acc_col_hist = np.zeros((N, 2))
    for k in range(N):
        acc_col = _ou_step(acc_col, noise.acc_colored_tau, noise.acc_colored_sigma)
        acc_col_hist[k, :] = acc_col

    # 3) 白噪声
    acc_white = noise.acc_noise_std * rng.standard_normal((N, 2))

    # 4) 标定误差：比例因子（对每个轴一个相对误差）
    acc_scale = 1.0 + noise.acc_scale_sigma * rng.standard_normal(2)

    # 5) 安装失准：2D 下用一个小旋转角模拟（把真值 a_true 旋转到“传感器轴”）
    mis = float(noise.acc_misalignment_std * rng.standard_normal())
    c, s = float(np.cos(mis)), float(np.sin(mis))
    Rmis = np.array([[c, -s], [s, c]])

    # 组合：a_meas = scale * (Rmis * a_true) + bias + colored + white
    a_meas = (a_true @ Rmis.T) * acc_scale + acc_bias_hist + acc_col_hist + acc_white

    # --- DVL（速度）---
    # DVL 在工程上常见：底锁丢失 -> NaN；偶发异常回波 -> 离群点
    dvl_bias = np.zeros(2, dtype=float)
    dvl_bias_hist = np.zeros((N, 2))
    for k in range(N):
        dvl_bias = _ou_step(dvl_bias, noise.dvl_bias_tau, noise.dvl_bias_sigma)
        dvl_bias_hist[k, :] = dvl_bias

    dvl_white = noise.dvl_noise_std * rng.standard_normal((N, 2))
    dvl_meas = x_true[:, 2:4] + dvl_bias_hist + dvl_white

    # 掉锁/丢测：输出 NaN（上层融合通常会跳过 NaN）
    if noise.dvl_dropout_prob > 0:
        drop = rng.random(N) < noise.dvl_dropout_prob
        dvl_meas[drop, :] = np.nan

    # 离群点：用较大尺度的拉普拉斯噪声（重尾）注入
    if noise.dvl_outlier_prob > 0:
        out = rng.random(N) < noise.dvl_outlier_prob
        if np.any(out):
            dvl_meas[out, :] = dvl_meas[out, :] + rng.laplace(
                loc=0.0, scale=noise.dvl_outlier_scale, size=(int(np.sum(out)), 2)
            )

    # --- USBL ---
    usbl_meas = np.full((N, 2), np.nan)
    usbl_mask = np.zeros(N, dtype=bool)
    usbl_step = max(1, int(noise.usbl_period / dt))

    # USBL 时延：量测对应过去的真值
    latency_steps = int(max(0.0, noise.usbl_latency) / dt)

    # 相关噪声协方差（2D）：Sigma = s^2 [[1, rho],[rho,1]]
    rho = float(np.clip(noise.usbl_rho, -1.0, 1.0))
    base_cov = np.array([[1.0, rho], [rho, 1.0]])

    for k in range(0, N, usbl_step):
        # 丢测
        if noise.usbl_dropout_prob > 0 and (rng.random() < noise.usbl_dropout_prob):
            continue

        k_true = max(0, k - latency_steps)
        usbl_mask[k] = True

        # USBL 噪声：支持“常值 std”与“随位移变化 std”
        # - constant: std = usbl_noise_std
        # - distance: std = usbl_noise_factor * Δs（Δs 为相邻两次 USBL 真值位置间距）
        mode = str(getattr(noise, "usbl_noise_mode", "constant")).strip().lower()
        std = float(noise.usbl_noise_std)
        if mode in ("distance", "dist", "range"):
            factor = float(getattr(noise, "usbl_noise_factor", 0.01))
            min_std = float(getattr(noise, "usbl_noise_min_std", 0.0))
            max_std = float(getattr(noise, "usbl_noise_max_std", 1e9))

            k_prev = max(0, k - usbl_step)
            k_true_prev = max(0, k_prev - latency_steps)
            ds = float(np.linalg.norm(x_true[k_true, 0:2] - x_true[k_true_prev, 0:2]))
            std = float(np.clip(factor * ds, min_std, max_std))

        Sigma = (std ** 2) * base_cov
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            L = std * np.eye(2)

        e = L @ rng.standard_normal(2)
        usbl_meas[k, :] = x_true[k_true, 0:2] + e

        # 多路径离群：重尾（拉普拉斯）扰动
        if noise.usbl_outlier_prob > 0 and (rng.random() < noise.usbl_outlier_prob):
            usbl_meas[k, :] = usbl_meas[k, :] + rng.laplace(
                loc=0.0, scale=noise.usbl_outlier_scale, size=2
            )

    return a_meas, dvl_meas, usbl_meas, usbl_mask
