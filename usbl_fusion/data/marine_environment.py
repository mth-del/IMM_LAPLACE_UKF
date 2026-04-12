"""
海洋环境扰动（现象学叠加层）

在 `simulate_sensors` 生成的高斯量测之上，再叠加与洋流、湍流、声速/声线类
效应等价的慢变偏置与附加噪声，用于区分「理想高斯仿真」与「更复杂水下环境」。

说明（建模取舍）：
- 本工程状态为平面 [pN,pE,vN,vE]，不做完整流体动力学。
- 「洋流」用 **DVL 速度上的慢变 OU 偏置 + 可选常值均值** 近似（表观对底速度偏差）。
- 「湍流/剪切」用 **DVL 上附加白噪声**。
- 「声速剖面/层化」用 **USBL 位置上的慢变 OU 偏置** 近似水平漂移。
- 不对 INS 加计默认加项（可后续扩展）；需要时可在外部对 a_meas 再叠加。

用法：
    from data.marine_environment import MarineEnvConfig, apply_marine_disturbances
    a2, d2, u2 = apply_marine_disturbances(
        t, a_meas, dvl_meas, usbl_meas, usbl_mask, sim_cfg, marine_cfg
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.simulator import SimConfig


@dataclass
class MarineEnvConfig:
    """全部为 0 时表示关闭该项（与纯高斯仿真一致）。"""

    # --- 等效洋流：在 DVL 速度上叠加慢变偏置（m/s）---
    current_mean_N: float = 0.0
    current_mean_E: float = 0.0
    current_ou_tau: float = 120.0
    # OU 分量稳态标准差（围绕 0 的慢变波动，与 current_mean_* 相加）
    current_ou_sigma: float = 0.0

    # --- DVL 湍流/剪切：附加白噪声标准差 (m/s) ---
    dvl_turbulence_std: float = 0.0

    # --- USBL：水平位置慢变偏置（OU，单位 m；稳态 RMS = usbl_bias_ou_sigma）---
    usbl_bias_ou_tau: float = 180.0
    usbl_bias_ou_sigma: float = 0.0

    # --- USBL：多路径/散射引起的偶发尖峰（拉普拉斯，scale 同 np.random.laplace）---
    usbl_burst_prob: float = 0.0
    usbl_burst_scale: float = 0.5


def _ou_step(
    x: np.ndarray,
    dt: float,
    tau: float,
    steady_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """离散 OU：稳态标准差为 steady_std。"""
    if tau <= 0.0:
        return float(steady_std) * rng.standard_normal(x.shape)
    alpha = float(np.exp(-dt / tau))
    q = float(np.sqrt(max(0.0, 1.0 - alpha * alpha)))
    return alpha * x + float(steady_std) * q * rng.standard_normal(x.shape)


def apply_marine_disturbances(
    t: np.ndarray,
    a_meas: np.ndarray,
    dvl_meas: np.ndarray,
    usbl_meas: np.ndarray,
    usbl_mask: np.ndarray,
    sim_cfg: SimConfig,
    marine: MarineEnvConfig,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回新的 (a_meas, dvl_meas, usbl_meas) 副本；不在原地修改输入。
    """
    dt = float(sim_cfg.dt)
    rng = np.random.default_rng(int(seed) if seed is not None else int(sim_cfg.seed) + 20250412)

    a_out = np.asarray(a_meas, dtype=float).copy()
    dvl_out = np.asarray(dvl_meas, dtype=float).copy()
    usbl_out = np.asarray(usbl_meas, dtype=float).copy()

    n = dvl_out.shape[0]
    mask = np.asarray(usbl_mask, dtype=bool).reshape(-1)

    cur_ou = np.zeros(2, dtype=float)
    mean = np.array([marine.current_mean_N, marine.current_mean_E], dtype=float)

    for k in range(n):
        if marine.current_ou_sigma > 0.0:
            cur_ou = _ou_step(cur_ou, dt, marine.current_ou_tau, marine.current_ou_sigma, rng)
        bias = mean + cur_ou
        row = dvl_out[k]
        if not np.any(np.isnan(row)):
            dvl_out[k, 0:2] = row[0:2] + bias
            if marine.dvl_turbulence_std > 0.0:
                dvl_out[k, 0:2] = dvl_out[k, 0:2] + marine.dvl_turbulence_std * rng.standard_normal(2)

    b_usbl = np.zeros(2, dtype=float)
    for k in range(n):
        if marine.usbl_bias_ou_sigma > 0.0:
            b_usbl = _ou_step(b_usbl, dt, marine.usbl_bias_ou_tau, marine.usbl_bias_ou_sigma, rng)
        if k < len(mask) and mask[k] and not np.any(np.isnan(usbl_out[k, :])):
            usbl_out[k, 0:2] = usbl_out[k, 0:2] + b_usbl
            if marine.usbl_burst_prob > 0.0 and rng.random() < marine.usbl_burst_prob:
                usbl_out[k, 0:2] = usbl_out[k, 0:2] + rng.laplace(
                    0.0, marine.usbl_burst_scale, size=2
                )

    _ = t  # 预留按时间调参（潮汐等）
    return a_out, dvl_out, usbl_out
