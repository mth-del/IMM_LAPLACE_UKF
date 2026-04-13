"""
海洋环境扰动（现象学叠加层）

在 `simulate_sensors` 生成的高斯量测之上，再叠加与洋流、湍流、声速/声线类
效应等价的慢变偏置与附加噪声。

**USBL 偶发尖峰 `usbl_burst_dist`（与仅用高斯噪声的差别）**：
- **gaussian**（默认）：触发时在 N/E 上各加独立 **N(0, b²)**，`b = usbl_burst_scale` 为标准差 (m)。
  大幅误差概率随幅值按 **指数平方衰减**（尾较薄），与标准 EKF 高斯量测假设一致。
- **laplace**：触发时各轴为 **Laplace(0, b)**，`b` 为 numpy 的 scale 参数 (m)，方差 **Var = 2b²**。
  相对同尺度 b，**|误差| 大时概率更高（重尾）**，更粗拟多路径尖峰；与线性高斯 EKF 名义假设偏差更大，
  常表现为 USBL 更新时位置/融合轨出现 **更偶发、更大的跳变**。

洋流 OU、DVL 湍流、USBL OU 慢偏的随机驱动均为高斯。

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
from tools.logging_setup import logger as _marine_logger


@dataclass
class MarineEnvConfig:
    """全部为 0 时表示关闭该项（与纯高斯仿真一致）。"""

    # --- 等效洋流：在 DVL 速度上叠加慢变偏置（m/s）---
    current_mean_N: float = 0.0
    current_mean_E: float = 0.0
    current_ou_tau: float = 120.0
    current_ou_sigma: float = 0.0

    # --- DVL 湍流/剪切：附加白噪声标准差 (m/s) ---
    dvl_turbulence_std: float = 0.0

    # --- USBL：水平位置慢变偏置（OU，单位 m）---
    usbl_bias_ou_tau: float = 180.0
    usbl_bias_ou_sigma: float = 0.0

    # --- USBL：偶发尖峰（Bernoulli 触发；分布见 usbl_burst_dist）---
    usbl_burst_prob: float = 0.0
    # gaussian：各轴标准差 (m)；laplace：numpy Laplace 的 scale (m)
    usbl_burst_scale: float = 0.5
    # gaussian | laplace | gs | normal（后二者同 gaussian）；大小写不敏感
    usbl_burst_dist: str = "gaussian"


def _ou_step(
    x: np.ndarray,
    dt: float,
    tau: float,
    steady_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if tau <= 0.0:
        return float(steady_std) * rng.standard_normal(x.shape)
    alpha = float(np.exp(-dt / tau))
    q = float(np.sqrt(max(0.0, 1.0 - alpha * alpha)))
    return alpha * x + float(steady_std) * q * rng.standard_normal(x.shape)


def _burst_noise_xy(rng: np.random.Generator, dist: str, scale: float) -> np.ndarray:
    d = dist.strip().lower()
    b = float(scale)
    if b <= 0.0:
        return np.zeros(2, dtype=float)
    if d in ("laplace", "lap", "l"):
        return rng.laplace(0.0, b, size=2)
    # gaussian / gs / normal / g / "" / unknown -> 高斯
    return b * rng.standard_normal(2)


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

    burst_dist = str(getattr(marine, "usbl_burst_dist", "gaussian") or "gaussian")

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
    n_usbl_valid = 0
    n_burst = 0
    for k in range(n):
        if marine.usbl_bias_ou_sigma > 0.0:
            b_usbl = _ou_step(b_usbl, dt, marine.usbl_bias_ou_tau, marine.usbl_bias_ou_sigma, rng)
        if k < len(mask) and mask[k] and not np.any(np.isnan(usbl_out[k, :])):
            n_usbl_valid += 1
            usbl_out[k, 0:2] = usbl_out[k, 0:2] + b_usbl
            if marine.usbl_burst_prob > 0.0 and rng.random() < marine.usbl_burst_prob:
                n_burst += 1
                usbl_out[k, 0:2] = usbl_out[k, 0:2] + _burst_noise_xy(
                    rng, burst_dist, float(marine.usbl_burst_scale)
                )

    if marine.usbl_burst_prob > 0.0:
        _marine_logger.info(
            "marine USBL 尖峰统计：有效 USBL 步数={}，Bernoulli 触发次数={}，dist={}，p={}，scale={}",
            n_usbl_valid,
            n_burst,
            burst_dist,
            marine.usbl_burst_prob,
            float(marine.usbl_burst_scale),
        )

    _ = t
    return a_out, dvl_out, usbl_out
