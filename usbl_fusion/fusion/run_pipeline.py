"""
INS 死 reckoning + USBL/DVL EKF 融合流水线（与 main 共用）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.simulator import NoiseConfig, SimConfig
from fusion.ekf import EKFConfig, EkfUsblDvlIns
from models.dynamics import propagate_state


@dataclass
class FusionTrackState:
    """动态 USBL R（distance 模式）依赖的跨步状态，避免用模块级可变全局量。"""

    last_usbl_k_true: int | None = None


def run_ekf_tracks(
    x_true: np.ndarray,
    a_meas: np.ndarray,
    dvl_meas: np.ndarray,
    usbl_meas: np.ndarray,
    usbl_mask: np.ndarray,
    sim_cfg: SimConfig,
    noise_cfg: NoiseConfig,
    *,
    track_state: FusionTrackState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (x_ins, x_fused, x_usbl_ins)，形状与 x_true 相同。
    """
    if track_state is None:
        track_state = FusionTrackState()

    N = int(x_true.shape[0])

    x_ins = np.zeros_like(x_true)
    x_ins[0, :] = x_true[0, :]
    for k in range(N - 1):
        x_ins[k + 1, :] = propagate_state(x_ins[k, :], a_meas[k, :], sim_cfg.dt)

    ekf_cfg = EKFConfig(
        dt=sim_cfg.dt,
        acc_noise_std=noise_cfg.acc_noise_std,
        dvl_noise_std=noise_cfg.dvl_noise_std,
        usbl_noise_std=noise_cfg.usbl_noise_std,
    )
    ekf = EkfUsblDvlIns(ekf_cfg)
    ekf_usbl_ins = EkfUsblDvlIns(ekf_cfg)

    x_fused = np.zeros_like(x_true)
    x_fused[0, :] = ekf.x
    x_usbl_ins = np.zeros_like(x_true)
    x_usbl_ins[0, :] = ekf_usbl_ins.x

    mask = np.asarray(usbl_mask, dtype=bool).reshape(-1)

    for k in range(N - 1):
        ekf.predict(a_meas[k, :])
        ekf_usbl_ins.predict(a_meas[k, :])

        ekf.update_dvl(dvl_meas[k, :])

        if mask[k] and not np.any(np.isnan(usbl_meas[k, :])):
            mode = str(getattr(noise_cfg, "usbl_noise_mode", "constant")).strip().lower()
            if mode in ("distance", "dist", "range"):
                latency_steps = int(max(0.0, float(noise_cfg.usbl_latency)) / float(sim_cfg.dt))
                k_true = max(0, k - latency_steps)
                last_k_true = track_state.last_usbl_k_true

                if last_k_true is None:
                    std = float(noise_cfg.usbl_noise_std)
                else:
                    ds = float(
                        np.linalg.norm(x_true[k_true, 0:2] - x_true[int(last_k_true), 0:2])
                    )
                    factor = float(getattr(noise_cfg, "usbl_noise_factor", 0.01))
                    min_std = float(getattr(noise_cfg, "usbl_noise_min_std", 0.0))
                    max_std = float(getattr(noise_cfg, "usbl_noise_max_std", 1e9))
                    std = float(np.clip(factor * ds, min_std, max_std))

                rho = float(np.clip(float(getattr(noise_cfg, "usbl_rho", 0.0)), -1.0, 1.0))
                r_mat = (std**2) * np.array([[1.0, rho], [rho, 1.0]])

                ekf.update_usbl(usbl_meas[k, :], R=r_mat)
                ekf_usbl_ins.update_usbl(usbl_meas[k, :], R=r_mat)
                track_state.last_usbl_k_true = k_true
            else:
                ekf.update_usbl(usbl_meas[k, :])
                ekf_usbl_ins.update_usbl(usbl_meas[k, :])

        x_fused[k + 1, :] = ekf.x
        x_usbl_ins[k + 1, :] = ekf_usbl_ins.x

    return x_ins, x_fused, x_usbl_ins


def planar_pos_err_norm(x_true: np.ndarray, x_est: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x_est[:, 0:2] - x_true[:, 0:2], axis=1)
