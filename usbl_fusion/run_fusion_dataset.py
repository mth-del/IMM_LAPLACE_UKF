#!/usr/bin/env python3
"""
真实 CSV（IMU + DVL + GT）驱动二维 INS+DVL EKF，与纯 INS 推算对比。

用法（在 usbl_fusion 目录下）：
  python run_fusion_dataset.py
  python run_fusion_dataset.py --dataset-dir dataset --out-csv outputs/fused_dataset.csv
  python run_fusion_dataset.py --plot-only            # 只读 fused CSV 重新出图
  python run_fusion_dataset.py --no-plots             # 只写 CSV 不保存图

说明：
- GT 四元数用于 **体轴重力补偿**：g_b = R_wb^T g_w，比力线加速度 a_b = f_b - g_b。
- **水平面与 GT 位置轴对齐**：GT 姿态与 DVL/IMU 机体系往往不一致，不能直接用 R_wb 旋 DVL。
  脚本在 DVL 时刻用 GT 数值速度（位置差分）与 DVL 前向速度做 **2D Kabsch**，估计常值旋转
  R_align（2×2），使 v_map ≈ R_align @ v_dvl_xy；DVL 量测与 IMU 水平加计均经 R_align 映射到
  GT 的 pos_x–pos_y 平面（pN=pos_x, pE=pos_y）。
- 姿态来自 GT 属离线演示；若需纯惯导姿态需另行积分陀螺。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fusion.ekf import EKFConfig, EkfUsblDvlIns
from models.dynamics import propagate_state


G_W = np.array([0.0, 0.0, 9.80665], dtype=float)


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Body -> world (passive rotation): v_w = R @ v_b. 四元数顺序 x,y,z,w (与 CSV 一致)。"""
    x, y, z, w = qx, qy, qz, qw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """球面线性插值，q 为 shape (4,) 的 [qx,qy,qz,qw]。"""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 /= max(np.linalg.norm(q0), 1e-12)
    q1 /= max(np.linalg.norm(q1), 1e-12)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / max(np.linalg.norm(out), 1e-12)
    theta_0 = float(np.arccos(dot))
    sin_t0 = float(np.sin(theta_0))
    s0 = float(np.sin((1.0 - t) * theta_0)) / sin_t0
    s1 = float(np.sin(t * theta_0)) / sin_t0
    out = s0 * q0 + s1 * q1
    return out / max(np.linalg.norm(out), 1e-12)


def _interp_pose(
    t_ns: int,
    gt_t: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """在 GT 时间轴上插值位置（线性）与姿态（SLERP）。t_ns 在范围外则钳位到端点。"""
    t = float(t_ns)
    if t <= float(gt_t[0]):
        return gt_pos[0].copy(), gt_quat[0].copy()
    if t >= float(gt_t[-1]):
        return gt_pos[-1].copy(), gt_quat[-1].copy()
    j = int(np.searchsorted(gt_t, t, side="right"))
    t0, t1 = float(gt_t[j - 1]), float(gt_t[j])
    alpha = (t - t0) / max(t1 - t0, 1e-18)
    p = (1.0 - alpha) * gt_pos[j - 1] + alpha * gt_pos[j]
    q = _slerp(gt_quat[j - 1], gt_quat[j], alpha)
    return p, q


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0].astype(np.int64), data[:, 1:].astype(float)


def _gt_vel_xy_midpoints(
    t_gt_ns: np.ndarray, gt_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT 水平速度样本点取在相邻位姿中点时刻（与 np.diff 一致）。"""
    tg = t_gt_ns.astype(float) * 1e-9
    px = gt_pos[:, 0].astype(float)
    py = gt_pos[:, 1].astype(float)
    dt = np.diff(tg)
    dt = np.maximum(dt, 1e-12)
    vx = np.diff(px) / dt
    vy = np.diff(py) / dt
    t_mid = 0.5 * (tg[1:] + tg[:-1])
    return t_mid, vx.astype(float), vy.astype(float)


def _interp_gt_vel_xy(t_sec: float, t_mid: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(np.interp(t_sec, t_mid, vx)),
            float(np.interp(t_sec, t_mid, vy)),
        ],
        dtype=float,
    )


def _kabsch_align2(v_body: np.ndarray, v_map: np.ndarray) -> np.ndarray:
    """
    最优正交旋转 R（2×2），使 sum ||R v_body_i - v_map_i||^2 最小；det(R)=+1。
    v_body, v_map 均为 (N,2)。
    """
    if v_body.shape[0] < 2:
        return np.eye(2, dtype=float)
    H = np.zeros((2, 2), dtype=float)
    for i in range(v_body.shape[0]):
        H += np.outer(v_map[i], v_body[i])
    u, _, vt = np.linalg.svd(H)
    r = u @ vt
    if float(np.linalg.det(r)) < 0.0:
        u[:, 1] *= -1.0
        r = u @ vt
    return r


def _estimate_R_align_dvl_gt(
    t_dvl_ns: np.ndarray,
    vel_b: np.ndarray,
    t_gt_ns: np.ndarray,
    gt_pos: np.ndarray,
    *,
    v_min: float = 0.02,
) -> np.ndarray:
    """用 DVL 时刻的 GT 水平速度与 DVL xy 速度估计 R_align。"""
    t_mid, vx, vy = _gt_vel_xy_midpoints(t_gt_ns, gt_pos)
    vbs: list[np.ndarray] = []
    vgs: list[np.ndarray] = []
    for i in range(len(t_dvl_ns)):
        ts = float(t_dvl_ns[i]) * 1e-9
        vg = _interp_gt_vel_xy(ts, t_mid, vx, vy)
        vb = np.asarray(vel_b[i, 0:2], dtype=float).reshape(2)
        if float(np.linalg.norm(vg)) >= v_min and float(np.linalg.norm(vb)) >= v_min:
            vgs.append(vg)
            vbs.append(vb)
    if len(vbs) < 4:
        return np.eye(2, dtype=float)
    vb_arr = np.stack(vbs, axis=0)
    vg_arr = np.stack(vgs, axis=0)
    return _kabsch_align2(vb_arr, vg_arr)


def run(
    dataset_dir: Path,
    out_csv: Path,
    acc_noise_std: float,
    dvl_noise_std: float,
    usbl_dummy_std: float,
) -> np.ndarray:
    imu_path = dataset_dir / "imu.csv"
    dvl_path = dataset_dir / "dvl.csv"
    gt_path = dataset_dir / "gt.csv"
    for p in (imu_path, dvl_path, gt_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    t_imu, imu_rest = _load_csv(imu_path)
    acc_b = imu_rest[:, 3:6]

    t_dvl, dvl_rest = _load_csv(dvl_path)
    vel_b_dvl = dvl_rest[:, 0:3]

    t_gt, gt_rest = _load_csv(gt_path)
    gt_pos = gt_rest[:, 0:3]
    gt_quat = gt_rest[:, 3:7]

    t0_ns = int(t_gt[0])
    t1_ns = int(t_gt[-1])
    mask_imu = (t_imu >= t0_ns) & (t_imu <= t1_ns)
    t_imu = t_imu[mask_imu]
    acc_b = acc_b[mask_imu]
    if len(t_imu) < 3:
        raise ValueError("IMU 在 GT 时间范围内的点数过少")

    # 初始状态：在首帧 IMU 时刻插值 GT 位姿；速度用 GT 位置数值微分
    p0, q0 = _interp_pose(int(t_imu[0]), t_gt, gt_pos, gt_quat)
    p1, _ = _interp_pose(int(t_imu[1]), t_gt, gt_pos, gt_quat)
    dt0 = float(t_imu[1] - t_imu[0]) * 1e-9
    v0 = (p1 - p0) / max(dt0, 1e-9)

    x0 = np.array([p0[0], p0[1], v0[0], v0[1]], dtype=float)

    r_align = _estimate_R_align_dvl_gt(t_dvl, vel_b_dvl, t_gt, gt_pos)
    print(
        "R_align (DVL xy -> GT horizontal): det={:.4f}".format(float(np.linalg.det(r_align)))
    )

    mean_dt = float(np.median(np.diff(t_imu.astype(float)))) * 1e-9
    cfg = EKFConfig(
        dt=mean_dt,
        acc_noise_std=acc_noise_std,
        dvl_noise_std=dvl_noise_std,
        usbl_noise_std=usbl_dummy_std,
    )
    ekf = EkfUsblDvlIns(cfg)
    ekf.x = x0.copy()

    x_ins = x0.copy()

    # DVL：按时间排序，在循环中推进索引
    order_dvl = np.argsort(t_dvl)
    t_dvl = t_dvl[order_dvl]
    vel_b_dvl = vel_b_dvl[order_dvl]
    dvl_i = 0
    n_dvl = len(t_dvl)

    rows: list[list[float]] = []

    for k in range(len(t_imu) - 1):
        t_k_ns = int(t_imu[k])
        t_k = t_k_ns
        t_next = int(t_imu[k + 1])
        dt = float(t_next - t_k_ns) * 1e-9
        if dt <= 0:
            continue

        _, q_k = _interp_pose(t_k, t_gt, gt_pos, gt_quat)
        qx, qy, qz, qw = float(q_k[0]), float(q_k[1]), float(q_k[2]), float(q_k[3])
        r_wb = _quat_to_R(qx, qy, qz, qw)
        f_b = acc_b[k]
        g_b = r_wb.T @ G_W
        a_b = f_b - g_b
        a_ne = r_align @ a_b[0:2]

        ekf.predict(a_ne, dt=dt)
        x_ins = propagate_state(x_ins, a_ne, dt)

        while dvl_i < n_dvl and int(t_dvl[dvl_i]) <= t_next:
            if int(t_dvl[dvl_i]) > t_k:
                vb = vel_b_dvl[dvl_i]
                v_meas = r_align @ np.asarray(vb[0:2], dtype=float).reshape(2)
                ekf.update_dvl(v_meas)
            dvl_i += 1

        t_sec = t_next * 1e-9
        rows.append(
            [
                t_sec,
                float(ekf.x[0]),
                float(ekf.x[1]),
                float(ekf.x[2]),
                float(ekf.x[3]),
                float(x_ins[0]),
                float(x_ins[1]),
                float(x_ins[2]),
                float(x_ins[3]),
            ]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "time_sec,pN_fused,pE_fused,vN_fused,vE_fused,"
        "pN_ins,pE_ins,vN_ins,vE_ins\n"
    )
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(header)
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"Wrote {len(rows)} rows -> {out_csv.resolve()}")
    print(
        "(姿态: GT 四元数用于重力方向；DVL/IMU 水平经 Kabsch 与 GT 速度对齐；"
        "ang_vel 未用于姿态积分)"
    )
    return np.asarray(rows, dtype=float)


def plot_fusion_results(
    data: np.ndarray,
    gt_path: Path,
    *,
    plots_dir: Path,
    gui: bool,
    prefix: str = "dataset_fusion",
) -> list[Path]:
    """
    data: (N, 8) 列与 fused_dataset.csv 一致（无表头）:
          time, pN_f, pE_f, vN_f, vE_f, pN_i, pE_i, vN_i, vE_i

    生成 3 个 figure：①轨迹（上全尺度 / 下 GT+EKF 放大）②仅 GT+EKF 放大 ③误差曲线。
    """
    from tools.mpl_setup import configure_matplotlib, save_all_figures

    configure_matplotlib(gui=gui)
    import matplotlib.pyplot as plt

    t = data[:, 0]
    p_nf, p_ef = data[:, 1], data[:, 2]
    p_ni, p_ei = data[:, 5], data[:, 6]

    gt_t, gt_rest = _load_csv(gt_path)
    t_gt_sec = gt_t.astype(float) * 1e-9
    gt_pos = gt_rest[:, 0:3]
    p_gt_n = np.interp(t, t_gt_sec, gt_pos[:, 0])
    p_gt_e = np.interp(t, t_gt_sec, gt_pos[:, 1])

    # --- 轨迹图 1：上全尺度（含 INS 发散），下仅 GT+EKF 局部放大 ---
    fig_traj, (ax_full, ax_zoom) = plt.subplots(
        2, 1, figsize=(7.5, 9.0), height_ratios=[1.2, 1.0]
    )

    ax_full.plot(p_ei, p_ni, "--", color="tab:orange", linewidth=1.6, label="INS only", zorder=1)
    ax_full.plot(p_ef, p_nf, color="tab:green", linewidth=2.0, label="INS+DVL EKF", zorder=2)
    ax_full.plot(
        p_gt_e,
        p_gt_n,
        color="tab:blue",
        linewidth=2.2,
        marker="o",
        markersize=2.5,
        markevery=max(1, len(t) // 40),
        label="GT (interp)",
        zorder=3,
    )
    ax_full.set_xlabel("E / pos_y (m)")
    ax_full.set_ylabel("N / pos_x (m)")
    ax_full.set_aspect("equal", adjustable="box")
    ax_full.grid(True)
    ax_full.legend(loc="best", fontsize=9)
    ax_full.set_title("Full scale: INS drift vs fusion & GT")

    ax_zoom.plot(p_ef, p_nf, color="tab:green", linewidth=2.0, label="INS+DVL EKF", zorder=2)
    ax_zoom.plot(
        p_gt_e,
        p_gt_n,
        color="tab:blue",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        markevery=max(1, len(t) // 50),
        label="GT (interp)",
        zorder=3,
    )
    e_lo = float(min(p_gt_e.min(), p_ef.min()))
    e_hi = float(max(p_gt_e.max(), p_ef.max()))
    n_lo = float(min(p_gt_n.min(), p_nf.min()))
    n_hi = float(max(p_gt_n.max(), p_nf.max()))
    e_span = max(e_hi - e_lo, 1e-6)
    n_span = max(n_hi - n_lo, 1e-6)
    pad = max(0.05, 0.08 * max(e_span, n_span))
    ax_zoom.set_xlim(e_lo - pad, e_hi + pad)
    ax_zoom.set_ylim(n_lo - pad, n_hi + pad)
    ax_zoom.set_aspect("equal", adjustable="box")
    ax_zoom.grid(True)
    ax_zoom.legend(loc="best", fontsize=9)
    ax_zoom.set_xlabel("E / pos_y (m)")
    ax_zoom.set_ylabel("N / pos_x (m)")
    ax_zoom.set_title("Zoom: GT vs EKF (meter scale)")

    fig_traj.tight_layout()

    # --- 单独一张：仅 GT + EKF（与下图同范围，便于报告/幻灯片）---
    fig_ge, ax_ge = plt.subplots(figsize=(6.5, 6.0))
    ax_ge.plot(p_ef, p_nf, color="tab:green", linewidth=2.0, label="INS+DVL EKF")
    ax_ge.plot(
        p_gt_e,
        p_gt_n,
        color="tab:blue",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        markevery=max(1, len(t) // 50),
        label="GT (interp)",
    )
    ax_ge.set_xlim(e_lo - pad, e_hi + pad)
    ax_ge.set_ylim(n_lo - pad, n_hi + pad)
    ax_ge.set_aspect("equal", adjustable="box")
    ax_ge.grid(True)
    ax_ge.legend(loc="best")
    ax_ge.set_xlabel("E / pos_y (m)")
    ax_ge.set_ylabel("N / pos_x (m)")
    ax_ge.set_title("GT vs INS+DVL EKF (zoom only)")
    fig_ge.tight_layout()

    err_ins = np.hypot(p_ni - p_gt_n, p_ei - p_gt_e)
    err_f = np.hypot(p_nf - p_gt_n, p_ef - p_gt_e)
    plt.figure()
    plt.plot(t, err_ins, "--", color="tab:orange", linewidth=1.4, label="INS only")
    plt.plot(t, err_f, color="tab:green", linewidth=1.6, label="INS+DVL EKF")
    plt.xlabel("Time (s)")
    plt.ylabel("Position error vs GT (m)")
    plt.grid(True)
    plt.legend()
    plt.title("Dataset fusion: position error norm")

    if gui:
        plt.show()
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = save_all_figures(plots_dir, prefix=prefix)
    print(f"Saved {len(saved)} figure(s) -> {plots_dir.resolve()}")
    return saved


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="IMU+DVL+GT CSV -> 2D EKF fusion")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(root / "dataset"),
        help="含 imu.csv / dvl.csv / gt.csv 的目录",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(root / "outputs" / "fused_dataset.csv"),
        help="输出轨迹 CSV",
    )
    parser.add_argument("--acc-noise-std", type=float, default=0.05)
    parser.add_argument(
        "--dvl-noise-std",
        type=float,
        default=0.02,
        help="DVL 速度量测噪声标准差 (m/s)；若轨迹抖动可试 0.05~0.1",
    )
    parser.add_argument(
        "--usbl-dummy-std",
        type=float,
        default=1.0,
        help="未使用 USBL 时仅占位 EKFConfig.usbl_noise_std",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成图（默认会保存轨迹与误差 PNG）",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="",
        help="图片保存目录（默认 outputs/plots_dataset）",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="弹窗显示图（否则保存 PNG）",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="只绘图：读取已有 fused CSV，不重新跑融合",
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir) if args.plots_dir else (root / "outputs" / "plots_dataset")

    if args.plot_only:
        fused = Path(args.out_csv)
        if not fused.is_file():
            raise SystemExit(f"--plot-only: 未找到 {fused}")
        data = np.loadtxt(fused, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
    else:
        data = run(
            Path(args.dataset_dir),
            Path(args.out_csv),
            acc_noise_std=args.acc_noise_std,
            dvl_noise_std=args.dvl_noise_std,
            usbl_dummy_std=args.usbl_dummy_std,
        )

    if not args.no_plots:
        gt_path = Path(args.dataset_dir) / "gt.csv"
        if not gt_path.is_file():
            raise SystemExit(f"绘图需要 GT 文件: {gt_path}")
        plot_fusion_results(data, gt_path, plots_dir=plots_dir, gui=bool(args.gui))


if __name__ == "__main__":
    main()
