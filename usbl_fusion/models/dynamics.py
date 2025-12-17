# models/dynamics.py
import numpy as np


def get_F_B(dt: float):
    """状态转移矩阵 F 和输入矩阵 B, 状态 x=[pN,pE,vN,vE]^T """
    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    B = np.array([
        [0.5 * dt ** 2, 0.0],
        [0.0, 0.5 * dt ** 2],
        [dt, 0.0],
        [0.0, dt],
    ])
    return F, B


def propagate_state(x: np.ndarray, a: np.ndarray, dt: float) -> np.ndarray:
    """只给 INS 使用的死 Reckoning (方便对比)"""
    F, B = get_F_B(dt)
    return F @ x + B @ a
