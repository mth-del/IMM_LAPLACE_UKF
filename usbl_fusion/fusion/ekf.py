'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:17:00
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:17:08
FilePath: /usbl_fusion/fusion/ekf.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# fusion/ekf.py
import numpy as np
from dataclasses import dataclass
from models.dynamics import get_F_B


@dataclass
class EKFConfig:
    dt: float
    acc_noise_std: float
    dvl_noise_std: float
    usbl_noise_std: float


class EkfUsblDvlIns:
    """集中式 EKF: x=[pN,pE,vN,vE]^T"""

    def __init__(self, cfg: EKFConfig):
        self.cfg = cfg
        self.F, self.B = get_F_B(cfg.dt)

        # 过程噪声：INS 加速度噪声通过 B 传递
        Q_acc = (cfg.acc_noise_std ** 2) * (self.B @ self.B.T)
        Q_extra = 1e-6 * np.eye(4)
        self.Q = Q_acc + Q_extra

        # 观测矩阵
        self.H_dvl = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.R_dvl = (cfg.dvl_noise_std ** 2) * np.eye(2)

        self.H_usbl = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        self.R_usbl = (cfg.usbl_noise_std ** 2) * np.eye(2)

        # 初值
        self.x = np.zeros(4)               # 可以根据需要设初值
        self.P = np.diag([100.0, 100.0, 1.0, 1.0])

    def predict(self, a_meas: np.ndarray):
        """利用 INS 加速度做预测"""
        self.x = self.F @ self.x + self.B @ a_meas
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_dvl(self, vel_meas: np.ndarray):
        """DVL 速度更新"""
        z = vel_meas
        H = self.H_dvl
        R = self.R_dvl

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_usbl(self, pos_meas: np.ndarray):
        """USBL 位置更新"""
        z = pos_meas
        H = self.H_usbl
        R = self.R_usbl

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
