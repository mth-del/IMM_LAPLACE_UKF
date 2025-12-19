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

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        EKF 线性量测更新（理论对应）：
        - 量测模型：z = H x + v,  v ~ N(0, R)
        - 创新：y = z - H x
        - 创新协方差：S = H P H^T + R
        - 卡尔曼增益：K = P H^T S^{-1}
        - 状态更新：x <- x + K y
        - 协方差更新（Joseph 形式，更稳）：P <- (I-KH)P(I-KH)^T + K R K^T
        """
        z = np.asarray(z, dtype=float).reshape(-1)

        S = H @ self.P @ H.T + R
        # 用 solve 代替 inv：K = P H^T S^{-1}
        # np.linalg.solve(A, B) 解 A X = B
        # 令 A = S (m×m), B = (P H^T)^T (m×n) => X = (S^{-1}(P H^T)^T) (m×n)
        # K = X^T (n×m)
        PHt = self.P @ H.T
        K = np.linalg.solve(S, PHt.T).T

        y = z - H @ self.x
        self.x = self.x + K @ y

        I = np.eye(self.P.shape[0])
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        # 数值误差下保证对称性（不改变理论意义）
        self.P = 0.5 * (self.P + self.P.T)

    def predict(self, a_meas: np.ndarray):
        """利用 INS 加速度做预测"""
        a = np.asarray(a_meas, dtype=float).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f"INS a_meas 维度应为 (2,), 实际为 {a.shape}")
        self.x = self.F @ self.x + self.B @ a
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_dvl(self, vel_meas: np.ndarray):
        """
        DVL 速度更新
        - 量测：z_v = [vN, vE]^T
        - H_dvl 选取状态中的速度分量
        """
        z = np.asarray(vel_meas, dtype=float).reshape(-1)
        if z.shape[0] != 2:
            raise ValueError(f"DVL vel_meas 维度应为 (2,), 实际为 {z.shape}")
        self._update(z=z, H=self.H_dvl, R=self.R_dvl)

    def update_usbl(self, pos_meas: np.ndarray):
        """
        USBL 位置更新
        - 量测：z_p = [pN, pE]^T
        - H_usbl 选取状态中的位置分量
        """
        z = np.asarray(pos_meas, dtype=float).reshape(-1)
        if z.shape[0] != 2:
            raise ValueError(f"USBL pos_meas 维度应为 (2,), 实际为 {z.shape}")
        self._update(z=z, H=self.H_usbl, R=self.R_usbl)
