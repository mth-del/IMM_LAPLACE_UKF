'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:28:20
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:56:23
FilePath: /usbl_fusion/models/coordinate_transformation.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# models/coordinate_transformation.py
import numpy as np

def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    创建船体坐标系到导航坐标系的旋转矩阵。
    
    Args:
        roll (float): 横摇角 (绕 x 轴)
        pitch (float): 俯仰角 (绕 y 轴)
        yaw (float): 艏向角 (绕 z 轴)
        
    Returns:
        np.ndarray: 旋转矩阵 (3×3)
    """
    # 计算每个旋转矩阵
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # 合成旋转矩阵
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    return R


def transform_to_navigation_frame(target_position: np.ndarray, ship_position: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    将目标位置从 USBL 阵列坐标系转换为导航坐标系。
    
    Args:
        target_position (np.ndarray): 目标在 USBL 阵列坐标系下的位置 (x_B, y_B, z_B)
        ship_position (np.ndarray): 船的当前位置 (x_N, y_N, z_N)
        roll (float): 横摇角 (radians)
        pitch (float): 俯仰角 (radians)
        yaw (float): 艏向角 (radians)
        
    Returns:
        np.ndarray: 转换后的目标位置 (x_N, y_N, z_N)
    """
    # 船体坐标系到导航坐标系的旋转矩阵
    R = rotation_matrix(roll, pitch, yaw)
    
    # 目标相对于船的坐标
    relative_position = target_position - ship_position
    
    # 旋转到导航坐标系
    transformed_position = np.dot(R, relative_position)
    
    return transformed_position
