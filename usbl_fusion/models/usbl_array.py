import numpy as np

def create_usbl_array(d: float, L: float, l: float):
    """
    创建 USBL 阵列并返回阵列的坐标。
    阵列包含四个基元，位置安排根据论文中的设计方案：
    - 元1在原点
    - 元2在 x 轴正方向，距离 L
    - 元3在第一象限对角线，距元1为 l
    - 元4在 y 轴正方向，距离 L
    
    Args:
        d (float): 传统阵列间距，当前不使用
        L (float): 长基线
        l (float): 短基线
        
    Returns:
        np.ndarray: 阵列中 4 个基元的坐标 (N×3)
    """
    arr = np.array([
        [0.0, 0.0, 0.0],         # 元1：原点
        [L, 0.0, 0.0],           # 元2：x 轴正向
        [l * np.cos(np.pi / 4), l * np.sin(np.pi / 4), 0.0],  # 元3：第一象限
        [0.0, L, 0.0],           # 元4：y 轴正向
    ])
    return arr


def calculate_direction_and_range(target_position: np.ndarray, hydrophones: np.ndarray):
    """
    根据目标位置与阵列的阵元坐标计算目标相对于阵列的方向和距离。
    
    Args:
        target_position (np.ndarray): 目标在全局坐标系中的位置 (x, y, z)
        hydrophones (np.ndarray): 阵列的坐标 (N×3)，每行是一个阵元坐标
        
    Returns:
        np.ndarray: 目标相对于每个阵元的距离 (N,)
        np.ndarray: 目标相对于每个阵元的方向 (N×2)，包含方位角和仰角
    """
    target_vector = target_position[:2]  # 只关心 x, y
    distances = np.linalg.norm(hydrophones[:, :2] - target_vector, axis=1)
    
    # 计算方位角（theta）和仰角（phi）
    directions = np.zeros((len(hydrophones), 2))
    for i, hydrophone in enumerate(hydrophones):
        dx = target_vector[0] - hydrophone[0]
        dy = target_vector[1] - hydrophone[1]
        range_ = np.linalg.norm([dx, dy])
        
        theta = np.arctan2(dy, dx)  # 方位角
        directions[i, :] = [theta, range_]
        
    return distances, directions
