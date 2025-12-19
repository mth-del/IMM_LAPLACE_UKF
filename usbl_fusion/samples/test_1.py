'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:30:21
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:53:38
FilePath: /usbl_fusion/samples/test_1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# 说明：本脚本位于 samples/ 子目录。
# 若直接 `python samples/test_1.py` 运行，Python 默认不会把上级目录(usbl_fusion/)加入模块搜索路径，
# 因此需要在这里补一行 sys.path，确保 `models/`、`logging_setup.py` 等可被正常导入。

import sys
from pathlib import Path

# 把 usbl_fusion/ 加入 sys.path，保证脚本可独立运行
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from models.usbl_array import create_usbl_array, calculate_direction_and_range
from models.coordinate_transformation import transform_to_navigation_frame
from tools.logging_setup import setup_logger, logger

def main():
    setup_logger()
    logger.info("sample test_1 runtime")
    # 参数设置
    d = 0.04   # 传统阵列间距（m）
    l = 0.04   # 短基线（m）
    L = 0.32   # 长基线（m）
    ship_position = np.array([100, 200, 600])  # 船的当前位置 [x, y, z] (m)
    target_position = np.array([150, 250, 600])  # 目标的全局位置 (m)

    # 生成阵列
    hydrophones = create_usbl_array(d, L, l)
    
    # 计算目标的相对位置和方向
    distances, directions = calculate_direction_and_range(target_position, hydrophones)
    
    logger.info("目标相对阵列的距离：{}", distances)
    logger.info("目标相对阵列的方向（方位角和距离）：{}", directions)
    
    # 计算目标在导航坐标系下的位置
    roll, pitch, yaw = 0.0, 0.0, 0.5  # 假设船的姿态
    transformed_position = transform_to_navigation_frame(target_position, ship_position, roll, pitch, yaw)
    
    logger.info("目标在导航坐标系下的位置：{}", transformed_position)


if __name__ == "__main__":
    main()
