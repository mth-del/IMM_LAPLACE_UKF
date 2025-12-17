'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:30:21
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:53:38
FilePath: /usbl_fusion/samples/test_1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from models.usbl_array import create_usbl_array, calculate_direction_and_range
from models.coordinate_transformation import transform_to_navigation_frame

def main():
    print("mian runtime")
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
    
    print("目标相对阵列的距离：", distances)
    print("目标相对阵列的方向（方位角和距离）：", directions)
    
    # 计算目标在导航坐标系下的位置
    roll, pitch, yaw = 0.0, 0.0, 0.5  # 假设船的姿态
    transformed_position = transform_to_navigation_frame(target_position, ship_position, roll, pitch, yaw)
    
    print("目标在导航坐标系下的位置：", transformed_position)


if __name__ == "__main__":
    main()
