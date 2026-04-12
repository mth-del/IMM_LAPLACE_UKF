'''
Author: MOYUREN_Sea 1766853670@qq.com
Date: 2025-12-12 13:18:34
LastEditors: MOYUREN_Sea 1766853670@qq.com
LastEditTime: 2025-12-12 13:59:20
FilePath: /usbl_fusion/mian.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# main.py — 入口委托 sim_hub（多 sim 注册与调度）
from __future__ import annotations

from sim_hub import cli_main


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
