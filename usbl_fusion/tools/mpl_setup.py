"""
Matplotlib backend 选择与无 GUI 兜底。

背景：
- 某些环境下 matplotlib 默认 backend 可能是 gtk4agg。
- 当 PyGObject/GTK4 依赖不完整或版本不匹配时，plt.show() 会触发运行时崩溃/断言失败。

策略：
- 默认无 GUI：强制使用 Agg（稳定、可保存图片）
- 需要 GUI：优先 QtAgg（若安装 PyQt6/PySide6），否则 TkAgg（若 tkinter 可用），最后回退 Agg
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Optional


def _can_import(module: str) -> bool:
    try:
        import_module(module)
        return True
    except Exception:
        return False


def configure_matplotlib(*, gui: bool, force: bool = True) -> str:
    """
    在 import matplotlib.pyplot 之前调用。

    返回值：最终使用的 backend 名称。
    """
    import os
    import matplotlib

    # 若用户显式指定 MPLBACKEND，则尊重用户选择
    if os.getenv("MPLBACKEND"):
        return matplotlib.get_backend()

    backend: str
    if not gui:
        backend = "Agg"
    else:
        if _can_import("PyQt6") or _can_import("PySide6"):
            backend = "QtAgg"
        elif _can_import("tkinter"):
            backend = "TkAgg"
        else:
            backend = "Agg"

    try:
        matplotlib.use(backend, force=force)
        return backend
    except Exception:
        # 最终兜底：Agg 基本不依赖 GUI
        matplotlib.use("Agg", force=True)
        return "Agg"


def save_all_figures(
    out_dir: str | Path,
    *,
    prefix: str = "figure",
    dpi: int = 150,
) -> list[Path]:
    """
    保存当前进程中所有已创建的 matplotlib figure。
    需在绘图完成且已 import matplotlib.pyplot 后调用。
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for fignum in plt.get_fignums():
        fig = plt.figure(fignum)
        path = out_dir / f"{prefix}_{int(fignum):02d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
    return saved


