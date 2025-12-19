"""
项目日志统一入口（推荐）：loguru

特点：
- 开箱即用：logger.info/debug/warning/error
- 控制台彩色输出
- 文件落盘 + 轮转(rotation) + 保留(retention)
- 可用环境变量调参：LOG_LEVEL / LOG_DIR
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger


logger = _logger


class _InterceptHandler(logging.Handler):
    """把标准库 logging 的日志转发给 loguru（便于未来接入第三方库）"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logger(
    *,
    level: Optional[str] = None,
    log_dir: Optional[str | os.PathLike[str]] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    初始化全局 logger（建议在程序入口调用一次）

    - level: 日志级别，默认读环境变量 LOG_LEVEL，否则 INFO
    - log_dir: 日志目录，默认读环境变量 LOG_DIR，否则 ./logs
    """
    level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    log_dir = Path(log_dir or os.getenv("LOG_DIR") or "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 清理默认 handler，避免重复输出
    logger.remove()

    # 控制台输出（开发友好）
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=False,  # 诊断信息很冗长，默认关；需要再打开
        enqueue=True,
    )

    # 文件落盘（分析/复现实验方便）
    logger.add(
        str(log_dir / "usbl_fusion_{time:YYYYMMDD}.log"),
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,
    )

    # 接管标准库 logging
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.getLevelName(level))
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False


