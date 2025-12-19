"""
读取配置文件并构造 SimConfig / NoiseConfig。

为什么用 INI：
- Python 标准库 `configparser` 原生支持
- 文件里可以写注释，适合“参数需要解释”的场景

用法：
    from pathlib import Path
    from tools.config_loader import load_sim_and_noise_config
    sim_cfg, noise_cfg = load_sim_and_noise_config(Path("configs/sim_noise.ini"))
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple, Type, TypeVar

from data.simulator import SimConfig, NoiseConfig

T = TypeVar("T")


def _cast_like(value: str, like: Any) -> Any:
    """把 ini 的字符串值转换成与 like 同类型的值。"""
    if isinstance(like, bool):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(float(value))  # 允许 ini 写 10.0
    if isinstance(like, float):
        return float(value)
    return value


def _dataclass_from_section(cls: Type[T], section: Dict[str, str]) -> T:
    """
    只用 dataclass 已定义字段构造对象：
    - ini 里多余字段会被忽略（方便你加注释/占位）
    - ini 里缺字段会使用 dataclass 默认值
    """
    obj = cls()  # type: ignore[call-arg]
    for f in fields(cls):
        if f.name in section:
            setattr(obj, f.name, _cast_like(section[f.name], getattr(obj, f.name)))
    return obj


def load_sim_and_noise_config(path: str | Path) -> Tuple[SimConfig, NoiseConfig]:
    """
    从 ini 加载配置：
    - [sim]  -> SimConfig
    - [noise]-> NoiseConfig
    """
    path = Path(path)
    parser = ConfigParser()
    parser.read(path, encoding="utf-8")

    sim_cfg = _dataclass_from_section(SimConfig, dict(parser["sim"]) if parser.has_section("sim") else {})
    noise_cfg = _dataclass_from_section(NoiseConfig, dict(parser["noise"]) if parser.has_section("noise") else {})
    return sim_cfg, noise_cfg


