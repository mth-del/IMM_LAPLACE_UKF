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
    # ConfigParser 允许“续行值”（下一行以空白开头会拼接到上一行的 value）。
    # 为了避免 ini 里不小心缩进导致数值字段混入多行文本，这里只取第一行并去掉行内注释。
    v = (value.splitlines()[0] if isinstance(value, str) else str(value)).strip()
    # 手动剔除行内注释（即使 ConfigParser 未开启 inline_comment_prefixes 也能稳健解析）
    for sep in (";", "#"):
        if sep in v:
            v = v.split(sep, 1)[0].strip()

    if isinstance(like, bool):
        return v.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(float(v))  # 允许 ini 写 10.0
    if isinstance(like, float):
        return float(v)
    return v


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
    # 支持 ini 行内注释（例如：turn_direction = 1  ; +1 逆时针）
    parser = ConfigParser(inline_comment_prefixes=(";", "#"))
    parser.read(path, encoding="utf-8")

    sim_cfg = _dataclass_from_section(SimConfig, dict(parser["sim"]) if parser.has_section("sim") else {})
    noise_cfg = _dataclass_from_section(NoiseConfig, dict(parser["noise"]) if parser.has_section("noise") else {})
    return sim_cfg, noise_cfg


