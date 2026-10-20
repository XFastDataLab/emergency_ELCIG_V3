"""配置加载：YAML + 环境变量占位符 ${ENV} 展开。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class AttrDict(dict):
    """支持点号访问的嵌套字典。"""

    def __init__(self, d: Dict):
        for k, v in d.items():
            self[k] = AttrDict(v) if isinstance(v, dict) else v

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def get(self, key: str, default=None):
        return self[key] if key in self else default


class Config(AttrDict):
    @classmethod
    def from_file(cls, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(_expand_env(raw))

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        return cls(_expand_env(d))

    def with_overrides(self, **overrides) -> "Config":
        """返回合并了覆盖项的新配置（用于消融矩阵切换开关）。"""
        import copy
        data = copy.deepcopy(dict(self))
        for k, v in overrides.items():
            data[k] = v
        return Config(data)
