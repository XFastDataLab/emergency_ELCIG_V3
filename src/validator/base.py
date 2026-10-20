"""校验器抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import sys
import os


from models import ErrorRecord, Script


class Validator(ABC):
    @abstractmethod
    def validate(self, script: Script) -> List[ErrorRecord]:
        """返回脚本中违反本维度约束的全部错误。"""
        raise NotImplementedError

    @abstractmethod
    def constraint_count(self, script: Script) -> int:
        """返回本维度在当前脚本上"实际受检的约束数量"（用于 CSR 分母）。"""
        raise NotImplementedError
