"""组合校验器（规则验证器）：汇聚时序/职责/资源三个维度的约束检查。"""
from __future__ import annotations

from typing import List

from models import ErrorRecord, Script
from rules.registry import RuleSet
from validator.base import Validator
from validator.resource_validator import ResourceValidator
from validator.responsibility_validator import ResponsibilityValidator
from validator.temporal_validator import TemporalValidator


class RuleValidator(Validator):
    def __init__(self, ruleset: RuleSet, strict: bool = False, temporal_strict: bool = False):
        self.ruleset = ruleset
        # temporal_strict: True=前置缺失即硬错（经典链式，消融对照）；False=偏序宽容（默认）
        self.temporal = TemporalValidator(ruleset.temporal, strict=temporal_strict)
        self.responsibility = ResponsibilityValidator(
            ruleset.responsibility_by_role,
            conflicts=ruleset.responsibility_conflicts,
            strict=strict,
        )
        self.resource = ResourceValidator(ruleset.resource)

    def validate(self, script: Script) -> List[ErrorRecord]:
        errors: List[ErrorRecord] = []
        errors += self.temporal.validate(script)
        errors += self.responsibility.validate(script)
        errors += self.resource.validate(script)
        # 按严重程度排序：时序(1) > 资源(2) > 职责(3)，优先处理影响后续执行条件的错误
        errors.sort(key=lambda e: e.severity)
        return errors

    def constraint_count(self, script: Script) -> int:
        return (
            self.temporal.constraint_count(script)
            + self.responsibility.constraint_count(script)
            + self.resource.constraint_count(script)
        )

    def constraint_breakdown(self, script: Script):
        return {
            "temporal": self.temporal.constraint_count(script),
            "responsibility": self.responsibility.constraint_count(script),
            "resource": self.resource.constraint_count(script),
        }
