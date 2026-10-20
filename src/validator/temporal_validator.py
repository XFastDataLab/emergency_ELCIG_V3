"""时序约束验证器：检查动作偏序（前置→后置）是否被满足。

语义（偏序 / partial order）：
  每条规则 (before, after) 表示「动作 before 应先于 after 执行」。
  - 当 before、after 都出现在脚本中时，校验两者之间是否满足先后（逆序/时间冲突即报错）。
  - 当 after 出现但 before 缺失时：
      * strict=True  （经典链式模式）：前置缺失视为硬错——强制要求链上每一环都齐全；
      * strict=False （默认，偏序宽容模式）：仅产生 temporal_hint 提示，不计入硬错误/CSR。
        应急流程中常会省略某些中间环节（如链 1→2→3→4 中允许没有 3，只要 1、2 在 4 之前即可），
        因此默认对「缺失的中间前置」宽容放过，只对「在场步骤的真实时序倒置」报错。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from models import ErrorRecord, Script
from .base import Validator


class TemporalValidator(Validator):
    def __init__(self, rules: List[Tuple[str, str]], strict: bool = False):
        # rules: [(before_action, after_action), ...]
        self.rules = rules
        self.strict = strict

    def constraint_count(self, script: Script) -> int:
        actions_present = {s.action for s in script.steps}
        return sum(1 for _, after in self.rules if after in actions_present)

    def validate(self, script: Script) -> List[ErrorRecord]:
        errors: List[ErrorRecord] = []
        # 记录每个动作首次出现的步骤序号（按脚本顺序）与时间
        first_idx: Dict[str, int] = {}
        first_time: Dict[str, str] = {}
        for i, s in enumerate(script.steps):
            if s.action not in first_idx:
                first_idx[s.action] = i
                first_time[s.action] = s.time

        def _time_leq(a, b) -> bool:
            # 简单按字符串比较 "HH:MM"
            return a is not None and b is not None and a <= b

        for before, after in self.rules:
            if after not in first_idx:
                continue  # 后置动作未出现，本规则不适用
            after_idx = first_idx[after]
            if before not in first_idx:
                # 前置动作缺失 -> 宽容模式仅提示，严格模式报硬错
                if self.strict:
                    errors.append(ErrorRecord(
                        error_type="temporal",
                        step_id=after_idx + 1,
                        location=f"动作'{after}'（步骤{after_idx + 1}）",
                        description=f"前置动作'{before}'缺失，违反时序约束：'{before}'必须先于'{after}'执行。",
                        violated_rule=f"时序:{before}→{after}",
                        severity=1,
                    ))
                else:
                    errors.append(ErrorRecord(
                        error_type="temporal_hint",
                        step_id=after_idx + 1,
                        location=f"动作'{after}'（步骤{after_idx + 1}）",
                        description=(f"提示：前置动作'{before}'未在脚本中出现。"
                                     f"偏序模式下允许省略该中间环节，仅作提示、不计入错误。"),
                        violated_rule=f"时序:{before}→{after}",
                        severity=1,
                    ))
                continue
            before_idx = first_idx[before]
            # 顺序倒置（无论严格/宽容，在场步骤的真实先后倒置都是逻辑错误）
            if before_idx > after_idx:
                errors.append(ErrorRecord(
                    error_type="temporal",
                    step_id=after_idx + 1,
                    location=f"动作'{after}'（步骤{after_idx + 1}）出现在'{before}'（步骤{before_idx + 1}）之前",
                    description=f"时序倒置：'{before}'必须先于'{after}'执行。",
                    violated_rule=f"时序:{before}→{after}",
                    severity=1,
                ))
            # 显式时间冲突（若两者都有时间且顺序与时间不一致）
            elif (first_time[before] and first_time[after]
                  and not _time_leq(first_time[before], first_time[after])):
                errors.append(ErrorRecord(
                    error_type="temporal",
                    step_id=after_idx + 1,
                    location=f"'{before}'({first_time[before]}) 与 '{after}'({first_time[after]})",
                    description=f"时间冲突：'{before}'({first_time[before]}) 晚于 '{after}'({first_time[after]})，违反时序约束。",
                    violated_rule=f"时序:{before}→{after}",
                    severity=1,
                ))
        return errors
