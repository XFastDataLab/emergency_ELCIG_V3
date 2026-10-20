"""职责约束验证器：检查 (角色, 动作) 是否属于该部门的合法职责映射。

一个部门对应多个职责——以「部门 -> 其合法职责动作集合」表示。
某步 (角色, 动作) 只要命中该部门职责集合中的「任意一个」动作，即视为合法、通过。

判定采用三态，而非"白名单之外一律判错"，以克服枚举式白名单的两类误报：
  1) 步骤合理但白名单未收录；
  2) 语义与白名单一致、但表述/别名不在归一化集合。

三态逻辑：
  * 命中白名单 (动作属于该部门职责集合)  -> 合法，通过；
  * 命中矛盾黑名单 (conflicts)           -> 职责错配（severity=3，硬错误）；
  * 二者都未命中（未知组合）             -> 默认宽容：仅产生 severity=1 的"待复核"提示，
                                           不计入硬错误（strict=False 时）。
                                           strict=True 时退化为原行为（未知即判错），
                                           用于论文消融对照。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from models import ErrorRecord, Script
from rules.matching import three_character_match
from .base import Validator


class ResponsibilityValidator(Validator):
    def __init__(self,
                 allowed_actions_by_role: Dict[str, Set[str]],
                 conflicts: Optional[Set[Tuple[str, str]]] = None,
                 strict: bool = False):
        # allowed_actions_by_role: 部门 -> 其全部合法职责动作集合。
        #   一个部门对应多个职责；某步 (role, action) 只要 action 属于该部门
        #   职责集合中的「任意一个」，即视为合法、校验通过。
        self.allowed_actions_by_role = allowed_actions_by_role
        self.conflicts = conflicts or set()
        # strict=True ：白名单之外一律判为职责错配（论文基线 / 消融对照，保留原行为）
        # strict=False：命中矛盾黑名单才报硬错；未知组合仅提示，不计入硬错误
        self.strict = strict

    def constraint_count(self, script: Script) -> int:
        return len(script.steps)

    def validate(self, script: Script) -> List[ErrorRecord]:
        errors: List[ErrorRecord] = []
        for s in script.steps:
            role_actions = self.allowed_actions_by_role.get(s.role)
            # 1) 命中白名单：该部门职责集合中包含当前动作 -> 通过（满足任一职责即可）
            if role_actions is not None and s.action in role_actions:
                continue
            # 2) 命中矛盾黑名单 -> 职责错配（severity=3，硬错误）
            if (s.role, s.action) in self.conflicts:
                errors.append(self._hard_error(s))
                continue
            # 仅在当前部门的合法职责内放宽动作匹配，不合并部门；明确冲突优先。
            if role_actions and any(three_character_match(s.action, a) for a in role_actions):
                continue
            # 3) 白名单与黑名单都未命中：未知组合
            if self.strict:
                # 严格模式：未知即判错（原行为，用于消融对照）
                errors.append(self._hard_error(s))
            else:
                # 宽容模式：仅产生"待复核"提示，不计入硬错误，
                # 避免白名单枚举不全 / 别名或表述不同造成的误报。
                errors.append(self._unknown_hint(s))
        return errors

    def _hard_error(self, s) -> ErrorRecord:
        return ErrorRecord(
            error_type="responsibility",
            step_id=s.step_id,
            location=f"步骤{s.step_id}：角色'{s.role}'执行动作'{s.action}'",
            description=f"职责错配：角色'{s.role}'不包含动作'{s.action}'。",
            violated_rule=f"职责:({s.role},{s.action})",
            severity=3,
        )

    def _unknown_hint(self, s) -> ErrorRecord:
        allowed = sorted(self.allowed_actions_by_role.get(s.role, set()))
        desc = (
            f"职责待复核：角色'{s.role}'执行动作'{s.action}'未收录于职责白名单，"
            f"且非明确矛盾组合。"
        )
        if allowed:
            desc += f"该部门法定职责为：{'、'.join(allowed)}；"
        desc += ("可能为白名单未覆盖的合理使用（如新增职责或同义表述），"
                 "建议人工确认；默认不计入职责错误。")
        return ErrorRecord(
            error_type="responsibility_hint",
            step_id=s.step_id,
            location=f"步骤{s.step_id}：角色'{s.role}'执行动作'{s.action}'",
            description=desc,
            violated_rule=f"职责:({s.role},{s.action})",
            severity=1,
        )
