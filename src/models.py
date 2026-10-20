"""核心数据模型：结构化执行单元、脚本、错误记录。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ResourceDemand:
    """单条资源需求：资源类型 + 数量。"""
    resource_type: str
    quantity: int

    def to_dict(self) -> dict:
        return {"resource_type": self.resource_type, "quantity": self.quantity}


@dataclass
class ScriptStep:
    """一个演练步骤 = 执行时间 + 执行角色 + 执行动作 + 所需资源。

    - action: 执行动作的【精简概括】（建议 2-4 字，如“人员疏散”“启动响应”
      “泄漏控制”），用于职责/时序规则比对，必须经过规范化（normalize_action）。
    - description: 该步骤的【完整自然语言描述】（长文本放这里，不放 action），
      保留预案原文细节，供人工阅读与修正器上下文使用。
    """
    step_id: int
    time: Optional[str] = None          # "08:10" 或 None（未显式给出时间）
    role: str = ""                      # 执行角色（标准化后）
    action: str = ""                    # 执行动作（精简概括，标准化/接地后）
    description: str = ""               # 执行动作完整描述（长文本）
    resources: List[ResourceDemand] = field(default_factory=list)
    raw_text: str = ""                  # 该步骤原始自然语言描述
    action_raw: str = ""                # 模型原始动作表述（语义接地前），供审计
    action_grounded: bool = True        # True=已映射到提供的标准动作；False=语义不同、自动生成的4字动作

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "time": self.time,
            "role": self.role,
            "action": self.action,
            "description": self.description,
            "resources": [r.to_dict() for r in self.resources],
            "raw_text": self.raw_text,
            "action_raw": self.action_raw,
            "action_grounded": self.action_grounded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScriptStep":
        return cls(
            step_id=int(d.get("step_id", 0)),
            time=d.get("time"),
            role=d.get("role", ""),
            action=d.get("action", ""),
            description=d.get("description", ""),
            resources=[
                ResourceDemand(str(r.get("resource_type", "")), int(r.get("quantity", 0)))
                for r in d.get("resources", []) or []
            ],
            raw_text=d.get("raw_text", ""),
            action_raw=d.get("action_raw", ""),
            action_grounded=bool(d.get("action_grounded", True)),
        )


@dataclass
class ErrorRecord:
    """规则验证器输出的结构化错误记录（精确定位 + 约束依据）。"""
    error_type: str        # "temporal" | "responsibility" | "resource"
    step_id: int           # 出错步骤编号（职责/资源）；时序错误指向被违反的后置动作步骤
    location: str          # 人类可读的位置描述
    description: str       # 错误描述
    violated_rule: str     # 违反的具体规则（约束依据）
    severity: int = 2      # 1=最高（时序前置错误），2=资源，3=职责（影响后续执行的程度排序）

    TYPE_CN = {"temporal": "时序", "responsibility": "职责", "resource": "资源"}

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "error_type_cn": self.TYPE_CN.get(self.error_type, self.error_type),
            "step_id": self.step_id,
            "location": self.location,
            "description": self.description,
            "violated_rule": self.violated_rule,
            "severity": self.severity,
        }


@dataclass
class Script:
    """一次迭代得到的完整脚本。"""
    scenario_id: str
    steps: List[ScriptStep] = field(default_factory=list)
    raw_text: str = ""      # 当前脚本的自然语言全文
    grounding_report: list = field(default_factory=list)  # 动作语义接地报告（不序列化）

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "raw_text": self.raw_text,
            "steps": [s.to_dict() for s in self.steps],
        }

    def render(self) -> str:
        """把结构化脚本渲染回自然语言，便于送回 LLM 修正。"""
        lines = []
        for s in self.steps:
            res = "，".join(f"{r.resource_type}×{r.quantity}" for r in s.resources) or "无"
            t = s.time or "（未指定时间）"
            desc = f" | {s.description}" if s.description else ""
            lines.append(f"步骤{s.step_id} | {t} | {s.role} | {s.action}{desc} | 资源：{res}")
        return "\n".join(lines)
