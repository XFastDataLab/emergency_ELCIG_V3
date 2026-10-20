"""错误反馈构建：把 ErrorRecord 集合转换为 LLM 可执行的定向修正指令。"""
from __future__ import annotations

from typing import List

from models import ErrorRecord


def build_feedback(errors: List[ErrorRecord]) -> str:
    """构建结构化定向修正反馈（自然语言指令）。"""
    ordered = sorted(errors, key=lambda e: e.severity)
    lines = [
        "规则验证器检测到以下逻辑错误，请进行【定向修正】：仅修改出错步骤，"
        "保留其他正确步骤不变，最小化改动范围。",
    ]
    for i, e in enumerate(ordered, 1):
        cn = ErrorRecord.TYPE_CN.get(e.error_type, e.error_type)
        lines.append(
            f"{i}. [错误类型：{cn}] 位置：{e.location}；"
            f"违反规则：{e.violated_rule}；说明：{e.description}；"
            f"修正要求：修正该步骤对应角色职责范围"
        )
    lines.append("请输出修正后的完整脚本（JSON，包含全部步骤）。")
    return "\n".join(lines)


def errors_to_json(errors: List[ErrorRecord]) -> List[dict]:
    return [e.to_dict() for e in errors]
