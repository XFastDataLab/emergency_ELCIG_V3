"""脚本解析：把模型输出（JSON 或行格式文本）转换为结构化 Script。"""
from __future__ import annotations

import json
import re
from typing import List

from models import ResourceDemand, Script, ScriptStep
from rules.registry import normalize_action, normalize_role
from action_grounding import ground_script


def _is_long_action(text: str) -> bool:
    """判断一段文本是否更像“完整描述”而非“精简动作”：
    含中文标点/分隔符，或长度超过 8 个字符。"""
    t = (text or "").strip()
    if not t:
        return False
    return ("，" in t or "、" in t or "；" in t or "。" in t
            or "," in t or ";" in t or len(t) > 8)


def _parse_resources_inline(res_part: str) -> List[dict]:
    """解析行格式里的资源串：类型×数量, 类型×数量 → 资源字典列表。"""
    out: List[dict] = []
    res_part = (res_part or "").strip()
    if res_part and res_part != "无":
        for item in res_part.split(","):
            mm = re.match(r"(.+?)×(\d+)", item.strip())
            if mm:
                out.append({"resource_type": mm.group(1).strip(), "quantity": int(mm.group(2))})
    return out


def _coerce_step_dict(s: str) -> dict:
    """模型把某一步输出为纯文本/JSON 字符串而非对象时，尽量还原为步骤字典。

    还原优先级：① 整体是 JSON 对象；② 行格式「步骤N | 时间 | 角色 | 动作 | 资源」；
    ③ 兜底：整段作为 description（缺失 role/action 由后续验证器报出，不崩溃）。
    """
    t = (s or "").strip().strip("`").strip()
    if not t:
        return {}
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.match(
        r"步骤\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.*)",
        t,
    )
    if m:
        action = m.group(4).strip()
        return {
            "step_id": int(m.group(1)),
            "time": m.group(2).strip(),
            "role": m.group(3).strip(),
            "action": action,
            "description": action if _is_long_action(action) else "",
            "resources": _parse_resources_inline(m.group(5)),
        }
    return {"description": t}


def steps_from_json(obj: dict, scenario_id: str,
                    provided_actions=None, threshold: float = 0.34) -> Script:
    """从 LLM 返回的 JSON（含 steps 数组）构造 Script，并对动作/角色做标准化 + 语义接地。

    - action 应为 2-4 字精简动作；若模型把长描述误塞进 action，则把长文本搬到
      description，并对 action 做规范化（含长串子串兜底提炼）。
    - description 保留步骤完整自然语言描述。
    - 生成期语义接地：将每个 step 的动作自动映射到“我们提供的标准动作”本体
      （provided_actions 优先；缺省由职责规则自动推导），方便后续验证器校验；
      若语义不同（本体无对应），自动生成 4 字精简动作。
    """
    steps: List[ScriptStep] = []
    raw = obj.get("steps", []) or []
    if not isinstance(raw, list) or not raw:
        raise ValueError("steps 必须为非空数组，空脚本不能作为实验成功结果")
    for i, s in enumerate(raw, start=1):
        if isinstance(s, str):
            # 模型可能把单个步骤输出为纯文本/JSON 字符串而非对象，先尽量还原为字典
            s = _coerce_step_dict(s)
        if not isinstance(s, dict):
            print(f"[parser] 跳过无法解析的步骤 #{i}（期望对象，实际 {type(s).__name__}）：{str(s)[:80]}")
            continue
        step = ScriptStep.from_dict(s)
        step.step_id = i
        orig_action = (s.get("action") or "").strip()
        # 若模型把长描述误塞进 action，先搬到 description，避免职责/时序比对失效
        if not step.description and _is_long_action(step.action):
            step.description = step.action
        step.action_raw = orig_action or step.action   # 保留模型原始动作表述（接地前）
        step.action = normalize_action(step.action)
        step.role = normalize_role(step.role)
        steps.append(step)
    if not steps:
        raise ValueError("模型输出中没有可解析的步骤")
    sc = Script(scenario_id=scenario_id, steps=steps)
    # 生成期动作语义接地（写入 script.grounding_report，供生成器打印）
    sc.grounding_report = ground_script(sc, scenario_id,
                                        threshold=threshold,
                                        provided_actions=provided_actions)
    return sc


def parse_regex(text: str, scenario_id: str) -> Script:
    """解析行格式文本（模板生成器 / 模型自然语言输出）：
       步骤N | 时间 | 角色 | 动作 | 资源：类型×数量, 类型×数量
    """
    steps: List[ScriptStep] = []
    pattern = re.compile(
        r"步骤\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*资源[:：]\s*(.*)"
    )
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        sid = int(m.group(1))
        t = m.group(2).strip()
        role = normalize_role(m.group(3).strip())
        action_raw = m.group(4).strip()
        action = normalize_action(action_raw)
        res_part = m.group(5).strip()
        resources: List[ResourceDemand] = []
        if res_part and res_part != "无":
            for item in res_part.split(","):
                mm = re.match(r"(.+?)×(\d+)", item.strip())
                if mm:
                    resources.append(ResourceDemand(mm.group(1).strip(), int(mm.group(2))))
        t_val = None if t in ("（未指定时间）", "未指定时间", "-") else t
        desc = action_raw if _is_long_action(action_raw) else ""
        steps.append(ScriptStep(step_id=sid, time=t_val, role=role, action=action,
                                 description=desc, resources=resources,
                                 raw_text=line.strip()))
    return Script(scenario_id=scenario_id, steps=steps, raw_text=text)
