"""动作语义接地：将大模型生成的自由动作映射到我们提供的标准动作本体。

设计目标（对应用户需求：生成时把每个 step 的动作自动语义映射到“我们提供的 action”，
方便后续验证器校验；若语义不同则自行创建 4 字精简描述）：

  两阶段映射：
    1) 别名 / 规范名精确归一（复用 rules.registry.normalize_action）；
    2) 词汇-语义相似度匹配（在动作本体中找最相似的规范动作）。
  若以上均未命中（语义不同、本体无对应动作），则自动生成一个【4 字精简动作描述】，
  保证 action 字段始终是“可比对”的精简动作，便于后续验证器（职责白名单）校验。

  接地状态 status:
    - "ontology" : 归一化后已直接命中标准动作本体；
    - "semantic" : 词汇-语义匹配命中本体中的某规范动作；
    - "auto"     : 语义不同、本体无对应，已自动生成 4 字动作（action_grounded=False）。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Set

from models import Script, ScriptStep
from rules.matching import three_character_match
from rules.registry import (
    ACTION_ALIASES,
    CANON_TO_ALIASES,
    get_action_ontology,
    normalize_action,
)

# 仅保留中文（用于 4 字抽取与字符 n-gram）
_CN = re.compile(r"[一-鿿]")
_PUNCT = set("，。、；：,.;:！？!?（）()《》<> \t\n")


def _clean(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in _PUNCT)


def _char_ngrams(s: str, n: int = 2) -> Set[str]:
    s = _clean(s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _similarity(text: str, canon: str, aliases: Set[str]) -> float:
    """(text) 与规范动作 canon 的词汇-语义相似度 [0,1]。

    - 规范动作或其任一别名作为子串出现在 text 中 -> 高度相似（0.95~1.0）；
    - 否则用字符二元组 Jaccard 重叠估计语义相近程度。
    """
    if canon and canon in text:
        return 1.0
    for al in aliases:
        if al and al in text:
            return 0.95
    tg = _char_ngrams(text, 2)
    cg = _char_ngrams(canon, 2)
    if tg and cg:
        union = len(tg | cg)
        if union:
            return len(tg & cg) / union
    return 0.0


def semantic_match(action: str, description: str, ontology: Set[str],
                   threshold: float = 0.34) -> Optional[str]:
    """在动作本体中寻找与 (action+description) 词汇-语义最相似的规范动作。"""
    text = _clean(action) + _clean(description)
    if not text:
        return None
    best: Optional[str] = None
    best_score = 0.0
    # 三字匹配只比较动作本身，避免长描述把无关职责映射过来。
    matches = sorted(canon for canon in ontology if three_character_match(action, canon))
    if len(matches) == 1:
        return matches[0]
    for canon in sorted(ontology):
        score = _similarity(text, canon, CANON_TO_ALIASES.get(canon, set()))
        if score > best_score:
            best_score = score
            best = canon
    if best is not None and best_score >= threshold:
        return best
    return None


def make_four_char_label(text: str) -> str:
    """语义不同（本体无对应）时，自动生成 4 字精简动作描述。

    策略：仅保留中文，清洗后取 2-4 字；过长则截取前 4 字，过短则以“处置”兜底补足，
    确保为 2-4 字精简动作，便于后续验证器比对。
    """
    clean = "".join(_CN.findall(text or ""))
    if not clean:
        return "处置"
    if 2 <= len(clean) <= 4:
        return clean
    if len(clean) > 4:
        return clean[:4]
    # 长度 1：补“处置”
    return (clean + "处置")[:4]


def ground_step(step: ScriptStep, ontology: Set[str], threshold: float = 0.34) -> Dict:
    """对单步做动作接地，就地更新 step.action / action_grounded，返回状态字典。"""
    current = step.action
    # 1) 归一化后已直接命中标准动作本体
    if current in ontology:
        step.action_grounded = True
        return {"status": "ontology", "raw": step.action_raw, "grounded": current}
    # 2) 词汇-语义匹配命中本体中的规范动作
    match = semantic_match(current, step.description, ontology, threshold)
    if match is not None:
        step.action = match
        step.action_grounded = True
        return {"status": "semantic", "raw": step.action_raw, "grounded": match}
    # 3) 语义不同：自动生成 4 字动作
    label = make_four_char_label(current or step.description)
    step.action = label
    step.action_grounded = False
    return {"status": "auto", "raw": step.action_raw, "grounded": label}


def ground_script(script: Script, scenario_id: str, threshold: float = 0.34,
                  provided_actions: Optional[Set[str]] = None) -> List[Dict]:
    """对整脚本做动作接地；返回接地报告并挂载到 script.grounding_report。"""
    ontology = provided_actions if provided_actions is not None else get_action_ontology(scenario_id)
    report: List[Dict] = []
    for s in script.steps:
        if not s.action_raw:
            s.action_raw = s.action
        rep = ground_step(s, ontology, threshold)
        rep["step_id"] = s.step_id
        report.append(rep)
    script.grounding_report = report
    script.raw_text = script.render()
    return report


class LLMActionGrounder:
    """可选：用一次 LLM 批量调用做“真正的语义映射 + 4 字生成”。

    确定性接地（action_grounding 默认）已能覆盖绝大多数同义表述；本类在
    use_llm=true 时进一步用大模型理解语义，将自由动作映射到标准动作集合，
    或对集合外动作生成 4 字描述。失败时由调用方回退到确定性接地结果。
    """

    def __init__(self, client, threshold: float = 0.34):
        self.client = client
        self.threshold = threshold

    def ground(self, script: Script, scenario_id: str,
               provided_actions: Optional[Set[str]] = None) -> List[Dict]:
        ontology = provided_actions or get_action_ontology(scenario_id)
        if not script.steps or self.client is None:
            return []
        payload = [
            {"step_id": s.step_id, "action": s.action_raw or s.action,
             "description": s.description}
            for s in script.steps
        ]
        ontology_str = "、".join(sorted(ontology))
        system = (
            "你是应急演练动作标准化专家。给定一组步骤的 action 与 description，以及【标准动作集合】，"
            "请将每个步骤的动作语义映射到标准动作集合中最贴切的一项；"
            "若集合中无语义对应的动作（即该步骤语义不属于标准动作体系），请为该步骤自行生成"
            "一个4字中文精简动作描述。只输出 JSON，不要解释。"
        )
        user = (
            f"【标准动作集合】{ontology_str}\n"
            f"【待接地步骤】{json.dumps(payload, ensure_ascii=False)}\n"
            '请输出 JSON 数组：[{"step_id":N,"action":"<映射或生成的4字动作>","grounded":true/false}]'
        )
        try:
            from llm.client import extract_json
            text = self.client.chat(system, user, json_mode=True)
            obj = extract_json(text)
        except Exception as e:  # 任何异常都回退到确定性接地（已在 steps_from_json 完成）
            print(f"[grounding] LLM 语义接地失败，回退确定性接地：{e}")
            return []
        if isinstance(obj, dict) and "steps" in obj:
            obj = obj["steps"]
        if not isinstance(obj, list):
            return []
        by_id = {d.get("step_id"): d for d in obj if isinstance(d, dict)}
        report: List[Dict] = []
        for s in script.steps:
            d = by_id.get(s.step_id)
            if not d:
                continue
            new_action = (d.get("action") or "").strip()
            if not new_action:
                continue
            # 规整为 <=4 字精简动作
            if len(new_action) > 4:
                new_action = new_action[:4]
            s.action = new_action
            s.action_grounded = bool(d.get("grounded", False))
            report.append({"step_id": s.step_id, "status": "llm",
                           "raw": s.action_raw, "grounded": new_action})
        script.grounding_report = report
        script.raw_text = script.render()
        return report
