"""修正器：基于反馈对脚本做定向修正。

- LLMCorrector：真实模式，调用大模型按结构化反馈局部修正（本文 ELCIG 方法）。
- RuleGuidedCorrector：离线模式，依据规则做确定性修正，使闭环在无网络时也能收敛、演示 ERR/CSR 提升。
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

from feedback import build_feedback
from audit import emit, print_ai_request, print_ai_response
from llm.client import LLMClient, extract_json
from models import Script, ScriptStep
from parser import steps_from_json
from prompts import build_correct
from rules.registry import RuleSet


class Corrector(ABC):
    @abstractmethod
    def correct(self, script: Script, errors: List, scenario: Dict) -> Script:
        raise NotImplementedError


class LLMCorrector(Corrector):
    def __init__(self, client: LLMClient, ruleset=None):
        self.client = client
        self.ruleset = ruleset

    def correct(self, script: Script, errors: List, scenario: Dict) -> Script:
        fb = build_feedback(errors)
        system, user = build_correct(scenario, script.render(), fb, self.ruleset)
        label = f"ELCIG 修正｜场景 {scenario['id']}"
        print_ai_request(label, system, user)
        text = self.client.chat(system, user, json_mode=True)
        print_ai_response(label, text)
        obj = extract_json(text)
        if not isinstance(obj, dict) or "steps" not in obj:
            emit("correction_rejected", reason="invalid_json_or_missing_steps")
            print("[ELCIG 修正] 回复 JSON 无效或缺少 steps，本轮保留原脚本。")
            return script  # 解析失败则保持原脚本
        try:
            return steps_from_json(obj, scenario["id"])
        except (ValueError, TypeError, AttributeError) as exc:
            emit("correction_rejected", reason=str(exc))
            print(f"[ELCIG 修正] 脚本解析失败，本轮保留原脚本：{exc}")
            return script


class RuleGuidedCorrector(Corrector):
    """确定性规则引导修正：直接消除校验器报告的各类错误，使 ELCIG 闭环必然收敛。"""

    def __init__(self, ruleset: RuleSet):
        self.ruleset = ruleset

    def correct(self, script: Script, errors: List, scenario: Dict) -> Script:
        steps = [copy.deepcopy(s) for s in script.steps]

        # 1) 职责错配：把角色改为该动作的法定角色
        for e in errors:
            if e.error_type == "responsibility":
                st = self._find(steps, e.step_id)
                if st:
                    allowed = self.ruleset.allowed_roles_for(st.action)
                    if allowed:
                        st.role = next(iter(allowed))

        # 2) 时序（缺失前置插入 + 倒置重排 + 时间重戳），重复至稳定
        self._fix_temporal(steps)

        # 3) 资源超配 / 并发竞争（此时时间已稳定）
        self._fix_resources(steps)

        for i, st in enumerate(steps, 1):
            st.step_id = i
        sc = Script(scenario_id=script.scenario_id, steps=steps)
        sc.raw_text = sc.render()
        return sc

    # ---------------- 内部工具 ----------------
    @staticmethod
    def _find(steps, step_id):
        for s in steps:
            if s.step_id == step_id:
                return s
        return None

    def _fix_temporal(self, steps: List[ScriptStep]):
        for _ in range(10):
            if not self._fix_temporal_once(steps):
                break

    def _fix_temporal_once(self, steps: List[ScriptStep]) -> bool:
        changed = False
        for before, after in self.ruleset.temporal:
            after_idx = next((i for i, s in enumerate(steps) if s.action == after), None)
            if after_idx is None:
                continue
            before_idx = next((i for i, s in enumerate(steps) if s.action == before), None)
            if before_idx is None:
                # 插入缺失的前置步骤，时间重戳为后置步骤时间以保持一致
                role = next(iter(self.ruleset.allowed_roles_for(before)), "应急指挥部")
                steps.insert(after_idx, ScriptStep(
                    step_id=0, time=steps[after_idx].time,
                    role=role, action=before, resources=[],
                ))
                changed = True
                continue
            if before_idx > after_idx:
                # 把前置步骤移动到后置步骤之前，并重戳时间保持一致
                st = steps.pop(before_idx)
                st.time = steps[after_idx].time
                steps.insert(after_idx, st)
                changed = True
        return changed

    def _fix_resources(self, steps: List[ScriptStep]):
        # 2.1 单任务封顶
        for s in steps:
            for r in s.resources:
                avail = self.ruleset.resource.get(r.resource_type)
                if avail is not None and r.quantity > avail:
                    r.quantity = avail
        # 2.2 同时间窗口并发缩放
        groups: Dict[str, List] = defaultdict(list)
        for s in steps:
            if not s.time:
                continue
            for idx, r in enumerate(s.resources):
                groups[s.time].append((s, idx))
        for t, items in groups.items():
            by_type: Dict[str, List] = defaultdict(list)
            for s, idx in items:
                by_type[s.resources[idx].resource_type].append((s, idx))
            for rt, lst in by_type.items():
                avail = self.ruleset.resource.get(rt)
                if avail is None:
                    continue
                total = sum(s.resources[idx].quantity for s, idx in lst)
                if total <= avail:
                    continue
                scale = avail / total
                new_qs = [max(0, int(s.resources[idx].quantity * scale)) for s, idx in lst]
                diff = sum(new_qs) - avail
                order = sorted(range(len(new_qs)), key=lambda k: -new_qs[k])
                i = 0
                while diff > 0 and any(new_qs):
                    k = order[i % len(order)]
                    if new_qs[k] > 0:
                        new_qs[k] -= 1
                        diff -= 1
                    i += 1
                for (s, idx), q in zip(lst, new_qs):
                    s.resources[idx].quantity = q

