"""资源约束验证器：单任务需求<=供给；同时间窗口并发任务总需求<=供给。"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from models import ErrorRecord, ResourceDemand, Script
from .base import Validator


class ResourceValidator(Validator):
    def __init__(self, supply: Dict[str, int]):
        self.supply = supply  # 资源类型 -> 供给上限

    def constraint_count(self, script: Script) -> int:
        per_task = sum(len(s.resources) for s in script.steps)
        # 并发聚合：每个"时间窗口×资源类型"计 1 条约束
        groups: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in script.steps:
            if not s.time:
                continue
            for r in s.resources:
                groups[s.time][r.resource_type] += r.quantity
        concurrency = sum(1 for g in groups.values() for _ in g)
        return per_task + concurrency

    def validate(self, script: Script) -> List[ErrorRecord]:
        errors: List[ErrorRecord] = []

        # 0) 资源类型不在本地定义的资源约束清单中 -> 资源配置错误（硬错误）
        #    “我们提供的资源约束”即 supply 字典的键集合；step 用到清单外的资源即判错。
        for s in script.steps:
            for r in s.resources:
                if r.resource_type not in self.supply:
                    errors.append(ErrorRecord(
                        error_type="resource",
                        step_id=s.step_id,
                        location=f"步骤{s.step_id} 资源'{r.resource_type}'",
                        description=(
                            f"资源类型'{r.resource_type}'不在本地定义的资源约束清单中"
                            f"（合法资源类型：{', '.join(sorted(self.supply))}），"
                            f"属资源配置错误。"
                        ),
                        violated_rule="资源约束:资源类型必须在约束清单内",
                        severity=2,
                    ))

        # 1) 单任务资源超配
        for s in script.steps:
            for r in s.resources:
                avail = self.supply.get(r.resource_type)
                if avail is None:
                    continue
                if r.quantity > avail:
                    errors.append(ErrorRecord(
                        error_type="resource",
                        step_id=s.step_id,
                        location=f"步骤{s.step_id} 资源'{r.resource_type}'",
                        description=f"单任务资源超配：需求 {r.quantity} > 可用 {avail}。",
                        violated_rule=f"资源:{r.resource_type}<= {avail}",
                        severity=2,
                    ))

        # 2) 同时间窗口并发资源竞争
        groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        step_of_time: Dict[str, int] = {}
        for s in script.steps:
            if not s.time:
                continue
            step_of_time.setdefault(s.time, s.step_id)
            for r in s.resources:
                groups[s.time][r.resource_type].append(r.quantity)

        for t, res_map in groups.items():
            for rt, qs in res_map.items():
                avail = self.supply.get(rt)
                if avail is None:
                    continue
                total = sum(qs)
                if total > avail:
                    errors.append(ErrorRecord(
                        error_type="resource",
                        step_id=step_of_time.get(t, 0),
                        location=f"时间窗口 {t} 资源'{rt}'",
                        description=f"并发资源竞争：窗口内总需求 {total} > 可用 {avail}。",
                        violated_rule=f"资源并发:{rt}<= {avail}",
                        severity=2,
                    ))
        return errors
