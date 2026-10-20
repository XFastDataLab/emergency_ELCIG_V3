"""Independent synthetic ground truth and rule-validator metrics.

Labels are constructed from controlled mutations BEFORE the tested validator runs.
These are template-based artificial cases, not independently expert-annotated field data.
"""
from __future__ import annotations
import copy
from collections import Counter
from models import ResourceDemand, Script, ScriptStep
from rules.registry import get_rules
from scenarios import get_scenario
from validator.composite import RuleValidator
from audit import emit, write_json


def error_reduction_rate(initial, final):
    # With no initial errors reduction is undefined; retain a numeric legacy field,
    # but report zero (no reduction), never claim a 100% improvement.
    if initial == 0:
        return 0.0
    return round((initial-final)/initial, 4)


def constraint_satisfaction_rate(total, errors):
    return round((total-errors)/total, 4) if total else 1.0


def ledc(detected, actual):
    return round(detected/actual, 4) if actual else None


def _label(kind, step_id, rule):
    return {"error_type": kind, "step_id": step_id, "violated_rule": rule}


def _complete_base(sid):
    """Close prerequisites and topologically order a manually specified scenario base."""
    rules = get_rules(sid)
    steps = [ScriptStep.from_dict(x) for x in get_scenario(sid)["correct_base"]]
    by_action = {s.action: s for s in steps}
    changed = True
    while changed:
        changed = False
        for before, after in rules.temporal:
            if after in by_action and before not in by_action:
                roles = sorted(rules.allowed_roles_for(before))
                if not roles:
                    raise ValueError(f"缺少基础动作职责定义: {before}")
                by_action[before] = ScriptStep(0, role=roles[0], action=before)
                changed = True
    ordered = []
    while by_action:
        ready = [a for a in by_action if not any(b in by_action for b,c in rules.temporal if c == a)]
        if not ready:
            raise ValueError("基础时序规则存在环，无法构造对照样本")
        for action in ready:
            ordered.append(by_action.pop(action))
    for i, step in enumerate(ordered, 1):
        step.step_id = i
        step.resources = []
    return ordered


# Unequal, predeclared injection amounts; never tune labels to detector outputs.
VALIDATOR_INJECTION_PLAN = {
    "S1": {"temporal_pairs": (2, 3), "responsibility_steps": (4, 5),
           "resource_single_tasks": 3, "resource_unknown_types": 1},
    "S2": {"temporal_pairs": (3, 2), "responsibility_steps": (5, 4),
           "resource_single_tasks": 2, "resource_unknown_types": 2},
    "S3": {"temporal_pairs": (3, 4), "responsibility_steps": (4, 6),
           "resource_single_tasks": 3, "resource_unknown_types": 2},
}


def _injection_base(sid):
    """A valid topological order with adjacent prerequisite pairs for local edits."""
    rules = get_rules(sid)
    pending = {s.action: s for s in _complete_base(sid)}
    original = {action: i for i, action in enumerate(pending)}
    ordered = []
    while pending:
        ready = [a for a in pending if not any(b in pending for b, c in rules.temporal if c == a)]
        previous = ordered[-1].action if ordered else None
        action = min(ready, key=lambda a: (0 if (previous, a) in rules.temporal else 1, original[a]))
        ordered.append(pending.pop(action))
    for index, step in enumerate(ordered, 1):
        step.step_id = index
    return ordered


def build_validator_cases(sid, count=12):
    """Eight profiles with local temporal swaps, multiple duty/resource mutations.

    Each swapped disjoint adjacent prerequisite pair violates exactly one edge.
    Duty edits use different steps; resource edits cover excess, unknown type and
    true concurrency in each resource profile. Labels precede detector execution.
    """
    rules = get_rules(sid)
    plan = VALIDATOR_INJECTION_PLAN[sid]
    base = _injection_base(sid)
    role_candidates = []
    for step in base:
        roles = [role for role, action in sorted(rules.responsibility_conflicts)
                 if action == step.action and role not in rules.allowed_roles_for(action)]
        if roles:
            role_candidates.append((step.action, roles))
    if len(role_candidates) < max(plan["responsibility_steps"]):
        raise ValueError(f"{sid} 没有可用职责冲突样本")
    pairs = []
    for i in range(len(base) - 1):
        if (base[i].action, base[i+1].action) in rules.temporal and (not pairs or i > pairs[-1] + 1):
            pairs.append(i)
    if len(pairs) < max(plan["temporal_pairs"]):
        raise ValueError(f"{sid} 可独立交换的相邻时序规则不足，需要审核注入方案")
    cases = []
    for index in range(count):
        profile, variant = index % 8, index // 8
        steps = copy.deepcopy(base)
        offset = 360 + variant % 240
        for i,s in enumerate(steps):
            minute = offset + i*3
            s.time = f"{minute//60:02d}:{minute%60:02d}"
        expected, edits = [], []
        if profile & 1:
            selected = [pairs[(variant + j) % len(pairs)] for j in range(plan["temporal_pairs"][variant % 2])]
            changed_edges = [(base[i].action, base[i+1].action) for i in selected]
            for i in selected:
                if variant % 2:
                    steps[i], steps[i+1] = steps[i+1], steps[i]
                else:
                    steps[i].time, steps[i+1].time = steps[i+1].time, steps[i].time
            for i, step in enumerate(steps, 1):
                step.step_id = i
            edits.append("local_step_swaps" if variant % 2 else "local_timestamp_swaps")
            ids = {s.action:s.step_id for s in steps}
            expected += [_label("temporal", ids[after], f"时序:{before}→{after}")
                         for before, after in changed_edges]
        if profile & 2:
            for slot in range(plan["responsibility_steps"][variant % 2]):
                action, roles = role_candidates[(profile + variant + slot) % len(role_candidates)]
                role = roles[(variant + slot) % len(roles)]
                step = next(s for s in steps if s.action == action)
                step.role = role
                expected.append(_label("responsibility", step.step_id, f"职责:({role},{step.action})"))
            edits.append("multiple_explicit_responsibility_conflicts")
        rt = sorted(rules.resource)[variant % len(rules.resource)]
        cap = rules.resource[rt]
        if profile & 4:
            resource_types = sorted(rules.resource)
            for slot in range(plan["resource_single_tasks"]):
                rt = resource_types[(profile + variant * 3 + slot) % len(resource_types)]
                cap = rules.resource[rt]
                step = steps[(profile + variant + slot) % len(base)]
                step.resources.append(ResourceDemand(rt, cap + 1 + variant))
                expected += [_label("resource", step.step_id, f"资源:{rt}<= {cap}"),
                             _label("resource", step.step_id, f"资源并发:{rt}<= {cap}")]
            edits.append("multiple_single_task_over_capacity")
            for slot in range(plan["resource_unknown_types"]):
                step = steps[(profile + variant + plan["resource_single_tasks"] + slot) % len(base)]
                step.resources.append(ResourceDemand(f"未登记测试装备{slot+1}", 1+variant))
                expected.append(_label("resource", step.step_id, "资源约束:资源类型必须在约束清单内"))
            edits.append("unknown_resource_types")
            # Use a distinct resource and two legal repeated actions at a new time.
            rt = resource_types[(profile + variant * 3 + plan["resource_single_tasks"]) % len(resource_types)]
            cap = rules.resource[rt]
            start_id = len(steps)+1
            for j in range(2):
                extra = copy.deepcopy(base[-1])
                extra.step_id, extra.time = start_id+j, "23:00"
                extra.resources = [ResourceDemand(rt, cap//2+1)]
                steps.append(extra)
            expected.append(_label("resource", start_id, f"资源并发:{rt}<= {cap}"))
            edits.append("two_task_concurrency")
        else:
            # Boundary-safe controls also exercise resource checking at capacity.
            steps[-1].resources = [ResourceDemand(rt, cap)]
        script = Script(sid, steps)
        script.raw_text = script.render()
        cases.append({"sample_id": f"{sid}-{index+1:04d}", "scenario": sid,
            "profile": profile, "variant": variant, "mutations": edits,
            "dataset_version": "local-multiple-errors-v2", "injection_plan": copy.deepcopy(plan),
            "annotation_source": "人工设计错误模板的确定性参数化标注，未经人工逐条复核",
            "script": script.to_dict(), "expected_errors": expected})
    return cases


def build_resource_boundary_case():
    """One independently labeled stress case; do not consult detector output.

    Two additive allocations of the same resource exceed a single-task capacity.
    Missing time is permitted by ScriptStep; it must not erase that capacity limit.
    """
    sid, resource_type = "S1", "消防车"
    capacity = get_rules(sid).resource[resource_type]
    quantity = capacity // 2 + 1
    steps = _complete_base(sid)
    for i, step in enumerate(steps):
        minute = 480 + i * 3
        step.time = f"{minute // 60:02d}:{minute % 60:02d}"
    target = next(s for s in steps if s.role == "消防救援组" and s.action == "灭火处置")
    target.time = None
    target.resources = [ResourceDemand(resource_type, quantity),
                        ResourceDemand(resource_type, quantity)]
    target.description = (f"同一次灭火任务同时部署两支队伍：A区申请消防车{quantity}辆，"
                          f"B区另申请消防车{quantity}辆，两批独立追加、不能共用；"
                          f"合计{2 * quantity}辆，现有库存{capacity}辆。未填写具体执行时间。")
    script = Script(sid, steps)
    script.raw_text = script.render()
    return {"sample_id": "S1-B001", "scenario": sid,
        "profile": "resource_boundary", "variant": 0,
        "mutations": ["split_same_resource_in_one_step", "omit_step_time"],
        "annotation_source": "人工设计的资源聚合边界用例，标注在验证器运行前按总量计算",
        "annotation_reason": f"同一步骤消防车需求应合并：{quantity}+{quantity}={2 * quantity}>{capacity}，违反单任务资源上限。",
        "miss_reason": "单任务检查未合并同类资源条目；时间为空的步骤又跳过并发聚合。",
        "boundary_case": True,
        "script": script.to_dict(),
        "expected_errors": [_label("resource", target.step_id, f"资源:{resource_type}<= {capacity}")]}


def _boundary_base(sid):
    steps = _complete_base(sid)
    for i, step in enumerate(steps):
        minute = 480 + i * 3
        step.time = f"{minute // 60:02d}:{minute % 60:02d}"
    return steps


def build_temporal_boundary_case():
    sid = "S2"
    steps = _boundary_base(sid)
    if any(s.action == "影响解除" for s in steps):
        raise ValueError("台风边界基础脚本已包含影响解除，需要重新审核测试标注")
    role = sorted(get_rules(sid).allowed_roles_for("渔船出港"))[0]
    target = ScriptStep(len(steps) + 1, time="10:00", role=role, action="渔船出港",
                        description="台风影响尚未解除，提前组织渔船出港；本步骤不满足影响解除这一必要前置条件。")
    steps.append(target)
    script = Script(sid, steps)
    script.raw_text = script.render()
    return {"sample_id": "S2-B001", "scenario": sid,
        "profile": "temporal_boundary", "variant": 0,
        "mutations": ["omit_required_release_condition"], "boundary_case": True,
        "annotation_source": "人工设计的必要前置条件边界用例，标注先于检测生成",
        "annotation_reason": "台风影响未解除便组织渔船出港，违反影响解除→渔船出港的必要前置条件。",
        "miss_reason": "当前temporal_strict=false将所有缺失前置都降为提示，未区分必要条件与可省略环节；提示不计入检出错误。严格模式会检出。",
        "script": script.to_dict(),
        "expected_errors": [_label("temporal", target.step_id, "时序:影响解除→渔船出港")]}


def build_overlap_boundary_case():
    sid, resource_type = "S3", "挖掘机"
    capacity = get_rules(sid).resource[resource_type]
    quantity = capacity // 2 + 1
    steps = _boundary_base(sid)
    origin = next(s for s in steps if s.action == "救援搜救")
    first_id = len(steps) + 1
    for start, end, zone in (("10:00", "10:30", "A"), ("10:05", "10:35", "B")):
        extra = copy.deepcopy(origin)
        extra.step_id, extra.time = len(steps) + 1, start
        extra.resources = [ResourceDemand(resource_type, quantity)]
        extra.description = f"{zone}区任务从{start}至{end}连续独占挖掘机{quantity}台，期间不能调拨或共用。"
        steps.append(extra)
    script = Script(sid, steps)
    script.raw_text = script.render()
    return {"sample_id": "S3-B001", "scenario": sid,
        "profile": "resource_overlap_boundary", "variant": 0,
        "mutations": ["overlapping_occupation_with_different_start_times"], "boundary_case": True,
        "annotation_source": "人工设计的资源占用区间边界用例，标注先于检测生成",
        "annotation_reason": f"10:05至10:30两项任务同时占用挖掘机：{quantity}+{quantity}={2 * quantity}>{capacity}。",
        "miss_reason": "并发检查只合并完全相同的开始时间字符串，不读取描述中的结束时间和占用时长，漏掉不同开始时间下的区间重叠。",
        "script": script.to_dict(),
        "expected_errors": [_label("resource", first_id, f"资源并发:{resource_type}<= {capacity}")]}


def build_boundary_cases(sid):
    builders = {"S1": build_resource_boundary_case, "S2": build_temporal_boundary_case,
                "S3": build_overlap_boundary_case}
    return [builders[sid]()]


def _key(error):
    return error["error_type"], error["step_id"], error["violated_rule"]


def _error_counts(injected, detected, matched):
    """Keep readable error quantities and the paper's LEDR metric only.

    Additional detections never inflate the number of detected injected errors.
    """
    return {"injected_errors": injected, "detected_errors": detected,
            "detected_injected_errors": matched, "missed_errors": injected - matched,
            "ledr": round(matched / injected, 4) if injected else None}


def _unmatched_details(errors, unmatched):
    remaining = unmatched.copy()
    details = []
    for error in errors:
        key = _key(error)
        if remaining[key]:
            details.append(error)
            remaining[key] -= 1
    return details


def validator_dataset_effectiveness(sid, count=12, strict=False, temporal_strict=False, output_dir=None,
                                    extra_cases=None):
    cases = build_validator_cases(sid, count)
    for case in extra_cases or []:
        if case["scenario"] != sid:
            raise ValueError("附加样本的场景与当前评测场景不一致")
        cases.append(copy.deepcopy(case))
    count = len(cases)
    if output_dir is not None:
        write_json(output_dir/"dataset.json", cases)  # Frozen before any detector call.
    validator = RuleValidator(get_rules(sid), strict=strict, temporal_strict=temporal_strict)
    totals = {k: Counter(injected=0,detected=0,matched=0) for k in ("temporal","responsibility","resource")}
    clean, details = 0,[]
    for case in cases:
        script = Script(sid, [ScriptStep.from_dict(x) for x in case["script"]["steps"]])
        detected = [e.to_dict() for e in validator.validate(script)]
        pred = Counter(_key(e) for e in detected if e["error_type"] in totals)
        gold = Counter(_key(e) for e in case["expected_errors"])
        matched = pred & gold
        missed, additional = gold - pred, pred - gold
        for name in totals:
            for metric, items in (("injected",gold),("detected",pred),("matched",matched)):
                totals[name][metric] += sum(n for k,n in items.items() if k[0] == name)
        if not gold:
            clean += 1
        row = {**case, "detected_errors": detected,
               "injected_error_count": sum(gold.values()),
               "detected_error_count": sum(pred.values()),
               "detected_injected_errors": sum(matched.values()),
               "missed_errors": sum(missed.values()),
               "missed_error_details": _unmatched_details(case["expected_errors"], missed),
               "additional_error_details": _unmatched_details(detected, additional)}
        details.append(row)
        emit("validator_case", **row)
    all_counts = Counter(injected=0,detected=0,matched=0)
    for counts in totals.values():
        all_counts.update(counts)
    return {"scenario": sid, "scenario_name": get_scenario(sid)["name"],
        "samples": count, "clean_samples": clean, "error_samples": count-clean,
        **_error_counts(**all_counts),
        "by_type": {k: _error_counts(**v) for k,v in totals.items()},
        "strict": strict, "temporal_strict": temporal_strict,
        "cases": details}
