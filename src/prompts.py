"""提示词构建：生成 / 修正 / 自评审。"""
from __future__ import annotations

from rules.registry import get_rules

ACTION_SPEC = (
    "每个步骤必须包含："
    "time(执行时间，如 08:10，无则填 null)、"
    "role(执行角色)、"
    "action(执行动作的【精简概括】，仅 4 字，如“人员疏散”“启动响应”“泄漏控制”"
    "“环境监测”“医疗救护”，用于职责/时序校验，必须精简)、"
    "description(该步骤的【完整自然语言描述】，把长段落、具体措施、依据等写在这里，"
    "不要塞进 action)、"
    "resources(所需资源列表，每项含 resource_type 与 quantity)。"
)
OUTPUT_SPEC = (
    '请仅输出 JSON，格式：'
    '{"steps":[{"step_id":1,"time":"08:10","role":"应急指挥部",'
    '"action":"启动响应","description":"启动危险化学品事故应急预案，成立现场指挥部，'
    '统一指挥协调各救援组行动","resources":[{"resource_type":"消防车","quantity":2}]}]}。'
    '注意：action 必须是4 字精简动作，详细描述一律放入 description,。quantity的值为整数型，不要带单位'
)


def _fmt_resource_rules(ruleset) -> str:
    """本地资源供给清单，作为生成时的资源约束提示（始终注入）。"""
    res = getattr(ruleset, "resource", None) or {}
    if not res:
        return "（未配置本地资源供给清单）"
    lines = [f"  - {rt}"
             for rt, qty in sorted(res.items())]
    return "\n".join(lines)


def _fmt_responsibility_rules(ruleset) -> str:
    """本地职责约束：动作 -> 允许角色，作为 RAG 补充信息。"""
    pairs = getattr(ruleset, "responsibility", set()) or set()
    act2roles: dict = {}
    for role, action in pairs:
        act2roles.setdefault(action, []).append(role)
    if not act2roles:
        return "（未配置本地职责约束）"
    lines = [f"  - 动作「{a}」可由：{('、'.join(sorted(rs)))}"
             for a, rs in sorted(act2roles.items())]
    return "\n".join(lines)


def _fmt_temporal_rules(ruleset) -> str:
    """本地时序约束：偏序 (before -> after)，作为 RAG 补充信息。"""
    pairs = getattr(ruleset, "temporal", []) or []
    if not pairs:
        return "（未配置本地时序约束）"
    lines = [f"  - 「{b}」须在「{a}」之前（偏序，允许省略中间环节）"
             for b, a in sorted(pairs)]
    return "\n".join(lines)


def build_generate(scenario: dict, knowledge: str, use_rag: bool, ruleset=None):
    if ruleset is None:
        try:
            ruleset = get_rules(scenario["id"])
        except Exception:
            ruleset = None
    # 该场景职责白名单中的动作词表（精简动作），作为 action 取值建议，引导模型输出可比对动作
    try:
        allowed_actions = sorted({a for (_, a) in ruleset.responsibility})
    except Exception:
        allowed_actions = []
    action_vocab_hint = (
        "action 建议从下列【精简动作词表】中选取最贴切者（也可自创4 字动作，"
        "但务必精简，不要写长句）：" + "、".join(allowed_actions) + "。"
    ) if allowed_actions else "action 必须是 4 字精简动作，不要写长句。"

    system = (
        "你是一名资深的应急演练脚本编制专家，熟悉各类突发事件应急预案的编制规范。"
        "请依据【应急场景说明】与【知识库参考预案】（若有），生成一份结构化、可执行、"
        "时间逻辑自洽的演练执行脚本。"
        + ACTION_SPEC
        + action_vocab_hint
        + "角色与动作请尽量采用预案原文表述（例如“市应急管理局”“XX街道办事处”“群众转移”"
          "“险情巡查”等）；若预案使用规范术语则沿用规范术语。验证阶段会对同义表述做自动"
          "归一化，个别未覆盖的组合仅作提示、不会判为错误。" + OUTPUT_SPEC
    )
    user = f"【应急场景】{scenario['name']}\n{scenario['description']}\n"

    # —— 始终注入本地资源约束（生成时 resources 须遵循）——
    user += (
        "\n【本地参考资源】\n"
        f"{_fmt_resource_rules(ruleset)}\n"
    )

    if use_rag:
        # RAG 知识库参考预案（优先依据）
        user += "\n【RAG 知识库参考预案】\n"
        if knowledge:
            user += (
                f"{knowledge}\n"
                "\n生成要求：\n"
                "1) 以参考预案为准，提取其中提及的全部处置环节、责任主体（单位/小组）、"
                "时间节点与所需资源，形成脚本步骤；\n"
                "2) 不要受任何示例清单限制——步骤的数量、内容、措辞均可多于或不同于场景说明"
                "中的示例，只要预案中有依据即可纳入；\n"
                "3) 每一步务必拆分两个字段：action 用【4 字精简动作】概括，把该步骤的完整描述、具体措施、预案依据等写入"
                "description（可较长）；\n"
                "4) 预案给出时间则填 time（如 08:30），未给出时间填 null；\n"
                "5) 资源能确定类型与数量就填，不确定则 resources 留空数组；\n"

            )
        else:
            user += ("（已启用 RAG 但本次未检索到参考预案，请依据场景说明与下列本地约束生成。）\n")
        # —— RAG 场景下补充本地职责/时序约束，与参考预案结合使用 ——
        user += (
            "\n【本地约束补充（来自规则库，与参考预案结合使用，避免与预案冲突）】\n"
            "· 职责约束（允许的 角色-动作 组合）：\n"
            f"{_fmt_responsibility_rules(ruleset)}\n"
            "· 时序约束（须满足的偏序）：\n"
            f"{_fmt_temporal_rules(ruleset)}\n"
        )
    else:
        user += (
            "\n（未启用知识库检索，仅依据场景说明生成。）请生成覆盖场景说明中各项措施的完整"
            "脚本，按时间顺序排列；未给出时间的步骤 time 填 null；action 用4 字精简概括（取自"
            "上方动作词表），长描述放入 description。\n"
        )
    return system, user


def build_correct(scenario: dict, current_script_render: str, feedback: str, ruleset=None):
    system = (
        "你是应急演练脚本修正专家。请根据规则验证器给出的结构化反馈，对脚本做【定向修正】："
        "只修改出错的步骤，未出错的步骤必须原样保留，以最小化改动范围。" + OUTPUT_SPEC
    )
    user = (
        f"【应急场景】{scenario['name']}\n\n"
        f"【当前脚本】\n{current_script_render}\n\n"
        f"【验证器反馈（需修正的问题）】\n{feedback}\n\n"
        f"请输出修正后的完整脚本 JSON（包含所有步骤，保持未出错步骤不变）。"
    )
    # —— 注入本地职责约束，供修正「职责错配 / 职责待复核」步骤时参考 ——
    #    大量修正来源于职责错误，故将合法 角色-动作 映射提供给大模型，便于把动作改派给法定角色。
    if ruleset is not None:
        resp_text = _fmt_responsibility_rules(ruleset)
        user += (
            "\n【本地职责约束（修正参考，来自规则库）】\n"
            "若反馈中出现「职责错配 / 职责待复核」类问题，请严格依据下列合法 角色-动作 映射，"
            "将动作改派给其法定角色（优先沿用白名单中的规范角色名，避免自造未收录角色）：\n"
            f"{resp_text}\n"
        )
    return system, user


def build_self_review(current_script_render: str):
    system = (
        "你是严格的应急演练逻辑审查员。请检查给定脚本在时序、职责、资源三方面的逻辑一致性，"
        "找出仍存在的错误。若没有错误，needs_fix 设为 false。"
        '输出 JSON：{"needs_fix": true/false, "issues": "错误简述", '
        '"steps": [修正后的完整步骤数组]}。'
        + OUTPUT_SPEC
        + "注意：steps 必须是「对象数组」，每个元素是含 step_id、time、role、action、"
          "description、resources 字段的对象（与生成格式一致），【严禁把步骤写成纯文本字符串】；"
          "action 用 2-4 字精简动作，长描述放入 description；"
          "若无需修正则原样保留当前脚本的所有步骤。"
    )
    user = f"【待审查脚本】\n{current_script_render}\n"
    return system, user
