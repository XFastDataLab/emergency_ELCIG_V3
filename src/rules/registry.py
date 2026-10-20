"""规则注册表：合并全局与分场景规则，输出校验器/修正器所需的 RuleSet。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from . import responsibility as R
from . import resource as RES
from . import temporal as T


@dataclass
class RuleSet:
    scenario_id: str
    temporal: List[Tuple[str, str]] = field(default_factory=list)          # 偏序规则
    responsibility: Set[Tuple[str, str]] = field(default_factory=set)        # 合法 (role, action)
    responsibility_by_action: Dict[str, Set[str]] = field(default_factory=dict)
    responsibility_conflicts: Set[Tuple[str, str]] = field(default_factory=set)  # 明确矛盾 (role, action)
    responsibility_by_role: Dict[str, Set[str]] = field(default_factory=dict)     # 角色 -> 其合法职责动作集合
    resource: Dict[str, int] = field(default_factory=dict)                  # 资源类型 -> 供给上限

    def allowed_roles_for(self, action: str) -> Set[str]:
        return self.responsibility_by_action.get(action, set())

    def allowed_actions_for(self, role: str) -> Set[str]:
        return self.responsibility_by_role.get(role, set())


def _expand_responsibility_entries(entries) -> List[Tuple[str, str]]:
    """兼容两种写法：
      * (role, action)         —— 单职责单条
      * (role, (a1, a2, ...))  —— 一个部门对应多个职责（合并写法）
    统一展开为 (role, action) 二元组列表。"""
    out: List[Tuple[str, str]] = []
    for entry in entries:
        role = entry[0]
        actions = entry[1]
        if isinstance(actions, (tuple, list, set)):
            out.extend((role, a) for a in actions)
        else:
            out.append((role, actions))
    return out


def _build_responsibility_by_action(pairs) -> Dict[str, Set[str]]:
    """动作 -> 可执行该动作的部门集合（反向索引，供修正器使用）。"""
    d: Dict[str, Set[str]] = {}
    for role, action in pairs:
        d.setdefault(action, set()).add(role)
    return d


def _build_responsibility_by_role(pairs) -> Dict[str, Set[str]]:
    """部门 -> 其全部合法职责动作集合（正向索引，供职责校验使用）。

    一个部门对应多个职责；校验时只要某步动作命中其中「任意一个」即视为合法、通过。
    """
    d: Dict[str, Set[str]] = {}
    for role, action in pairs:
        d.setdefault(role, set()).add(action)
    return d


def get_rules(scenario_id: str) -> RuleSet:
    """合并全局规则与指定场景的专属规则。"""
    # 合并全局与分场景时序规则，并去重（避免同一 (before,after) 重复报告）
    _seen = set()
    temporal: List[Tuple[str, str]] = []
    for rule in list(T.GLOBAL_TEMPORAL) + list(T.SCENARIO_TEMPORAL.get(scenario_id, [])):
        if rule not in _seen:
            _seen.add(rule)
            temporal.append(rule)

    all_resp_entries = (list(R.GLOBAL_RESPONSIBILITY)
                        + list(R.SCENARIO_RESPONSIBILITY.get(scenario_id, [])))
    resp_pairs = set(_expand_responsibility_entries(all_resp_entries))

    conflicts: Set[Tuple[str, str]] = set(R.GLOBAL_RESPONSIBILITY_CONFLICTS)
    conflicts |= set(R.SCENARIO_RESPONSIBILITY_CONFLICTS.get(scenario_id, []))

    resource = dict(RES.GLOBAL_RESOURCE)
    resource.update(RES.SCENARIO_RESOURCE.get(scenario_id, {}))

    return RuleSet(
        scenario_id=scenario_id,
        temporal=temporal,
        responsibility=resp_pairs,
        responsibility_by_action=_build_responsibility_by_action(resp_pairs),
        responsibility_by_role=_build_responsibility_by_role(resp_pairs),
        responsibility_conflicts=conflicts,
        resource=resource,
    )


def GLOBAL_RESPONSIBILITY_PAIRS() -> List[Tuple[str, str]]:
    """展开全局职责映射为 (role, action) 二元组列表（供白名单/本体使用）。"""
    return _expand_responsibility_entries(R.GLOBAL_RESPONSIBILITY)


# ---------------------------------------------------------------------------
# 动作 / 角色 标准化别名映射（把模型自由表述映射到规则中的规范名称）
# ---------------------------------------------------------------------------
ACTION_ALIASES = {
    "报告事故": "事故上报", "上报事故": "事故上报", "向主管部门报告": "事故上报",
    "通报事故": "信息通报", "发布信息": "信息通报",
    "启动应急预案": "启动响应", "宣布响应": "启动响应", "拉响警报": "启动响应",
    "队伍集结": "应急队伍集结", "集结队伍": "应急队伍集结",
    "调配资源": "资源调配", "调拨物资": "资源调配",
    "组织疏散": "人员疏散", "撤离群众": "人员疏散", "疏散群众": "人员疏散",
    "灭火": "灭火处置", "扑救": "灭火处置", "消防灭火": "灭火处置",
    "封堵泄漏": "泄漏控制", "控制泄漏": "泄漏控制", "堵漏": "泄漏控制",
    "切断源": "污染源切断", "切断污染源": "污染源切断",
    "洗消": "防化洗消", "洗消作业": "防化洗消", "去污": "防化洗消",
    "监测环境": "环境监测", "环境检测": "环境监测",
    "评估污染": "污染评估",
    "救治伤员": "医疗救护", "现场急救": "医疗救护",
    "转运伤员": "伤员转运", "送医": "伤员转运",
    "防疫": "卫生防疫",
    "搜救": "人员搜救", "搜寻幸存者": "人员搜救", "救援": "救援搜救",
    "警戒": "现场警戒", "设卡": "现场警戒",
    "管制交通": "交通管制", "封路": "交通管制",
    "处置险情": "险情处置", "排险": "险情处置",
    "排涝": "排涝作业", "抽水": "排涝作业",
    "抢修": "设施抢修", "设备抢修": "设施抢修",
    "安置": "安置救助", "开放安置点": "安置点开放",
    "回港": "船只回港", "渔船回港": "船只回港",
    "转移": "人员转移", "转移群众": "人员转移",
    "巡查": "险情巡查", "巡堤": "险情巡查",
    "监测险情": "险情监测", "监测隐患": "隐患监测",
    "预警": "台风预警", "发布预警": "台风预警",
    "排查": "危险区排查", "隐患排查": "危险区排查",
    "先期处置": "先期处置", "指挥": "指挥协调",
    # 常见口语/泛化变体（精确匹配）
    "扑灭火势": "灭火处置", "扑火": "灭火处置", "灭火救援": "灭火处置", "消防作业": "灭火处置",
    "人员撤离": "人员疏散", "群众疏散": "人员疏散", "组织撤离": "人员疏散", "疏散民众": "人员疏散",
    "人员转运": "伤员转运", "伤员后送": "伤员转运",
    "环境监控": "环境监测", "环保监测": "环境监测", "环境监视": "环境监测",
    "污染检测": "污染评估", "损害评估": "污染评估",
    "现场救援": "救援搜救", "搜救人员": "人员搜救", "搜救幸存者": "人员搜救",
    "交通疏导": "交通管制", "道路管制": "交通管制", "封路管制": "交通管制",
    "设施维护": "设施抢修", "设备维护": "设施抢修", "抢修设备": "设施抢修",
    "群众转移": "人员转移", "转移群众": "人员转移", "人员撤离转移": "人员转移",
    "危险排查": "危险区排查", "隐患排除": "危险区排查",
    "医疗救治": "医疗救护", "抢救伤员": "医疗救护", "现场抢救": "医疗救护",
    "发布台风预警": "台风预警", "台风预报": "台风预警", "台风预警发布": "台风预警",
    "发布灾害预警": "灾害预警", "灾害预报": "灾害预警",
    "发布信息通报": "信息通报", "情况通报": "信息通报",
    # 预案原文常见机构/阶段表述（精确匹配，便于 RAG 接地后归一化）
    "安置群众": "安置救助", "开设安置点": "安置点开放",
    "物资保障": "物资供应", "物资调配": "资源调配",
    "供电保障": "电力保障", "保电": "电力保障",
    "通信保障": "通信保障", "通保": "通信保障",
    "舆情引导": "舆情管控", "舆论管控": "舆情管控", "宣传": "宣传引导",
    "灾情监测": "险情监测",
}

ROLE_ALIASES = {
    "企业": "企业负责人", "公司": "企业负责人", "工厂负责人": "企业负责人",
    "指挥部": "应急指挥部", "应急办": "应急指挥部", "总指挥": "应急指挥部",
    "消防": "消防救援组", "消防队": "消防救援组",
    "医护": "医疗救护组", "医疗组": "医疗救护组", "医生": "医疗救护组",
    "环保": "环境监测组", "监测组": "环境监测组",
    "防化": "防化处置组", "洗消组": "防化处置组",
    "交警": "交通管制组", "公安": "交通管制组", "警察": "交通管制组",
    "疏散组": "疏散引导组", "引导组": "疏散引导组",
    "工程": "工程抢险组", "抢险队": "工程抢险组",
    "海事": "海事/海上救援", "海上救援": "海事/海上救援",
    "气象": "气象/监测组", "气象台": "气象/监测组",
    "社区": "社区/基层", "街道": "社区/基层", "村委": "社区/基层",
    # 常见口语/泛化变体（精确匹配，避免误伤具体机构名）
    "消防队员": "消防救援组", "消防官兵": "消防救援组", "消防战士": "消防救援组",
    "医务人员": "医疗救护组", "医护人员": "医疗救护组", "白衣天使": "医疗救护组",
    "环保人员": "环境监测组", "环保部门": "环境监测组",
    "交警部门": "交通管制组", "交警支队": "交通管制组", "交巡警": "交通管制组",
    "社区工作人员": "社区/基层", "居委会": "社区/基层", "街道办事处": "社区/基层",
    "企业": "企业负责人", "公司负责人": "企业负责人", "工厂负责人": "企业负责人",
    # 预案原文常见责任主体（精确匹配，避免误伤具体机构名）
    "应急管理局": "应急指挥部", "应急管理部": "应急指挥部", "应急管理部门": "应急指挥部",
    "消防救援支队": "消防救援组", "消防救援大队": "消防救援组", "消防总队": "消防救援组",
    "卫生院": "医疗救护组", "医院": "医疗救护组", "医疗机构": "医疗救护组",
    "生态环境局": "环境监测组", "生态环境部门": "环境监测组",
    "气象局": "气象/监测组", "气象部门": "气象/监测组",
    "交通局": "交通管制组", "交通运输局": "交通管制组",
    "街道办": "社区/基层", "乡镇": "社区/基层", "乡政府": "社区/基层", "镇政府": "社区/基层",
    "企业安全部门": "企业负责人", "安环部": "企业负责人",
}


def _extract_known_action(text: str) -> str:
    """从一段较长描述中，提取最长的“可识别动作短语”并映射为规范名（兜底用）。

    候选短语 = 全部别名键 + 全部别名规范值。命中最长子串，避免“洗消”误截“防化洗消”等。
    """
    text = text or ""
    if not text:
        return ""
    best = ""
    best_canon = ""
    for phrase in set(ACTION_ALIASES.keys()) | set(ACTION_ALIASES.values()):
        if not phrase:
            continue
        if phrase in text and len(phrase) > len(best):
            best = phrase
            best_canon = ACTION_ALIASES.get(phrase, phrase)
    return best_canon


def normalize_action(name: str) -> str:
    if not name:
        return name
    a = name.strip()
    # 1) 精确别名匹配（含规范名本身作为键的恒等映射）
    if a in ACTION_ALIASES:
        return ACTION_ALIASES[a]
    # 2) 长串 / 含分隔符 -> 尝试从原文中提取可识别的精简动作（防止长描述直接当 action 比对失效）
    if len(a) > 4 or any(sep in a for sep in ("，", "、", "；", "。", ",", ";")):
        cand = _extract_known_action(a)
        if cand:
            return cand
    return a


def normalize_role(name: str) -> str:
    if not name:
        return name
    return ROLE_ALIASES.get(name.strip(), name.strip())


# ---------------------------------------------------------------------------
# 动作本体（我们“提供的标准动作”集合）：供生成期语义接地 / 验证器职责比对使用
# ---------------------------------------------------------------------------
def _build_canon_to_aliases() -> Dict[str, Set[str]]:
    """反向别名表：规范动作 -> 其全部别名表述集合（用于语义相似度计算）。"""
    d: Dict[str, Set[str]] = {}
    for alias, canon in ACTION_ALIASES.items():
        d.setdefault(canon, set()).add(alias)
    return d


CANON_TO_ALIASES = _build_canon_to_aliases()


def get_action_ontology(scenario_id: str) -> Set[str]:
    """构造“我们提供的标准动作”本体（用于生成期语义映射 / 职责白名单比对）。

    来源 = 全局 + 分场景职责规则中的全部规范动作名，并合并别名映射的规范值，
    确保模型以任意同义表述生成时，都能映射回标准动作体系。
    """
    actions: Set[str] = set()
    for _, a in GLOBAL_RESPONSIBILITY_PAIRS():
        actions.add(a)
    for entry in R.SCENARIO_RESPONSIBILITY.get(scenario_id, []):
        act_set = entry[1]
        if isinstance(act_set, (tuple, list, set)):
            actions.update(act_set)
        else:
            actions.add(act_set)
    # 别名规范值（如“消防灭火”“现场抢险”等未直接进白名单、但属标准动作体系）
    actions |= set(ACTION_ALIASES.values())
    return actions
