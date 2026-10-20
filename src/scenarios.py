"""测试场景定义（论文第5章：S1 危化品泄漏 / S2 台风灾害 / S3 地质灾害）。

每个场景包含：
  - meta：标识与用于 RAG 检索/LLM 生成的场景描述
  - naive_draft：内置的"有缺陷初稿"（含时序/职责/资源错误），供离线模板生成器演示迭代修正
  - correct_base：满足全部约束的标准脚本，供"验证器有效性实验"注入已知错误、计算 LEDC
"""
from __future__ import annotations

from typing import Dict, List


def _step(step_id, time, role, action, resources=None):
    return {
        "step_id": step_id,
        "time": time,
        "role": role,
        "action": action,
        "resources": resources or [],
    }


SCENARIOS: Dict[str, dict] = {
    "S1": {
        "id": "S1",
        "name": "危险化学品泄漏",
        "emergency_type": "hazmat",
        "description": (
            "某化工企业储罐区发生易燃有毒化学品泄漏，需开危化品泄露演练，请生成演练脚本"
            "参考提供预案资料，包括但不限于以上措施。"
        ),
        "naive_draft": [
            _step(1, "08:00", "企业负责人", "启动响应"),                       # 时序错误：事故上报应先于启动响应
            _step(2, "08:10", "企业负责人", "事故上报"),
            _step(3, "08:20", "消防救援组", "环境监测"),                       # 职责错误：环境监测属环境监测组
            _step(4, "08:30", "应急指挥部", "人员疏散"),
            _step(5, "08:40", "防化处置组", "泄漏控制", [{"resource_type": "消防车", "quantity": 6}]),  # 资源超配(单任务): 6>5
            _step(6, "08:40", "消防救援组", "灭火处置", [{"resource_type": "消防车", "quantity": 3}]),  # 资源竞争(并发): 6+3>5
            _step(7, "08:50", "环境监测组", "环境监测"),
            # 缺：污染源切断（泄漏控制的前置）
        ],
        "correct_base": [
            _step(1, "08:00", "企业负责人", "事故上报"),
            _step(2, "08:05", "企业负责人", "信息通报"),
            _step(3, "08:10", "应急指挥部", "启动响应"),
            _step(4, "08:20", "应急指挥部", "资源调配"),
            _step(5, "08:25", "防化处置组", "污染源切断"),
            _step(6, "08:30", "防化处置组", "泄漏控制", [{"resource_type": "消防车", "quantity": 3}]),
            _step(7, "08:35", "消防救援组", "灭火处置", [{"resource_type": "消防车", "quantity": 2}]),
            _step(8, "08:40", "疏散引导组", "人员疏散"),
            _step(9, "08:50", "环境监测组", "环境监测"),
            _step(10, "09:00", "防化处置组", "防化洗消", [{"resource_type": "防化装备", "quantity": 4}]),
        ],
    },
    "S2": {
        "id": "S2",
        "name": "台风灾害",
        "emergency_type": "typhoon",
        "description": (
            "强台风即将登陆，沿海区域需开展预警发布、启动响应、船只回港、人员转移、"
            "险情巡查、排涝作业、安置救助与交通管制等防御工作。"
            "（以预案为准，生成的步骤可包括但不限于上述措施）"
        ),
        "naive_draft": [
            _step(1, "09:00", "应急指挥部", "启动响应"),                       # 时序错误：台风预警应先于启动响应
            _step(2, "09:10", "海事/海上救援", "船只回港"),
            _step(3, "09:20", "社区/基层", "人员转移"),
            _step(4, "09:30", "疏散引导组", "安置点开放"),
            _step(5, "09:40", "工程抢险组", "排涝作业", [{"resource_type": "排水泵", "quantity": 10}]),  # 资源超配: 10>8
            _step(6, "09:40", "工程抢险组", "险情处置", [{"resource_type": "排水泵", "quantity": 4}]),  # 资源竞争: 14>8
            # 缺：台风预警（启动响应的前置）
        ],
        "correct_base": [
            _step(1, "08:30", "气象/监测组", "台风预警"),
            _step(2, "09:00", "应急指挥部", "启动响应"),
            _step(3, "09:10", "海事/海上救援", "船只回港"),
            _step(4, "09:20", "社区/基层", "人员转移"),
            _step(5, "09:30", "疏散引导组", "安置点开放"),
            _step(6, "09:35", "社区/基层", "险情巡查"),
            _step(7, "09:40", "工程抢险组", "排涝作业", [{"resource_type": "排水泵", "quantity": 5}]),
            _step(8, "09:45", "工程抢险组", "险情处置", [{"resource_type": "排水泵", "quantity": 3}]),
            _step(9, "09:50", "交通管制组", "交通管制"),
        ],
    },
    "S3": {
        "id": "S3",
        "name": "地质灾害",
        "emergency_type": "geological",
        "description": (
            "山区突发滑坡地质灾害，需开展预警发布、启动响应、危险区排查、人员转移、"
            "救援搜救、医疗救护、险情监测与交通管制等救援行动。"
            "（以预案为准，生成的步骤可包括但不限于上述措施）"
        ),
        "naive_draft": [
            _step(1, "10:00", "应急指挥部", "启动响应"),                       # 时序错误：灾害预警应先于启动响应
            _step(2, "10:10", "社区/基层", "危险区排查"),
            _step(3, "10:20", "社区/基层", "人员转移"),
            _step(4, "10:30", "工程抢险组", "救援搜救", [{"resource_type": "挖掘机", "quantity": 8}]),  # 资源超配: 8>6
            _step(5, "10:40", "医疗救护组", "医疗救护"),
            _step(6, "10:50", "气象/监测组", "隐患监测"),
            _step(7, "11:00", "消防救援组", "环境监测"),                       # 职责错误：环境监测属环境监测组
            # 缺：灾害预警（启动响应的前置）
        ],
        "correct_base": [
            _step(1, "09:30", "气象/监测组", "灾害预警"),
            _step(2, "10:00", "应急指挥部", "启动响应"),
            _step(3, "10:10", "社区/基层", "危险区排查"),
            _step(4, "10:20", "社区/基层", "人员转移"),
            _step(5, "10:30", "工程抢险组", "救援搜救", [{"resource_type": "挖掘机", "quantity": 4}]),
            _step(6, "10:40", "医疗救护组", "医疗救护", [{"resource_type": "救护车", "quantity": 2}]),
            _step(7, "10:50", "气象/监测组", "隐患监测"),
            _step(8, "11:00", "环境监测组", "环境监测"),
            _step(9, "11:10", "交通管制组", "交通管制"),
        ],
    },
}


def get_scenario(scenario_id: str) -> dict:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"未知场景: {scenario_id}，可选: {list(SCENARIOS)}")
    return SCENARIOS[scenario_id]


def scenario_query(scenario_id: str) -> str:
    """用于 RAG 检索的场景查询（也可作为 LLM 生成提示的场景描述）。"""
    return get_scenario(scenario_id)["description"]


def scenario_rag_queries(scenario_id: str) -> List[str]:
    """多路 RAG 检索查询：从「场景说明 / 处置流程 / 职责分工 / 资源保障 / 阶段措施」
    五个角度召回预案内容，避免只召回泛化描述、导致生成脚本模板化。

    返回列表（elcig 会逐条检索并合并去重），兼容 ELCIGLoop 同时接受 str 与 list。
    """
    sc = get_scenario(scenario_id)
    name = sc["name"]
    desc = sc["description"]
    return [
        desc,
        # f"{name} 应急预案 应急响应 处置流程 时间顺序 步骤安排",
        # f"{name} 应急 责任单位 职责分工 组织指挥 谁负责",
        # f"{name} 应急 资源保障 物资 装备 队伍 调度",
        # f"{name} 应急 预警 响应 处置 恢复 各阶段 措施",
    ]
