"""脚本生成器：知识增强生成初稿（LLMGenerator，调用 dsutil 大模型）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Set

from llm.client import LLMClient, extract_json
from models import ResourceDemand, Script, ScriptStep
from parser import steps_from_json
from prompts import build_generate
from audit import emit


class Generator(ABC):
    @abstractmethod
    def generate(self, scenario: Dict, knowledge: str, use_rag: bool) -> Script:
        raise NotImplementedError


class LLMGenerator(Generator):
    def __init__(self, client: LLMClient, max_retries: int = 3,
                 action_grounding: Optional[Dict] = None):
        self.client = client
        self.max_retries = max_retries
        ag = action_grounding or {}
        # use_llm: 是否额外走一次 LLM 做“真正的语义映射 + 4 字生成”（默认关，确定性接地已够用）
        self.use_llm_grounding = bool(ag.get("use_llm", False))
        self.grounding_threshold = float(ag.get("similarity_threshold", 0.34))
        self.provided_actions_cfg = ag.get("provided_actions", {}) or {}

    def _provided_actions(self, scenario_id: str) -> Optional[Set[str]]:
        """若配置中显式“提供”了该场景的标准动作清单，则优先使用，否则交回自动推导。"""
        pa = self.provided_actions_cfg
        if isinstance(pa, dict) and scenario_id in pa and pa[scenario_id]:
            return set(pa[scenario_id])
        return None

    def _print_grounding(self, sc: Script):
        report = getattr(sc, "grounding_report", []) or []
        if not report:
            return
        mapped = [r for r in report if r.get("status") != "auto"]
        auto = [r for r in report if r.get("status") == "auto"]
        print(f"\n>> [动作语义接地] 共 {len(report)} 步："
              f"映射到标准动作 {len(mapped)} 步，语义不同自动生成4字动作 {len(auto)} 步。")
        for r in auto:
            print(f"   步骤{r['step_id']}：原动作'{r.get('raw')}' "
                  f"-> 自动生成4字动作'{r.get('grounded')}'（未命中标准动作本体）")

    def generate(self, scenario: Dict, knowledge: str, use_rag: bool) -> Script:
        system, user = build_generate(scenario, knowledge, use_rag)
        last_reason = "未知"
        provided = self._provided_actions(scenario["id"])
        for attempt in range(1, self.max_retries + 1):
            if use_rag:
                print(f"\n【RAG 输入提示词｜场景 {scenario['id']}｜生成 {attempt}/{self.max_retries}】")
                print((system + "\n\n" + user) if system else user)
                print("【RAG 输入提示词结束】", flush=True)
            text = self.client.chat(system, user, json_mode=True)
            if use_rag:
                print(f"\n【RAG 模型输出｜场景 {scenario['id']}｜生成 {attempt}/{self.max_retries}｜字符数 {len(text) if text is not None else 0}】")
                print(text if text is not None else "（模型调用未返回内容）")
                print("【RAG 模型输出结束】")
            if text is None:
                last_reason = "模型调用返回 None（网络/接口异常）"
                print(f"[generator] 第 {attempt}/{self.max_retries} 次生成：{last_reason}，稍后重试。")
                continue
            obj = extract_json(text)
            if isinstance(obj, dict) and "steps" in obj:
                try:
                    sc = steps_from_json(obj, scenario["id"],
                                         provided_actions=provided,
                                         threshold=self.grounding_threshold)
                except (ValueError, TypeError, AttributeError) as exc:
                    last_reason = str(exc)
                    emit("generation_parse_failure", attempt=attempt, reason=last_reason)
                    continue
                # 可选：LLM 语义接地（进一步将自由动作映射到标准动作 / 生成4字动作）
                if self.use_llm_grounding:
                    from action_grounding import LLMActionGrounder
                    grounder = LLMActionGrounder(self.client, threshold=self.grounding_threshold)
                    grounder.ground(sc, scenario["id"], provided_actions=provided)
                self._print_grounding(sc)
                return sc
            # 解析失败：打印诊断信息后重试（大概率由偶发格式问题或截断导致）
            last_reason = "JSON 解析失败"
            snippet = text[:600].replace("\n", " ")
            print(f"[generator] 第 {attempt}/{self.max_retries} 次生成：{last_reason}。")
            print(f"           原始输出前 600 字（用于诊断）：{snippet}")
        raise RuntimeError(
            f"LLM 生成未返回可解析的 steps JSON（已重试 {self.max_retries} 次，最后一次原因：{last_reason}）。"
            "请检查：① dsutil 的 max_tokens 是否足够（长脚本需增大）；② 模型是否稳定输出 JSON。"
        )
