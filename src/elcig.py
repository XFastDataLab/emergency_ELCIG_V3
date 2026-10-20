"""ELCIG 迭代闭环：生成 → 验证 → 错误定位 → 定向修正 → 再验证。

消融开关：
  - use_rag        : 是否在生成阶段接入 WeKnora 外部知识库（RAG）
  - use_validator  : 是否启用符号规则验证器（无则退化为 LLM-only / RAG-LLM / Self-Refine）
  - correction_mode: rule_feedback（ELCIG 本文）| self_refine（模型自反馈，不复用外部规则）
"""
from __future__ import annotations

from typing import Dict, List, Optional
import copy
import json
import random
from audit import emit, print_ai_request, print_ai_response

from llm.client import LLMClient, extract_json
from models import ErrorRecord, Script
from parser import steps_from_json
from prompts import build_self_review
from scenarios import get_scenario
from validator.composite import RuleValidator


def _count_types(errors: List[ErrorRecord]) -> Dict[str, int]:
    out = {"temporal": 0, "responsibility": 0, "resource": 0}
    for e in errors:
        out[e.error_type] = out.get(e.error_type, 0) + 1
    return out


# 仅作提示、不计入“硬错误/CSR”的类型（未知职责组合、可省略的中间前置）
_HINT_TYPES = ("responsibility_hint", "temporal_hint")


def _is_hard(e: ErrorRecord) -> bool:
    return e.error_type not in _HINT_TYPES


class ELCIGLoop:
    def __init__(self, cfg, generator, retriever, validator: RuleValidator,
                 corrector, client: Optional[LLMClient], scenario_query_fn):
        self.cfg = cfg
        self.generator = generator
        self.retriever = retriever
        self.validator = validator
        self.corrector = corrector
        self.client = client
        self.scenario_query = scenario_query_fn
        self.use_rag: bool = bool(cfg.rag.use_rag)
        self.use_validator: bool = bool(cfg.validator.use_validator)
        self.correction_mode: str = cfg.correction.mode
        self.max_iter: int = int(cfg.experiment.max_iterations)
        self.last_knowledge: str = ""   # 最近一次检索到的预案上下文（供展示/审计）

    # ------------------------------------------------------------------
    def _retrieve_knowledge(self, scenario_id: str):
        """多路 RAG 检索：逐条查询并合并去重，返回 (拼接文本, 片段数)。"""
        q = self.scenario_query(scenario_id)
        queries = list(q) if isinstance(q, (list, tuple)) else [q]
        top_k = int(self.cfg.rag.top_k)
        max_chunks = int(self.cfg.rag.get("max_context_chunks", 12))
        rchk = getattr(self.retriever, "retrieve_chunks", None)

        seen = set()
        chunks = []
        for query in queries:
            print(f"\n【RAG 检索输入｜场景 {scenario_id}｜top_k={top_k}】\n{query}", flush=True)
            if rchk is not None:
                items = rchk(query, top_k=top_k)
            else:
                txt = self.retriever.retrieve(query, top_k=top_k)
                items = [{"title": "参考", "content": txt}] if txt else []
            emit("retrieval_query", query=query, chunks=items)
            print(f"【RAG 检索输出｜场景 {scenario_id}】")
            print(json.dumps(items, ensure_ascii=False, indent=2))
            for it in items:
                content = (it.get("content") or "").strip()
                key = (it.get("title", ""), content)
                if not content or key in seen:
                    continue
                seen.add(key)
                chunks.append(it)
                if len(chunks) >= max_chunks:
                    break
            if len(chunks) >= max_chunks:
                break
        knowledge = "\n\n".join(
            f"【参考｜{c.get('title', '')}】\n{c.get('content', '')}" for c in chunks
        )
        emit("retrieval_context", chunks=chunks, knowledge=knowledge)
        return knowledge, len(chunks)

    # ------------------------------------------------------------------
    def _method_label(self) -> str:
        """根据当前开关推断方法名（用于错误打印的标签）。"""
        if self.use_validator:
            return "ELCIG"
        if self.correction_mode == "self_refine":
            return "Self-Refine"
        return "RAG-LLM" if self.use_rag else "LLM"

    # ------------------------------------------------------------------
    def _print_error_breakdown(self, label: str, errors: List[ErrorRecord]):
        """收集并打印错误信息：按 时序/职责/资源 三类统计硬错误，并列出明细。

        宽容模式下产生的 responsibility_hint / temporal_hint 仅作提示，
        单独计数、不计入“硬错误”，避免把合理步骤误判。
        """
        hard = [e for e in errors if _is_hard(e)]
        hints = [e for e in errors if not _is_hard(e)]
        bd = _count_types(hard)
        print(f"\n>> [{label}] 错误校验（时序/职责/资源）："
              f"时序={bd['temporal']}  职责={bd['responsibility']}  资源={bd['resource']}"
              f"（提示类={len(hints)}，未计入硬错）")
        if hard:
            print("   检出错误明细：")
            for i, e in enumerate(hard, 1):
                cn = ErrorRecord.TYPE_CN.get(e.error_type, e.error_type)
                print(f"   [{i}] 类型={cn}  步骤={e.step_id}  严重度={e.severity}")
                print(f"       位置：{e.location}")
                print(f"       描述：{e.description}")
                print(f"       违反规则：{e.violated_rule}")
        else:
            print("   （无硬错误）")
        if hints:
            print("   提示详情（不计入硬错误数，不用于修正停止判断）：")
            for i, e in enumerate(hints, 1):
                print(f"   [提示{i}] 类型={e.error_type}  步骤={e.step_id}")
                print(f"       位置：{e.location}")
                print(f"       描述：{e.description}")
                print(f"       相关规则：{e.violated_rule}")

    # ------------------------------------------------------------------
    def prepare_draft(self, scenario_id: str):
        knowledge = ""
        if self.use_rag:
            if self.retriever is None or not self.retriever.is_available():
                raise RuntimeError("RAG 检索不可用，不能将无检索初稿标记为 RAG 初稿。")
            knowledge, _ = self._retrieve_knowledge(scenario_id)
            if not knowledge.strip():
                raise RuntimeError("RAG 未返回有效片段，本次实验停止；详见检索日志。")
        script = self.generator.generate(get_scenario(scenario_id), knowledge, self.use_rag)
        emit("initial_draft", script=script.to_dict(), grounding_report=script.grounding_report)
        return script, knowledge

    def run(self, scenario_id: str, initial_script=None, knowledge=None):
        scenario = get_scenario(scenario_id)
        method_label = self._method_label()

        # 1) 知识增强：多路检索外部知识库（RAG），以预案内容驱动生成、避免模板化
        if initial_script is None:
            script, knowledge = self.prepare_draft(scenario_id)
        else:
            script = copy.deepcopy(initial_script)
            emit("shared_draft_reused", script=script.to_dict())
        self.last_knowledge = knowledge or ""

        # 3) 初始错误收集与打印（所有方法都做：LLM / RAG / Self-Refine / ELCIG）
        initial_errors = self.validator.validate(script) if self.validator else []
        history = [self._snapshot(0, script, initial_errors)]

        # 4) 分支：规则验证引导修正（ELCIG）/ 模型自反馈（Self-Refine）/ 一次性生成
        if self.use_validator:
            # ELCIG：迭代“验证→定向修正→再验证”
            errors = initial_errors
            deferred_errors: List[ErrorRecord] = []
            for k in range(1, self.max_iter + 1):
                hard = [e for e in errors if _is_hard(e)]
                carried_errors = deferred_errors
                deferred_errors = []
                feedback_errors = carried_errors + hard
                if k == 1 and len(hard) > 50:
                  
                    a = random.randint(1, len(hard) - 3)
                    selected = set(random.sample(range(len(hard)), a))
                    feedback_errors = [e for i, e in enumerate(hard) if i in selected]
                    deferred_errors = [e for i, e in enumerate(hard) if i not in selected]
               
                if not feedback_errors:
                    break
                print(f"\n【ELCIG 迭代 {k}/{self.max_iter}｜场景 {scenario_id}｜待修正硬错误 {len(feedback_errors)} 条"
                      f"｜本地检出 {len(hard)} 条｜合并保留 {len(carried_errors)} 条｜留待下轮 {len(deferred_errors)} 条】")
                emit("correction_start", iteration=k, mode="rule_feedback",
                     detected_error_count=len(hard),
                     carried_error_count=len(carried_errors),
                     feedback_error_count=len(feedback_errors),
                     deferred_error_count=len(deferred_errors),
                     feedback_errors=[e.to_dict() for e in feedback_errors],
                     deferred_errors=[e.to_dict() for e in deferred_errors])
                script = self.corrector.correct(script, feedback_errors, scenario)
                errors = self.validator.validate(script)
                history.append(self._snapshot(k, script, errors, deferred_error_count=len(deferred_errors)))
            final_errors = errors
            final_label = "ELCIG·最终错误"

        elif self.correction_mode == "self_refine" and self.client is not None:
            # Self-Refine 最多执行一次自反馈修正；max_iterations=0 时仍禁用修正。
            # 不复用外部规则，但仍用验证器收集/打印错误。
            errors = initial_errors
            self_refine_limit = min(1, self.max_iter)
            for k in range(1, self_refine_limit + 1):
                emit("correction_start", iteration=k, mode="self_refine")
                system, user = build_self_review(script.render())
                label = f"Self-Refine 迭代 {k}/{self_refine_limit}｜场景 {scenario_id}"
                print_ai_request(label, system, user)
                text = self.client.chat(system, user, json_mode=True)
                print_ai_response(label, text)
                obj = extract_json(text)
                if not isinstance(obj, dict) or not obj.get("needs_fix", False):
                    emit("self_refine_stop", reason="no_fix_or_invalid_response", parsed=obj)
                    break
                if "steps" in obj:
                    try:
                        new_script = steps_from_json(obj, scenario_id)
                    except (ValueError, TypeError, AttributeError) as exc:
                        emit("self_refine_stop", reason=str(exc))
                        break
                    if new_script.render() == script.render():
                        emit("self_refine_stop", reason="unchanged_script")
                        break
                    script = new_script
                else:
                    emit("self_refine_stop", reason="missing_steps")
                    break
                errors = self.validator.validate(script)
                history.append(self._snapshot(k, script, errors))
            final_errors = errors
            final_label = "Self-Refine·最终错误"

        else:
            # LLM-only / RAG-LLM：仅生成一次，不做修正；最终错误 = 初始错误
            final_errors = initial_errors
            final_label = f"{method_label}·最终错误（未修正）"

        # 5) 最终错误统计与打印（所有方法）
        self._print_error_breakdown(final_label, final_errors)
        emit("final_draft", script=script.to_dict(), errors=[e.to_dict() for e in final_errors])
        return script, history

    # ------------------------------------------------------------------
    def _snapshot(self, k: int, script: Script, errors: Optional[List[ErrorRecord]] = None,
                  deferred_error_count: int = 0):
        if errors is None and self.validator is not None:
            errors = self.validator.validate(script)
        errs = errors if errors is not None else []
        # 宽松模式下未知职责组合产生 responsibility_hint、缺失的中间前置产生 temporal_hint
        # （severity=1），仅作提示、不计入"需修正错误数"与 CSR，避免把合理步骤误判并拉低约束满足率。
        hard_errs = [e for e in errs if _is_hard(e)]
        hints = [e for e in errs if not _is_hard(e)]
        self._print_error_breakdown(f"{self._method_label()}·{'初始验证' if k == 0 else f'第{k}轮验证'}", errs)
        total = self.validator.constraint_count(script) if self.validator else 0
        satisfied = total - len(hard_errs)
        csr = (total - len(hard_errs)) / total if total else 1.0
        # CSR 分子/分母可视化：分子=满足约束数(total-硬错误)，分母=受检约束总数(total)
        print(f">> [迭代 {k}] CSR 约束满足率：满足约束 {satisfied} / 受检约束 {total} "
              f"（硬错误 {len(hard_errs)}） →  CSR = {round(csr, 4)}")
        snapshot = {
            "iteration": k,
            "error_count": len(hard_errs),
            "deferred_error_count": deferred_error_count,
            "pending_error_count": len(hard_errs) + deferred_error_count,
            "error_types": _count_types(hard_errs),
            "hint_count": len(hints),
            "total_constraints": total,
            "satisfied_constraints": satisfied,
            "csr": round(csr, 4),
            "errors": [e.to_dict() for e in errs],  # 完整（含 hint）供审计展示
            "script": script.to_dict(),
        }
        emit("validation", **snapshot)
        return snapshot
