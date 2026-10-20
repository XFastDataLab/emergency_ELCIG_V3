"""Paired initial drafts, repeated four-method experiments and auditable outputs."""
from __future__ import annotations
import copy
import json
import shutil
import time
from pathlib import Path
from config import Config
from corrector import LLMCorrector, RuleGuidedCorrector
from evaluation import constraint_satisfaction_rate, error_reduction_rate
from generator import LLMGenerator
from elcig import ELCIGLoop
from llm.client import LLMClient
from rag.weknora import WeKnoraRetriever
from rules.registry import get_rules
from scenarios import get_scenario, scenario_rag_queries
from validator.composite import RuleValidator
from audit import AuditedClient, audit_scope, call_stats, emit, fingerprint, new_directory, write_json
from validator_report import save_comparison_report

METHOD_CONFIGS = {
    "LLM": {"use_rag": False, "use_validator": False, "correction_mode": "none"},
    "RAG": {"use_rag": True, "use_validator": False, "correction_mode": "none"},
    "SelfRefine": {"use_rag": False, "use_validator": False, "correction_mode": "self_refine"},
    "ELCIG": {"use_rag": True, "use_validator": True, "correction_mode": "rule_feedback"},
}
ALIASES = {"LLM-only": "LLM", "RAG-LLM": "RAG", "Self-Refine": "SelfRefine"}
LABELS = {v: k for k, v in ALIASES.items()}
LABELS["ELCIG"] = "ELCIG"


def build_components(cfg):
    client = AuditedClient(LLMClient(cfg.llm.base_url, cfg.llm.api_key, cfg.llm.model,
                                   cfg.llm.temperature, cfg.llm.timeout))
    retriever = WeKnoraRetriever(cfg.rag.base_url, cfg.rag.api_key,
        cfg.rag.knowledge_base_id, cfg.rag.knowledge_base_ids, cfg.rag.resource_urls)
    return {"client": client, "retriever": retriever,
            "generator": LLMGenerator(client, action_grounding=cfg.get("action_grounding", {}))}


def _method_config(cfg, method):
    mg = METHOD_CONFIGS[method]
    data = copy.deepcopy(dict(cfg))
    data["rag"]["use_rag"] = mg["use_rag"]
    data["validator"]["use_validator"] = mg["use_validator"]
    data["correction"]["mode"] = mg["correction_mode"]
    return Config(data)


def _loop(cfg, method, sid, comps):
    mcfg = _method_config(cfg, method)
    rules = get_rules(sid)
    validator = RuleValidator(rules, strict=mcfg.validator.get("responsibility_strict", False),
                             temporal_strict=mcfg.validator.get("temporal_strict", False))
    corrector = LLMCorrector(comps["client"], rules) if comps["client"] else RuleGuidedCorrector(rules)
    return mcfg, ELCIGLoop(mcfg, comps["generator"], comps["retriever"], validator,
                        corrector, comps["client"], scenario_rag_queries)


def prepare_shared_draft(cfg, method, sid, comps, repeat, output_dir=None):
    owner = Path(output_dir) if output_dir is not None else new_directory(sid, LABELS[method])
    path = owner / ("shared_RAG_draft" if method == "RAG" else "shared_noRAG_draft")
    path.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with audit_scope(path, scenario=sid, repeat=repeat, stage="shared_generation"):
        _, loop = _loop(cfg, method, sid, comps)
        script, knowledge = loop.prepare_draft(sid)
        draft = {"script": script, "knowledge": knowledge, "draft_id": f"{owner.name}-{path.name}",
                 "draft_hash": fingerprint(script), "source_dir": str(path),
                 "repeat": repeat, "use_rag": method == "RAG"}
        write_json(path / "initial.json", script.to_dict())
        write_json(path / "metadata.json", {k: v for k, v in draft.items() if k != "script"})
    draft["generation_seconds"] = time.perf_counter() - start
    draft["generation_cost"] = call_stats(path)
    write_json(path / "metadata.json", {k: v for k, v in draft.items() if k != "script"})
    events = [json.loads(line) for line in (path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    write_json(path / "llm_messages.json", [e for e in events if e["event"] in ("llm_request", "llm_response")])
    write_json(path / "retrieval.json", [e for e in events if e["event"] in ("retrieval_query", "retrieval_context")])
    return draft


def run_method(cfg, method, scenario_id, comps, shared_draft=None, repeat=1, output_dir=None):
    method = ALIASES.get(method, method)
    mcfg, loop = _loop(cfg, method, scenario_id, comps)
    if shared_draft is None:
        output_dir = Path(output_dir) if output_dir is not None else new_directory(scenario_id, LABELS[method])
        shared_draft = prepare_shared_draft(cfg, "RAG" if mcfg.rag.use_rag else "LLM",
                                            scenario_id, comps, repeat, output_dir=output_dir)
    if (shared_draft["script"].scenario_id != scenario_id or
        shared_draft["repeat"] != repeat or shared_draft["use_rag"] != bool(mcfg.rag.use_rag)):
        raise ValueError("共享初稿的场景、重复编号或检索设置不匹配")
    path = Path(output_dir) if output_dir is not None else new_directory(scenario_id, LABELS[method])
    start = time.perf_counter()
    with audit_scope(path, scenario=scenario_id, method=LABELS[method], repeat=repeat):
        # Each result keeps a complete local copy of its shared generation data.
        shared_folder = "shared_RAG_draft" if shared_draft["use_rag"] else "shared_noRAG_draft"
        local_shared = path / shared_folder
        if Path(shared_draft["source_dir"]).resolve() != local_shared.resolve():
            shutil.copytree(shared_draft["source_dir"], local_shared)
        write_json(path / "initial.json", shared_draft["script"].to_dict())
        write_json(path / "shared_draft_reference.json", {
            "draft_id": shared_draft["draft_id"], "draft_hash": shared_draft["draft_hash"],
            "source_dir": shared_draft["source_dir"], "local_copy": shared_folder,
            "use_rag": shared_draft["use_rag"], "repeat": repeat,
        })
        emit("draft_reference", draft_id=shared_draft["draft_id"],
             draft_hash=shared_draft["draft_hash"], source_dir=shared_draft["source_dir"])
        final, history = loop.run(scenario_id, shared_draft["script"], shared_draft["knowledge"])
        initial_errors, final_errors = history[0]["error_count"], history[-1]["error_count"]
        total = history[-1]["total_constraints"]
        result = {
            "method": LABELS[method], "method_key": method, "scenario": scenario_id,
            "scenario_name": get_scenario(scenario_id)["name"], "repeat": repeat,
            "use_rag": mcfg.rag.use_rag, "use_validator": mcfg.validator.use_validator,
            "initial_errors": initial_errors, "final_errors": final_errors,
            "err": error_reduction_rate(initial_errors, final_errors),
            "eredr": error_reduction_rate(initial_errors, final_errors),
            "eredr_defined": initial_errors > 0,
            "csr": constraint_satisfaction_rate(total, final_errors),
            "total_final": total, "satisfied": total-final_errors,
            "iterations": len(history)-1,
            "error_curve": [h.get("pending_error_count", h["error_count"]) for h in history],
            "detected_error_curve": [h["error_count"] for h in history],
            "history": history, "knowledge": loop.last_knowledge,
            "draft_id": shared_draft["draft_id"], "draft_hash": shared_draft["draft_hash"],
            "shared_source_dir": shared_draft["source_dir"], "output_dir": str(path),
            "shared_draft_copy": str(local_shared),
            "zero_error_success": final_errors == 0,
            "initial_steps": len(history[0]["script"]["steps"]), "final_steps": len(final.steps),
            "generation_seconds": shared_draft["generation_seconds"],
            "generation_cost": shared_draft["generation_cost"],
            "refinement_seconds": time.perf_counter()-start,
        }
        initial_actions = {s["action"] for s in history[0]["script"]["steps"]}
        final_actions = {s.action for s in final.steps}
        result["initial_action_retention"] = len(initial_actions & final_actions)/len(initial_actions) if initial_actions else None
        result["end_to_end_seconds"] = result["generation_seconds"] + result["refinement_seconds"]
        write_json(path/"initial.json", history[0]["script"])
        write_json(path/"final.json", final.to_dict())
        write_json(path/"history.json", history)
    result["refinement_cost"] = call_stats(path)
    write_json(path/"result.json", result)
    return result


def run_ablation(cfg, repeats=None):
    repeats = repeats if repeats is not None else int(cfg.experiment.get("repeats", 1))
    if repeats < 1:
        raise ValueError("repeats 必须大于 0")
    comps = build_components(cfg)
    results = []
    for sid in cfg.experiment.scenarios:
        for repeat in range(1, repeats+1):
            # Exactly two successful drafts per (scenario, repeat), without cross-repeat caching.
            owners = {method: new_directory(sid, LABELS[method]) for method in ("LLM", "RAG")}
            plain = prepare_shared_draft(cfg, "LLM", sid, comps, repeat, output_dir=owners["LLM"])
            rag = prepare_shared_draft(cfg, "RAG", sid, comps, repeat, output_dir=owners["RAG"])
            for method in METHOD_CONFIGS:
                shared = rag if METHOD_CONFIGS[method]["use_rag"] else plain
                results.append(run_method(cfg, method, sid, comps, shared, repeat, output_dir=owners.get(method)))
    return results


def run_validator_effectiveness(cfg, samples=36, include_boundary_case=False):
    from evaluation import build_boundary_cases, validator_dataset_effectiveness
    if samples + (len(cfg.experiment.scenarios) if include_boundary_case else 0) > 100:
        raise ValueError("实验1样本总数最多100条，包含每个所选场景的1条边界样本")
    if samples < len(cfg.experiment.scenarios)*8:
        raise ValueError("样本数至少为场景数 × 8，以覆盖无错、三类单错及混合组合")
    result = []
    for i, sid in enumerate(cfg.experiment.scenarios):
        count = samples // len(cfg.experiment.scenarios) + (i < samples % len(cfg.experiment.scenarios))
        extra_cases = build_boundary_cases(sid) if include_boundary_case else []
        path = new_directory(sid, "Validator")
        with audit_scope(path, scenario=sid, samples=count + len(extra_cases)):
            for case in extra_cases:
                write_json(path / "boundary_sample.json", case)
            row = validator_dataset_effectiveness(sid, count,
                strict=cfg.validator.get("responsibility_strict", False),
                temporal_strict=cfg.validator.get("temporal_strict", False), output_dir=path,
                extra_cases=extra_cases)
            row["output_dir"] = str(path)
            write_json(path/"result.json", row)
        result.append(row)
    return result


def save_results(cfg, ablation, effectiveness):
    path = new_directory("ALL", "summary") / "results.json"
    write_json(path, {"ablation": ablation, "validator_effectiveness": effectiveness})
    if effectiveness:
        save_comparison_report(path.parent, effectiveness)
    return str(path)
