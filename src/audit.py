"""Append-only experiment audit, isolated by context; never serialize credentials."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import time

OUTPUT = Path(__file__).resolve().parents[1] / "output"
_active = ContextVar("experiment_audit", default=None)


def new_directory(scenario, method):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%m%d-%H%M")
    base = f"{scenario}-{method}-{stamp}"
    index = 1
    while True:
        path = OUTPUT / (base if index == 1 else f"{base}-{index:02d}")
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            index += 1
        else:
            return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint(script):
    data = script.to_dict() if hasattr(script, "to_dict") else script
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def emit(event, **data):
    path = _active.get()
    if path is not None:
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with (path / "trace.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def audit_scope(path, **metadata):
    token = _active.set(Path(path))
    emit("run_start", **metadata)
    try:
        yield
    except Exception as exc:
        emit("run_failure", exception_type=type(exc).__name__)
        raise
    finally:
        emit("run_end")
        _active.reset(token)


def print_ai_request(label, system, user):
    print(f"\n【{label}｜AI 输入提示词】")
    print((system + "\n\n" + user) if system else user)
    print("【AI 输入提示词结束】", flush=True)


def print_ai_response(label, text):
    print(f"\n【{label}｜AI 输出｜字符数 {len(text) if text is not None else 0}】")
    print(text if text is not None else "（模型调用未返回内容）")
    print("【AI 输出结束】")


class AuditedClient:
    """Proxy preserves the configured client's call semantics and captures all stages."""
    def __init__(self, client):
        self.client = client

    def chat(self, system, user, json_mode=False):
        emit("llm_request", system=system, user=user, json_mode=json_mode)
        start = time.perf_counter()
        response = self.client.chat(system, user, json_mode=json_mode)
        emit("llm_response", text=response, elapsed_seconds=time.perf_counter()-start)
        return response


def call_stats(path):
    events = [json.loads(line) for line in (Path(path)/"trace.jsonl").read_text(encoding="utf-8").splitlines()]
    responses = [x for x in events if x["event"] == "llm_response"]
    return {"llm_calls": sum(x["event"] == "llm_request" for x in events),
            "llm_seconds": sum(x["elapsed_seconds"] for x in responses),
            "response_characters": sum(len(x["text"] or "") for x in responses)}
