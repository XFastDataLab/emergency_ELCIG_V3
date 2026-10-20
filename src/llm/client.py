"""大模型客户端：封装用户已验证的 dsutil（火山方舟 DeepSeek）。

对外保持与原代码完全一致的同步接口：
    LLMClient(base_url, api_key, model, temperature, timeout)
    client.chat(system, user, json_mode=False) -> str | None

实际调用由 dsutil 完成（URL / API-KEY / 模型名均写在 dsutil 中，已测试可用），
本模块只负责：选模型、合并 system+user 为单 prompt、统一返回文本。
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from llm.dsutil import call_deepseek_v3, call_deepseek_v4


def extract_json(text: str):
    """从模型输出中稳健地提取 JSON 对象。

    兼容以下常见问题：
      - 被 ```json ... ``` 代码块包裹；
      - 前后夹带解释性文字；
      - 含 BOM / 控制字符；
      - 数组/对象末尾存在尾随逗号（如 [..,] / {..,}）。
    """
    if text is None:
        return None
    raw = text.strip()
    # 1) 剥离 ```json ... ``` 代码块（优先，最稳健）
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    # 2) 清理 BOM 与控制字符（保留换行/制表与中文标点）
    raw = raw.replace("\ufeff", "")
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    # 3) 截取最外层 { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = raw[start:end + 1]
    # 4) 尝试直接解析
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 5) 修复尾随逗号后重试
    fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


class LLMClient:
    def __init__(self, base_url: str = "", api_key: str = "",
                 model: str = "deepseek-v4-flash-ga-260731",
                 temperature: float = 0.3, timeout: int = 90):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        # dsutil 已内置 URL/KEY/MODEL，这里仅按 model 名决定走 V3 还是 V4 入口。
        # 注：dsutil 的 DeepSeek_V4_FLASH_OFFICAL 枚举在 callAI 中会硬编码回退到
        # DeepSeek_V4，故 V4 统一走 call_deepseek_v4（即 test() 验证过的路径）。
        self._is_v3 = "v3" in (model or "").lower()

    def chat(self, system: str, user: str, json_mode: bool = False) -> Optional[str]:
        """单次对话，返回模型文本；失败返回 None。

        dsutil 的接口只接收单条 prompt，故把 system + user 合并为一条提问。
        """
        prompt = (system + "\n\n" + user) if system else user
        try:
            if self._is_v3:
                return call_deepseek_v3(prompt, printPrompt=False, printResponse=False)
            return call_deepseek_v4(prompt, printPrompt=False, printResponse=False)
        except Exception as e:  # noqa: BLE001
            print(f"[LLMClient] 模型调用失败：{e}")
            return None
