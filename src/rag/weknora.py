"""WeKnora 知识库检索器（RAG 外部知识库）。

接入文档：https://github.com/Tencent/WeKnora/blob/main/docs/api/knowledge-base.md
单库混合搜索（向量 + 关键词，旧客户端兼容方式）：
    GET  {base_url}/knowledge-bases/:id/hybrid-search
说明：文档规定 GET 兼容模式同样需要携带 JSON 请求体（Content-Type: application/json）。
认证头：  X-API-Key: <your_api_key>
查询参数：resource_urls=handle(默认) | public
请求体（GET 也需 JSON）：
    {
      "query_text": "...",          # 必填，查询文本
      "vector_threshold": 0.5,      # 可选，向量相似度阈值(0-1)
      "keyword_threshold": 0.5,     # 可选，关键词匹配阈值(0-1)
      "match_count": 10,            # 可选，返回结果数量上限
      "disable_keywords_match": false,
      "disable_vector_match": false,
      "knowledge_ids": [...],       # 可选，仅在指定知识 ID 范围内召回
      "tag_ids": [...],             # 可选，标签过滤
      "only_recommended": false
      # 跨库检索时也可用 knowledge_base_ids（与路径 :id 二选一，body 优先级更高）
    }
响应：    {"data":[{"id","content","knowledge_title","knowledge_filename","score",...}], "success": true}
"""
from __future__ import annotations

import requests

from rag.base import RAGRetriever


class WeKnoraRetriever(RAGRetriever):
    def __init__(self, base_url: str, api_key: str,
                 knowledge_base_id: str = "", knowledge_base_ids: list = None,
                 resource_urls: str = "handle", timeout: int = 30):
        # base_url 形如 http://host:port/api/v1
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # self.knowledge_base_id = knowledge_base_id
        # self.knowledge_base_ids = knowledge_base_ids or []
        self.resource_urls = resource_urls
        self.timeout = timeout

    def is_available(self) -> bool:
        return True

    def _headers(self) -> dict:
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _build_body(self, query: str, top_k: int) -> dict:
        """构造单库混合搜索请求体。"""
        body: dict = {
            "query_text": query,
            "match_count": int(top_k),
        }
        # 用户若指定多个知识库，则放入 body（文档：cross-KB 时优先级高于路径 :id）
        # if self.knowledge_base_ids:
        #     body["knowledge_base_ids"] = self.knowledge_base_ids
        return body

    def _hybrid_search(self, query: str, top_k: int = 5) -> list:
        """单库混合搜索，返回结构化片段列表。

        每个元素：{"title", "content", "score", "knowledge_id", "knowledge_filename"}
        GET 为旧客户端兼容方式（文档指定），并附带 POST 兜底以防 GET 请求体被代理丢弃。
        """

        kb_id = "2802451b-a15e-412c-9fb2-c0b103efaea8"
        url = f"{self.base_url}/knowledge-bases/{kb_id}/hybrid-search"
        body = self._build_body(query, top_k)
        params = {"resource_urls": self.resource_urls}
        headers = self._headers()

        # 优先 GET（用户指定的旧接口兼容方式），失败则退 POST（推荐方式）
        last_err = None
        for method, verb in (("GET", requests.get), ("POST", requests.post)):
            try:
                resp = verb(url, headers=headers, json=body,
                            params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("success", False):
                    if method == "POST":
                        print("[WeKnora] GET 混合检索不可用，已回退 POST。")
                    return self._parse_chunks(payload)
                # success=False：打印错误后尝试下一方式
                last_err = f"success={payload.get('success')} error={payload.get('error')}"
                print(f"[WeKnora] 混合检索返回未成功（{method}）：{last_err}")
            except Exception as e:  # 检索失败不影响主流程，仅降级为无知识库
                last_err = e
                print(f"[WeKnora] {method} 混合检索失败：{e}")
        return []

    @staticmethod
    def _parse_chunks(payload: dict) -> list:
        out = []
        for c in (payload.get("data", []) or []):
            title = c.get("knowledge_title") or c.get("knowledge_filename") or "未知来源"
            content = (c.get("content") or "").strip()
            if content:
                out.append({
                    "title": title,
                    "content": content,
                    "score": c.get("score"),
                    "knowledge_id": c.get("knowledge_id"),
                    "knowledge_filename": c.get("knowledge_filename"),
                })
        return out

    def retrieve_chunks(self, query: str, top_k: int = 5) -> list:
        """返回结构化片段列表 [{"title", "content", ...}]，供多路检索合并去重。"""
        return self._hybrid_search(query, top_k=top_k)[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """兼容旧接口：直接返回拼接文本。"""

        print("======== RAG ========查询：" + query + "\n")
        chunks = self._hybrid_search(query, top_k=top_k)
        parts = []
        for i, c in enumerate(chunks):
            parts.append(f"【参考{i + 1}｜{c['title']}】\n{c['content']}")
        return "\n\n".join(parts)


if __name__ == '__main__':
    r = WeKnoraRetriever(
        base_url="http://113.46.176.252:8580/api/v1",
        api_key="REPLACE_WITH_ENV_VALUE",
        knowledge_base_id="2802451b-a15e-412c-9fb2-c0b103efaea8",
        knowledge_base_ids=[]
    )
    r.retrieve("台风应急措施")
