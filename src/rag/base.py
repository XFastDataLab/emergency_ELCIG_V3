"""RAG 检索器抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class RAGRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> str:
        """返回与 query 相关的知识片段拼接文本（作为生成上下文）。"""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    def retrieve_chunks(self, query: str, top_k: int = 5) -> list:
        """返回结构化知识片段列表 [{"title":..., "content":...}]，便于多路检索合并去重。

        默认实现直接基于 retrieve() 返回单段；具体检索器可重写为逐条返回。
        """
        text = self.retrieve(query, top_k=top_k)
        if not text:
            return []
        return [{"title": "参考", "content": text}]
