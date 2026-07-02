"""Embedding 抽象层。

提供向量化接口,供 SQLiteBackend 做 vec_search。
对应 spec: docs/specs/sqlite_backend.md (Embedding 模型章节)。

Embedding 输入: cell.decision + " | " + cell.rationale
"""
from __future__ import annotations

import hashlib
from typing import List, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """向量化协议。"""

    @property
    def dim(self) -> int:
        """向量维度。"""
        ...

    def embed(self, text: str) -> List[float]:
        """将文本转为 float 向量。"""
        ...


class DeterministicEmbedder:
    """确定性 hash 向量 (测试/原型用)。

    同一文本 → 同一向量,不依赖任何模型。
    向量元素由 SHA-256 派生,落在 [-1, 1]。
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        # chained hash: SHA-256 输出 32 字节,不够 dim 时链式派生更多字节
        vec: List[float] = []
        buf = hashlib.sha256(text.encode("utf-8")).digest()
        while len(vec) < self._dim:
            for b in buf:
                vec.append((b / 255.0) * 2.0 - 1.0)
                if len(vec) >= self._dim:
                    break
            buf = hashlib.sha256(buf).digest()
        return vec


class SentenceTransformerEmbedder:
    """真实 embedding — 支持中英文混合。

    模型级缓存: 同一 model_name 在进程内只加载一次,
    后续实例化直接复用已加载的模型 (~0ms vs 首次 ~5s)。
    """

    # 类级缓存: {model_name: SentenceTransformer}
    _model_cache: dict = {}

    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5"):
        if model_name not in self._model_cache:
            from sentence_transformers import SentenceTransformer
            self._model_cache[model_name] = SentenceTransformer(model_name)
        self._model = self._model_cache[model_name]
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    @classmethod
    def clear_cache(cls) -> None:
        """清空模型缓存 (测试用)。"""
        cls._model_cache.clear()


def embed_cell_text(decision: str, rationale: str) -> str:
    """构造 embedding 输入文本 (spec 规定: decision + ' | ' + rationale)。"""
    return f"{decision} | {rationale}"
