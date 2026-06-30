"""ContextInjector — before_step hook 的核心实现。

从 Tree 状态中按 ring 分层抽取 cell 并格式化为可注入文本。
三段静态预算: pinned (L3/L4 无条件) + relevant (L0/L1/L2 按相似度) + warnings。

对应 spec: docs/specs/context_injector.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from tree_harness.core.cell_model import Cell
from tree_harness.core.embedding import embed_cell_text
from tree_harness.store.tree_store import TreeStore


@dataclass
class InjectorConfig:
    max_tokens: int = 2000
    max_cells: int = 10
    min_energy: float = 0.0
    min_similarity: float = 0.3
    ring_weights: dict = field(default_factory=lambda: {
        "L0": 0.5, "L1": 1.0, "L2": 1.5, "L3": 2.0, "L4": 2.5,
    })


@dataclass
class RetrievedContext:
    """retrieve() 的返回值。"""
    cells: List[str]            # 被选中的 cell id (按打分降序)
    formatted_text: str         # 格式化后的注入文本 (不含 marker)
    token_count: int            # 估算 token 数
    retrieval_scores: dict      # {cell_id: score}


@dataclass
class WarningEntry:
    """单条 warning (直接 quarantine 或邻居一跳传播)。"""
    cell_id: str
    text: str                   # 已格式化的自然语言 warning
    is_direct: bool             # True=直接 quarantine; False=邻居传播
    ray_weight: float           # 排序用 (is_direct=True 时为 1.0)
    recency: int                # 距今多少 step (小者优先)


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算: words * 1.3。"""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


class ContextInjector:
    """before_step 的实现细节 — 只读,不触发任何算符。"""

    def __init__(self, tree_store: TreeStore, config: InjectorConfig):
        self.tree_store = tree_store
        self.config = config

    # ------------------------------------------------------------------
    # pinned 段: L3/L4 无条件注入
    # ------------------------------------------------------------------
    def format_pinned(
        self, cells: List[Cell], budget: int,
        pin_open: str = "<|PINNED_DO_NOT_COMPACT|>",
        pin_close: str = "<|/PINNED|>",
    ) -> str:
        if not cells:
            return ""

        # 按 ring 降序 + energy 降序
        sorted_cells = sorted(
            cells, key=lambda c: (c.ring, c.energy), reverse=True
        )

        # 逐个加入,超 budget 时按 energy 降序保留
        selected: List[Cell] = []
        for cell in sorted_cells:
            test_text = self._format_cells(selected + [cell])
            full = f"{pin_open}\n{test_text}\n{pin_close}"
            if _estimate_tokens(full) <= budget:
                selected.append(cell)
            # budget 不够就跳过 (按 energy 降序,排在前面的优先)

        if not selected:
            return ""

        body = self._format_cells(selected)
        return f"{pin_open}\n{body}\n{pin_close}"

    # ------------------------------------------------------------------
    # relevant 段: L0/L1/L2 按相似度+energy+ring_weight 打分
    # ------------------------------------------------------------------
    def retrieve(
        self,
        task_description: str,
        repo: str,
        ring_filter: List[str],
        token_budget: int,
    ) -> RetrievedContext:
        embedder = self.tree_store.sqlite.embedder
        if embedder is None:
            return RetrievedContext(cells=[], formatted_text="", token_count=0,
                                     retrieval_scores={})

        query_emb = embedder.embed(task_description)
        candidates = self.tree_store.vec_search(
            query_emb, top_k=20, min_score=self.config.min_similarity
        )

        # 过滤: ring + energy + status (vec_search 已只返回 active)
        alive = [
            (cell, sim) for cell, sim in candidates
            if cell.ring in ring_filter and cell.energy > self.config.min_energy
        ]

        # 综合打分
        scored = [(cell, self._score_cell(cell, sim)) for cell, sim in alive]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Token budget 截断
        selected: List[Cell] = []
        for cell, score in scored:
            if len(selected) >= self.config.max_cells:
                break
            test_text = self._format_cells(selected + [cell])
            if _estimate_tokens(test_text) <= token_budget:
                selected.append(cell)
            # budget 不够就跳过

        text = self._format_cells(selected)
        return RetrievedContext(
            cells=[c.id for c in selected],
            formatted_text=text,
            token_count=_estimate_tokens(text),
            retrieval_scores={c.id: s for c, s in scored[:len(selected)]},
        )

    # ------------------------------------------------------------------
    # warnings 段
    # ------------------------------------------------------------------
    def format_warnings(
        self, warnings: List[WarningEntry], budget: int,
        warn_open: str = "<|WARNING_DO_NOT_COMPACT|>",
        warn_close: str = "<|/WARNING|>",
    ) -> str:
        if not warnings:
            return ""

        # 排序: (is_direct desc, ray_weight desc, recency asc)
        sorted_w = sorted(
            warnings,
            key=lambda w: (not w.is_direct, -w.ray_weight, w.recency),
        )

        lines: List[str] = []
        current_tokens = _estimate_tokens(f"{warn_open}\n{warn_close}")
        for w in sorted_w:
            line = w.text
            test = _estimate_tokens("\n".join(lines + [line]))
            if current_tokens + test <= budget:
                lines.append(line)
            # budget 不够就跳过

        if not lines:
            return ""

        body = "\n".join(lines)
        return f"{warn_open}\n{body}\n{warn_close}"

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _score_cell(self, cell: Cell, similarity: float) -> float:
        """综合打分: similarity * energy * ring_weight。"""
        ring_weight = self.config.ring_weights.get(cell.ring, 1.0)
        energy_factor = max(cell.energy, 0.1)  # 避免零乘
        return similarity * energy_factor * ring_weight

    def _format_cells(self, cells: List[Cell]) -> str:
        """格式化为 agent 可读文本 (不含 marker)。"""
        if not cells:
            return ""

        lines = ["[Project Experience]"]
        lines.append("Below are relevant lessons from previous work on this repository.")
        lines.append("")

        for cell in cells:
            lines.append(f"• [{cell.ring}] {cell.decision}")
            lines.append(f"  Why: {cell.rationale}")
            if cell.context_preconditions:
                cond = "; ".join(pc.assertion for pc in cell.context_preconditions)
                lines.append(f"  Conditions: {cond}")
            if cell.domain_tags:
                lines.append(f"  Tags: {', '.join(cell.domain_tags)}")
            lines.append("")

        lines.append("[End of Project Experience]")
        return "\n".join(lines)
