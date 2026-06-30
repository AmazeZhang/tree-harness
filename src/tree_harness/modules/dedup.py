"""Dedup 子例程 — Cambium Engine 三步管线 Step B。

判定候选 cell 是否与 tree 中已有 active cell 重复:
- score > 0.95 → REINFORCE (完全重复,强化已有 cell)
- score ∈ (0.85, 0.95] → LLM 仲裁 (灰区)
- score <= 0.85 → INSERT_NEW (足够新颖)

无状态子例程: 不创建 cell、不写 oplog、不动 ray 拓扑。
对应 spec: docs/specs/dedup.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, List, Tuple

from tree_harness.core.cell_model import CandidateCell, Cell
from tree_harness.core.llm_client import LLMClient, parse_llm_json
from tree_harness.store.tree_store import TreeStore


@dataclass
class DedupConfig:
    threshold_exact: float = 0.95
    threshold_similar: float = 0.85
    search_top_k: int = 5
    only_active: bool = True


@dataclass
class DedupResult:
    action: Literal["INSERT_NEW", "REINFORCE"]
    matched_cell_id: Optional[str] = None
    similarity_score: Optional[float] = None
    reason: Optional[str] = None


class Dedup:
    """Dedup 去重子例程 — 无状态,不创建 cell,不写 oplog。

    输出只是一个 DedupResult 信号,由 CambiumEngine 决定执行 INSERT_NEW
    还是触发 energy_system.reference()。
    """

    def __init__(self, tree_store: TreeStore, llm_client: LLMClient, config: DedupConfig):
        self.tree_store = tree_store
        self.llm_client = llm_client
        self.config = config

    def check(self, candidate: CandidateCell) -> DedupResult:
        """对候选 cell 执行去重判定。"""
        matches = self._vec_match(candidate.embedding)
        if not matches:
            return DedupResult(action="INSERT_NEW", reason="tree_empty_or_no_match")

        top_cell, top_score = matches[0]

        if top_score > self.config.threshold_exact:
            return DedupResult(
                action="REINFORCE",
                matched_cell_id=top_cell.id,
                similarity_score=top_score,
                reason="exact_match",
            )

        if top_score > self.config.threshold_similar:
            verdict = self._llm_arbitrate(candidate, top_cell, top_score)
            if verdict == "same":
                return DedupResult(
                    action="REINFORCE",
                    matched_cell_id=top_cell.id,
                    similarity_score=top_score,
                    reason="llm_arbitrate_same",
                )
            else:
                return DedupResult(
                    action="INSERT_NEW",
                    matched_cell_id=top_cell.id,
                    similarity_score=top_score,
                    reason="llm_arbitrate_different",
                )

        return DedupResult(
            action="INSERT_NEW",
            similarity_score=top_score,
            reason="below_threshold",
        )

    def _vec_match(self, embedding: List[float]) -> List[Tuple[Cell, float]]:
        """向量检索找最相似的已有 active cell。"""
        if not embedding:
            return []
        return self.tree_store.vec_search(
            embedding,
            top_k=self.config.search_top_k,
            min_score=0.0,
        )

    def _llm_arbitrate(
        self, candidate: CandidateCell, matched: Cell, score: float
    ) -> Literal["same", "different"]:
        """LLM 仲裁灰区情况 (temperature=0, 结果可缓存)。"""
        system_prompt = (
            "You are a dedup arbitration assistant. "
            "Determine if two pieces of knowledge express the same decision."
        )
        user_prompt = (
            f"已有知识:\n"
            f"- Decision: {matched.decision}\n"
            f"- Rationale: {matched.rationale}\n"
            f"- Domain: {matched.context_domain}\n\n"
            f"新候选:\n"
            f"- Decision: {candidate.decision}\n"
            f"- Rationale: {candidate.rationale}\n"
            f"- Domain: {candidate.context_domain}\n\n"
            f"Similarity score: {score}\n\n"
            f'输出格式: {{"verdict": "same"|"different", "reason": "..."}}'
        )
        response = self.llm_client.complete(system_prompt, user_prompt)
        try:
            result = parse_llm_json(response)
            return result.get("verdict", "different")
        except Exception:
            return "different"
