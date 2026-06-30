"""Connector 子例程 — Cambium Engine 三步管线 Step C。

为新创建的 cell 建立 RAY 连接 (外→内方向):
新 cell 指向已有的同层或更内层 cell,表示 "我引用/依赖了你"。

无状态子例程: 不调其他算符、不写 oplog (oplog 写入由 TreeStore.add_ray 附带完成)。
对应 spec: docs/specs/connector.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from tree_harness.core.cell_model import Cell, RING_ORDER
from tree_harness.core.embedding import embed_cell_text
from tree_harness.store.tree_store import TreeStore


@dataclass
class ConnectorConfig:
    search_top_k: int = 10
    search_threshold: float = 0.5
    max_rays_per_cell: int = 5
    domain_overlap_bonus: float = 0.2


_RING_INDEX = {ring: idx for idx, ring in enumerate(RING_ORDER)}


class Connector:
    """Connector 连接子例程 — 无状态,不调其他算符,不写 oplog。"""

    def __init__(self, tree_store: TreeStore, config: ConnectorConfig):
        self.tree_store = tree_store
        self.config = config

    def connect(
        self, new_cell: Cell, episode_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """为 new_cell 寻找并建立 ray 连接。

        返回实际建立的 ray 列表: [(target_cell_id, weight)]
        """
        candidates = self._find_candidates(new_cell)
        if not candidates:
            return []

        # 计算权重
        weighted = [
            (cell, sim, self._compute_weight(new_cell, cell, sim))
            for cell, sim in candidates
        ]
        selected = self._select_top(weighted)

        # 建立 ray
        rays: List[Tuple[str, float]] = []
        for cell, weight in selected:
            self.tree_store.add_ray(
                new_cell.id, cell.id, weight, episode_id=episode_id
            )
            rays.append((cell.id, weight))
        return rays

    def _find_candidates(self, new_cell: Cell) -> List[Tuple[Cell, float]]:
        """向量检索 + ring 过滤 + 自身排除。"""
        embedder = self.tree_store.sqlite.embedder
        query_embedding = embedder.embed(
            embed_cell_text(new_cell.decision, new_cell.rationale)
        )
        results = self.tree_store.vec_search(
            query_embedding,
            top_k=self.config.search_top_k,
            min_score=self.config.search_threshold,
        )
        # 过滤: ring >= new_cell.ring (外→内原则), 排除自身
        new_ring_idx = _RING_INDEX[new_cell.ring]
        return [
            (cell, sim) for cell, sim in results
            if _RING_INDEX[cell.ring] >= new_ring_idx and cell.id != new_cell.id
        ]

    def _compute_weight(
        self, new_cell: Cell, target: Cell, similarity: float
    ) -> float:
        """计算 ray weight = similarity * (1 + domain_bonus), clip to [0, 1]。"""
        overlap = len(set(new_cell.domain_tags) & set(target.domain_tags))
        domain_bonus = min(overlap * self.config.domain_overlap_bonus, 0.4)
        weight = similarity * (1 + domain_bonus)
        return min(weight, 1.0)

    def _select_top(
        self, candidates: List[Tuple[Cell, float, float]]
    ) -> List[Tuple[Cell, float]]:
        """按 weight 降序取 top-k。"""
        sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        top = sorted_candidates[: self.config.max_rays_per_cell]
        return [(cell, weight) for cell, _, weight in top]
