"""RingPromotion —— maturity → ring 映射判定,含滞回防震荡、最小成熟期防早熟。

滞回带: demote_threshold 与 promote_threshold 之间为 dead zone,不发生升降。
跳级禁止: 只能逐级升/降。
对应 spec: docs/specs/ring_promotion.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from tree_harness.core.cell_model import RING_ORDER, PROMOTE_THRESHOLDS, DEMOTE_THRESHOLDS
from tree_harness.store.tree_store import TreeStore


@dataclass
class PromotionConfig:
    # 升层所需的最小成熟期 (episode 数),防早熟
    min_maturity_age: dict = field(default_factory=lambda: {
        "L0→L1": 3,
        "L1→L2": 10,
        "L2→L3": 30,
        "L3→L4": 100,
    })
    diversity_bonus: float = 0.1  # 跨 trigger 类型引用多样性奖励 (可选增强)


@dataclass
class PromotionReport:
    promoted: List[Tuple[str, str, str]]   # [(cell_id, from_ring, to_ring)]
    demoted: List[Tuple[str, str, str]]    # [(cell_id, from_ring, to_ring)]
    blocked: List[Tuple[str, str, str]]    # [(cell_id, target_ring, reason)]


class RingPromotion:
    """升降层调度: 扫描 active cell,执行 promote/demote 判定。"""

    def __init__(self, config: PromotionConfig, tree_store: TreeStore):
        self.config = config
        self.tree_store = tree_store
        self._episode = 0
        self._cell_birth: dict = {}  # cell_id → birth episode

    # ------------------------------------------------------------------
    # episode 计数 (用于最小成熟期判定)
    # ------------------------------------------------------------------
    def advance_episode(self) -> None:
        """推进一个 episode (在 episode 开始时调用)。"""
        self._episode += 1

    def register_cell(self, cell_id: str) -> None:
        """记录 cell 创建于当前 episode (若未注册则按当前 episode 计)。"""
        if cell_id not in self._cell_birth:
            self._cell_birth[cell_id] = self._episode

    def _cell_age(self, cell_id: str) -> int:
        """cell 自创建以来经过的 episode 数。"""
        self.register_cell(cell_id)
        return self._episode - self._cell_birth[cell_id]

    # ------------------------------------------------------------------
    # 单 cell 判定
    # ------------------------------------------------------------------
    def should_promote(self, cell, episode_count_since_creation: int) -> Optional[str]:
        """判断是否应升层,返回目标 ring 或 None。

        检查: maturity >= 阈值 AND episode_count >= min_maturity_age。
        """
        idx = RING_ORDER.index(cell.ring)
        if idx >= len(RING_ORDER) - 1:
            return None  # 已是 L4
        next_ring = RING_ORDER[idx + 1]
        threshold = PROMOTE_THRESHOLDS[next_ring]
        if cell.maturity < threshold:
            return None
        # 最小成熟期 (防早熟)
        min_age = self.config.min_maturity_age.get(f"{cell.ring}→{next_ring}", 0)
        if episode_count_since_creation < min_age:
            return None  # blocked: age 不足
        return next_ring

    def should_demote(self, cell) -> Optional[str]:
        """判断是否应降层,返回目标 ring 或 None。"""
        idx = RING_ORDER.index(cell.ring)
        if idx <= 0:
            return None  # 已是 L0
        threshold = DEMOTE_THRESHOLDS.get(cell.ring, 0.0)
        if cell.maturity < threshold:
            return RING_ORDER[idx - 1]
        return None

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    def execute_promotion(self, cell_id: str, target_ring: str,
                          episode_id: str) -> None:
        cell = self.tree_store.get_cell(cell_id)
        if cell is None:
            return
        self.tree_store.promote(cell_id, cell.ring, target_ring, episode_id)

    def execute_demotion(self, cell_id: str, target_ring: str,
                         episode_id: str) -> None:
        cell = self.tree_store.get_cell(cell_id)
        if cell is None:
            return
        self.tree_store.demote(cell_id, cell.ring, target_ring, episode_id)

    # ------------------------------------------------------------------
    # 批量评估
    # ------------------------------------------------------------------
    def evaluate_all(self, episode_id: str) -> PromotionReport:
        """扫描所有 active cell,执行升/降层判定,返回报告。"""
        report = PromotionReport(promoted=[], demoted=[], blocked=[])
        for cell in self.tree_store.sqlite.list_active():
            self.register_cell(cell.id)
            age = self._cell_age(cell.id)
            target = self.should_promote(cell, age)
            if target is not None:
                from_ring = cell.ring
                self.execute_promotion(cell.id, target, episode_id)
                report.promoted.append((cell.id, from_ring, target))
            else:
                # 检查是否 blocked (maturity 够但 age 不够)
                idx = RING_ORDER.index(cell.ring)
                if idx < len(RING_ORDER) - 1:
                    next_ring = RING_ORDER[idx + 1]
                    if cell.maturity >= PROMOTE_THRESHOLDS[next_ring]:
                        report.blocked.append(
                            (cell.id, next_ring, "min_maturity_age_not_met")
                        )
                # 降层判定
                demote_target = self.should_demote(cell)
                if demote_target is not None:
                    from_ring = cell.ring
                    self.execute_demotion(cell.id, demote_target, episode_id)
                    report.demoted.append((cell.id, from_ring, demote_target))
        return report
