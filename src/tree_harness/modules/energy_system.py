"""EnergySystem —— 管理所有 cell 的能量更新逻辑,驱动生命周期 (成长→木质化→腐朽)。

核心公式:
  能量:   E += δ_ref * ref_count - |δ_chal| * chal_count ;  E *= (1 - decay_rate[ring])
  成熟度: maturity += α * tanh(E / E_norm) - β * decay_rate[ring] ;  clip [0, 1]
对应 spec: docs/specs/energy_system.md
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from tree_harness.core.cell_model import RING_ORDER, PROMOTE_THRESHOLDS, DEMOTE_THRESHOLDS
from tree_harness.store.tree_store import TreeStore


@dataclass
class EnergyConfig:
    # δ 增量
    delta_reference: float = 0.10    # 一次成功引用
    delta_challenge: float = -0.15   # 一次挑战/否定 (不对称)

    # 每 episode 乘性衰减
    decay_rates: dict = field(default_factory=lambda: {
        "L0": 0.30,
        "L1": 0.10,
        "L2": 0.03,
        "L3": 0.01,
        "L4": 0.002,
    })

    # maturity 更新参数
    alpha: float = 0.05    # 引用驱动的成熟速率
    beta: float = 0.02     # 自然成熟惩罚
    e_norm: float = 1.0    # 归一化尺度

    # 腐朽候选阈值
    energy_threshold: float = -0.20
    idle_thresholds: dict = field(default_factory=lambda: {
        "L0": 2, "L1": 8, "L2": 30, "L3": 100, "L4": 500,
    })


class EnergySystem:
    """能量系统: reference / challenge / decay / maturity 更新。"""

    def __init__(self, config: EnergyConfig, tree_store: TreeStore):
        self.config = config
        self.tree_store = tree_store

    # ------------------------------------------------------------------
    def _decay_rate(self, cell) -> float:
        """user_directive cell 不自然衰减 (decay_rate=0)。"""
        if cell.source == "user_directive":
            return 0.0
        return self.config.decay_rates.get(cell.ring, 0.0)

    # ------------------------------------------------------------------
    # 引用 / 挑战 (episode 内可多次调用)
    # ------------------------------------------------------------------
    def reference(self, cell_id: str, episode_id: str) -> bool:
        """cell 被成功引用: energy += delta_reference。
        返回 True 表示成功强化, False 表示目标不存在或非 active。
        """
        cell = self.tree_store.get_cell(cell_id)
        if cell is None or cell.status != "active":
            return False
        new_energy = cell.energy + self.config.delta_reference
        self.tree_store.update_energy(cell_id, new_energy, "reference", episode_id)
        return True

    def challenge(self, cell_id: str, episode_id: str) -> bool:
        """cell 被挑战/否定: energy += delta_challenge (负值)。
        返回 True 表示成功施加挑战, False 表示目标不存在或非 active。
        """
        cell = self.tree_store.get_cell(cell_id)
        if cell is None or cell.status != "active":
            return False
        new_energy = cell.energy + self.config.delta_challenge
        self.tree_store.update_energy(cell_id, new_energy, "challenge", episode_id)
        return True

    def decay_one(self, cell_id: str, delta: float = -0.05,
                  episode_id: Optional[str] = None) -> bool:
        """对单个 cell 施加小幅能量衰减 (DecaySentinel uncertain verdict 用)。

        与 challenge 不同: decay_one 是 Sentinel 的信号级副作用,
        表示"无法确定 cell 是否有效,先减一点能量观察"。
        """
        cell = self.tree_store.get_cell(cell_id)
        if cell is None or cell.status != "active":
            return False
        new_energy = cell.energy + delta
        self.tree_store.update_energy(cell_id, new_energy, "decay_one", episode_id)
        return True

    # ------------------------------------------------------------------
    # 衰减 (每 episode 结束调用一次)
    # ------------------------------------------------------------------
    def decay_all(self, episode_id: str) -> None:
        """对所有 active cell 执行自然衰减: energy *= (1 - decay_rate[ring])。"""
        for cell in self.tree_store.sqlite.list_active():
            rate = self._decay_rate(cell)
            if rate == 0.0:
                continue
            new_energy = cell.energy * (1.0 - rate)
            self.tree_store.update_energy(cell.id, new_energy, "decay", episode_id)

    # ------------------------------------------------------------------
    # 成熟度更新
    # ------------------------------------------------------------------
    def update_maturity(self, cell_id: str, episode_id: Optional[str] = None) -> None:
        """更新单个 cell 的 maturity: maturity += α*tanh(E/E_norm) - β*decay_rate。"""
        cell = self.tree_store.get_cell(cell_id)
        if cell is None or cell.status != "active":
            return
        rate = self._decay_rate(cell)
        delta = (
            self.config.alpha * math.tanh(cell.energy / self.config.e_norm)
            - self.config.beta * rate
        )
        new_maturity = max(0.0, min(1.0, cell.maturity + delta))
        self.tree_store.update_maturity(cell_id, new_maturity, episode_id)

    def update_all_maturity(self, episode_id: str) -> None:
        """对所有 active cell 更新 maturity。"""
        for cell in self.tree_store.sqlite.list_active():
            self.update_maturity(cell.id, episode_id)

    # ------------------------------------------------------------------
    # 候选查询
    # ------------------------------------------------------------------
    def get_decay_candidates(self, limit: Optional[int] = None) -> List[str]:
        """返回 energy < energy_threshold 的 active cell id。

        limit 用于 OuterHarness after_step 抽样验证 (funnel_sample_size)。
        (idle 判定依赖 ray 拓扑时间戳,由 DecaySentinel 在 Phase 3 补充。)
        """
        cells = self.tree_store.sqlite.query_decay_candidates(
            self.config.energy_threshold, limit=limit
        )
        return [c.id for c in cells]

    def get_promotion_candidates(self) -> List[Tuple[str, str]]:
        """返回 maturity 跨升层阈值的 cell: [(cell_id, target_ring)] (逐级,禁跳级)。"""
        candidates: List[Tuple[str, str]] = []
        for cell in self.tree_store.sqlite.list_active():
            idx = RING_ORDER.index(cell.ring)
            if idx >= len(RING_ORDER) - 1:
                continue  # 已是 L4
            next_ring = RING_ORDER[idx + 1]
            if cell.maturity >= PROMOTE_THRESHOLDS[next_ring]:
                candidates.append((cell.id, next_ring))
        return candidates

    def get_demotion_candidates(self) -> List[Tuple[str, str]]:
        """返回 maturity 低于降级阈值的 cell: [(cell_id, target_ring)]。"""
        candidates: List[Tuple[str, str]] = []
        for cell in self.tree_store.sqlite.list_active():
            idx = RING_ORDER.index(cell.ring)
            if idx <= 0:
                continue  # 已是 L0
            threshold = DEMOTE_THRESHOLDS.get(cell.ring, 0.0)
            if cell.maturity < threshold:
                candidates.append((cell.id, RING_ORDER[idx - 1]))
        return candidates
