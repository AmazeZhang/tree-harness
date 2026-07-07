"""LignificationScheduler — 树的木质化 (ring promotion / merge / split)。

无状态算法服务,由 OuterHarness.after_episode() 在 episode 末调用。
执行 promote / demote / merge / split 算符,管理 ring capacity 溢出。

对应 spec: docs/specs/lignification.md
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from tree_harness.core.cell_model import (
    Cell, create_cell, RING_ORDER,
    PROMOTE_THRESHOLDS, DEMOTE_THRESHOLDS,
)
from tree_harness.core.embedding import embed_cell_text
from tree_harness.core.llm_client import parse_llm_json
from tree_harness.core.oplog import OpLog, OpEnum
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class LignificationConfig:
    """Lignification 维护周期参数。

    Ring 阈值不在此处定义,唯一权威源是 cell_model.PROMOTE_THRESHOLDS /
    DEMOTE_THRESHOLDS。滞回带 = 0.10 (任意层),由 cell_model 不变量保证。
    """
    ring_capacity: dict = field(default_factory=lambda: {
        "L0": 50, "L1": 30, "L2": 20,
        "L3": 60, "L4": 20,
    })
    overflow_policy: str = "force_promote"   # "force_promote" | "demote_oldest" | "block_new"

    # Merge / split
    merge_similarity_threshold: float = 0.82  # P2: 从 0.92 降低,防止近重复 cell 堆积
    merge_max_cluster_size: int = 5
    enable_split: bool = False               # 默认关闭,仅 LLM 主动发现时启用

    # 防早熟: 升层所需最小成熟期 (episode 数)
    min_maturity_age: dict = field(default_factory=lambda: {
        "L0→L1": 3,
        "L1→L2": 10,
        "L2→L3": 30,
        "L3→L4": 100,
    })


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------
@dataclass
class MaintenanceResult:
    """run_maintenance_cycle 的返回值,字段与 EpisodeReport 对齐。"""
    promoted: List[Tuple[str, str, str]]    # (cell_id, from_ring, to_ring)
    demoted: List[Tuple[str, str, str]]
    merged: List[Tuple[List[str], str]]     # (source_ids, merged_cell_id)
    split: List[Tuple[str, List[str]]]      # (source_id, child_ids)
    archived: List[str]                      # P0-2: 容量淘汰的 cell_id
    op_counts: dict                          # {PROMOTE, DEMOTE, MERGE, SPLIT, ARCHIVE}


# ---------------------------------------------------------------------------
# LignificationScheduler
# ---------------------------------------------------------------------------
class LignificationScheduler:
    """木质化调度器 — 管理 ring promotion / merge / split。"""

    def __init__(
        self,
        tree_store: TreeStore,
        energy_system: EnergySystem,
        llm_client,
        oplog: OpLog,
        config: LignificationConfig,
    ):
        self.tree_store = tree_store
        self.energy_system = energy_system
        self.llm_client = llm_client
        self.oplog = oplog
        self.config = config
        # episode 跟踪 (用于 min_maturity_age 检查)
        self._episode_count = 0
        self._cell_birth: dict[str, int] = {}

    # ------------------------------------------------------------------
    # episode 计数 (用于最小成熟期判定)
    # ------------------------------------------------------------------
    def advance_episode(self) -> None:
        """推进一个 episode (在 run_maintenance_cycle 开始时调用)。"""
        self._episode_count += 1

    def _register_cell(self, cell_id: str) -> None:
        """记录 cell 创建于当前 episode (若未注册)。"""
        if cell_id not in self._cell_birth:
            self._cell_birth[cell_id] = self._episode_count

    def _cell_age(self, cell_id: str) -> int:
        """cell 自创建以来经过的 episode 数。"""
        self._register_cell(cell_id)
        return self._episode_count - self._cell_birth[cell_id]

    # ------------------------------------------------------------------
    # Promote (升层)
    # ------------------------------------------------------------------
    def check_promotions(self, episode_id: str) -> List[Tuple[str, str, str]]:
        """检查并执行所有待升层的 cell,返回 [(cell_id, from_ring, to_ring)]。

        检查: maturity >= 阈值 AND cell_age >= min_maturity_age。
        """
        candidates = self.energy_system.get_promotion_candidates()
        promoted: List[Tuple[str, str, str]] = []
        for cell_id, target_ring in candidates:
            cell = self.tree_store.get_cell(cell_id)
            if cell is None or cell.status != "active":
                continue
            # 防早熟: 检查最小成熟期
            self._register_cell(cell_id)
            age = self._cell_age(cell_id)
            min_age = self.config.min_maturity_age.get(
                f"{cell.ring}→{target_ring}", 0,
            )
            if age < min_age:
                continue  # blocked: age 不足
            # 容量检查
            overflow = self._enforce_capacity(target_ring, cell_id, episode_id)
            if overflow == "block_new":
                continue
            if overflow == "overflow_force" and target_ring == "L3":
                # 直接升 L4
                target_ring = "L4"
                self._enforce_capacity("L4", cell_id, episode_id)
            # 执行升层
            reason = "overflow_force" if overflow == "overflow_force" else "normal"
            self.tree_store.promote(
                cell_id, cell.ring, target_ring,
                episode_id=episode_id, reason=reason,
            )
            promoted.append((cell_id, cell.ring, target_ring))
        return promoted

    def check_demotions(self, episode_id: str) -> List[Tuple[str, str, str]]:
        """检查并执行所有待降层的 cell。"""
        candidates = self.energy_system.get_demotion_candidates()
        demoted: List[Tuple[str, str, str]] = []
        for cell_id, target_ring in candidates:
            cell = self.tree_store.get_cell(cell_id)
            if cell is None or cell.status != "active":
                continue
            self.tree_store.demote(
                cell_id, cell.ring, target_ring,
                episode_id=episode_id, reason="normal",
            )
            demoted.append((cell_id, cell.ring, target_ring))
        return demoted

    # ------------------------------------------------------------------
    # Capacity enforcement
    # ------------------------------------------------------------------
    def _enforce_capacity(self, target_ring: str, _incoming_cell_id: str = "",
                          episode_id: Optional[str] = None) -> Optional[str]:
        """返回触发的 overflow reason,若无溢出返回 None。

        注意: 此方法会执行溢出处理 (demote/archive 等),不仅是检查。

        - L0 溢出: archive 最低 energy 的 cell (底部漏斗,清除僵尸)
        - L1/L2 溢出: demote 最低 maturity 的 cell 到下一层
        - L3 溢出: force_promote 策略 → 直接升 L4
        - L4 溢出: demote 最低 maturity 的 cell 到 L3
        """
        if target_ring not in self.config.ring_capacity:
            return None
        active_count = self.tree_store.count_active_by_ring(target_ring)
        if active_count < self.config.ring_capacity[target_ring]:
            return None

        # L0 特殊处理: 底部漏斗,archive 最低 energy cell
        if target_ring == "L0":
            # 找最低 energy 的非 user_directive cell
            candidates = self.tree_store.list_by_ring([target_ring], status="active")
            evictable = [c for c in candidates if c.source != "user_directive"
                         and c.id != _incoming_cell_id]
            if evictable:
                victim = min(evictable, key=lambda c: (c.energy, c.maturity))
                self.tree_store.archive_cell(
                    victim.id, reason="capacity_overflow",
                    episode_id=episode_id,
                )
                return "archived"
            return None

        policy = self.config.overflow_policy
        if policy == "force_promote" and target_ring == "L3":
            return "overflow_force"     # 调用方应直接升 L4

        if policy in ("force_promote", "demote_oldest"):
            victim = self.tree_store.oldest_active_in_ring(target_ring, by="maturity")
            if victim is not None and victim.id != _incoming_cell_id:
                # 将 victim 降一层
                idx = RING_ORDER.index(target_ring)
                if idx > 0:
                    demote_to = RING_ORDER[idx - 1]
                    self.tree_store.demote(
                        victim.id, target_ring, demote_to,
                        episode_id=episode_id, reason="overflow_demote",
                    )
                return "overflow_demote"

        if policy == "block_new":
            return "block_new"

        return None

    # ------------------------------------------------------------------
    # Merge (合并)
    # ------------------------------------------------------------------
    def attempt_merge(self, candidate_ids: List[str],
                      episode_id: Optional[str] = None) -> Optional[str]:
        """尝试合并一组 cell,返回新 cell id (如果成功)。"""
        if len(candidate_ids) < 2:
            return None
        cells = self.tree_store.get_cells_batch(candidate_ids)
        if len(cells) < 2:
            return None

        # 验证: 同 ring、同 domain_tag、都 active
        rings = {c.ring for c in cells}
        if len(rings) > 1:
            return None
        if not all(c.status == "active" for c in cells):
            return None

        # LLM 生成合并后的内容
        merged_content = self._llm_merge(cells)
        if merged_content is None:
            return None

        # 计算继承属性
        source_energies = [c.energy for c in cells]
        source_maturities = [c.maturity for c in cells]
        merged_energy = max(source_energies) * 0.8
        merged_maturity = max(source_maturities)  # 保留最高成熟度,避免合并拉低

        # 确定 ring (基于 maturity)
        merged_ring = self._ring_for_maturity(merged_maturity, cells[0].ring)

        # 收集 domain_tags
        all_tags = set()
        for c in cells:
            all_tags.update(c.domain_tags)

        # 收集 evidence
        all_evidence = []
        for c in cells:
            all_evidence.extend(c.evidence)

        # 收集 preconditions
        all_preconds = []
        for c in cells:
            all_preconds.extend(c.context_preconditions)

        # 创建新 cell
        new_cell = create_cell(
            source="distilled",
            trigger_task=cells[0].context_trigger_task,
            domain=cells[0].context_domain,
            decision=merged_content.get("decision", ""),
            rationale=merged_content.get("rationale", ""),
            preconditions=all_preconds,
            evidence=list({json.dumps(e, sort_keys=True, ensure_ascii=False) if isinstance(e, dict) else e for e in all_evidence}),
            domain_tags=list(all_tags),
            ring=merged_ring,
            maturity=merged_maturity,
            energy=merged_energy,
        )

        # 执行合并 (TreeStore facade)
        self.tree_store.merge_cells(
            source_ids=candidate_ids,
            merged_cell=new_cell,
            episode_id=episode_id,
        )

        return new_cell.id

    def _llm_merge(self, cells: List[Cell]) -> Optional[dict]:
        """LLM 生成合并后的 decision/rationale。"""
        system_prompt = (
            "You are a knowledge consolidation engine. Given multiple knowledge cells "
            "that express different aspects of the same concept, produce a single merged "
            "cell that captures the unified understanding. "
            'Respond with JSON: {"decision": "...", "rationale": "..."}'
        )
        cell_descriptions = "\n".join(
            f"- Cell {c.id} [{c.ring}]: {c.decision} — {c.rationale}"
            for c in cells
        )
        user_prompt = f"Cells to merge:\n{cell_descriptions}\n"

        raw = self.llm_client.complete(system_prompt, user_prompt)
        parsed = parse_llm_json(raw)
        if not parsed.get("decision"):
            return None
        return parsed

    # ------------------------------------------------------------------
    # Split (分裂)
    # ------------------------------------------------------------------
    def attempt_split(self, cell_id: str,
                      episode_id: Optional[str] = None) -> Optional[List[str]]:
        """尝试分裂一个 cell,返回新 cell id 列表 (如果成功)。"""
        if not self.config.enable_split:
            return None

        cell = self.tree_store.get_cell(cell_id)
        if cell is None or cell.status != "active":
            return None

        split_contents = self._llm_split(cell)
        if not split_contents or len(split_contents) < 2:
            return None

        child_cells: List[Cell] = []
        for content in split_contents:
            child = create_cell(
                source="distilled",
                trigger_task=cell.context_trigger_task,
                domain=cell.context_domain,
                decision=content.get("decision", ""),
                rationale=content.get("rationale", ""),
                preconditions=cell.context_preconditions,
                evidence=cell.evidence,
                domain_tags=cell.domain_tags,
                ring=self._ring_for_maturity(cell.maturity * 0.8, cell.ring),
                maturity=cell.maturity * 0.8,
                energy=cell.energy * 0.6,
            )
            child_cells.append(child)

        self.tree_store.split_cell(
            source_id=cell_id,
            child_cells=child_cells,
            episode_id=episode_id,
        )
        return [c.id for c in child_cells]

    def _llm_split(self, cell: Cell) -> Optional[List[dict]]:
        """LLM 判断 cell 是否应分裂,并生成分裂内容。"""
        system_prompt = (
            "You are a knowledge analysis engine. Given a knowledge cell, determine if it "
            "actually contains multiple independent concepts that should be split. If so, "
            "produce the split contents. If not, return an empty list. "
            'Respond with JSON: [{"decision": "...", "rationale": "..."}, ...]'
        )
        user_prompt = (
            f"Cell {cell.id} [{cell.ring}]:\n"
            f"Decision: {cell.decision}\n"
            f"Rationale: {cell.rationale}\n"
        )

        raw = self.llm_client.complete(system_prompt, user_prompt)
        parsed = parse_llm_json(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "cells" in parsed:
            return parsed["cells"]
        return None

    # ------------------------------------------------------------------
    # Merge candidate discovery
    # ------------------------------------------------------------------
    def _find_merge_candidates(self) -> List[List[str]]:
        """找到可合并的 cell 簇 (同 ring, 高相似度, 同 domain_tag)。"""
        embedder = self.tree_store.sqlite.embedder
        if embedder is None:
            return []

        candidates: List[List[str]] = []
        threshold = self.config.merge_similarity_threshold
        max_cluster = self.config.merge_max_cluster_size

        # 按 ring 分组
        for ring in RING_ORDER:
            cells = self.tree_store.list_by_ring([ring], status="active")
            if len(cells) < 2:
                continue

            # 计算所有 embedding
            embeddings = {}
            for c in cells:
                embeddings[c.id] = embedder.embed(
                    embed_cell_text(c.decision, c.rationale)
                )

            # 贪心聚类
            used = set()
            for i, ci in enumerate(cells):
                if ci.id in used:
                    continue
                cluster = [ci.id]
                used.add(ci.id)
                for j in range(i + 1, len(cells)):
                    cj = cells[j]
                    if cj.id in used:
                        continue
                    # 检查 domain_tag 交集
                    if not (set(ci.domain_tags) & set(cj.domain_tags)):
                        continue
                    sim = self._cosine_sim(embeddings[ci.id], embeddings[cj.id])
                    if sim >= threshold:
                        cluster.append(cj.id)
                        used.add(cj.id)
                        if len(cluster) >= max_cluster:
                            break
                if len(cluster) >= 2:
                    candidates.append(cluster)

        return candidates

    # ------------------------------------------------------------------
    # Maintenance cycle
    # ------------------------------------------------------------------
    def run_maintenance_cycle(self, episode_id: str) -> MaintenanceResult:
        """执行一轮完整的木质化维护,返回结构化结果。"""
        # 先注册当前所有 active cell (在 advance_episode 之前,这样新 cell 的 birth = 当前 episode)
        for cell in self.tree_store.sqlite.list_active():
            self._register_cell(cell.id)
        self.advance_episode()
        promoted = self.check_promotions(episode_id)
        demoted = self.check_demotions(episode_id)

        # 容量扫描: 检查所有有 capacity 限制的 ring
        # (check_promotions 只检查升层目标 ring,这里补充检查所有 ring)
        archived: List[str] = []
        for ring in RING_ORDER:
            if ring in self.config.ring_capacity:
                active_count = self.tree_store.count_active_by_ring(ring)
                if active_count > self.config.ring_capacity[ring]:
                    result = self._enforce_capacity(ring, episode_id=episode_id)
                    if result == "archived":
                        archive_entries = self.oplog.get_entries(op_filter="ARCHIVE")
                        if archive_entries:
                            archived.append(archive_entries[-1].payload.get("cell_id", ""))

        merged: List[Tuple[List[str], str]] = []
        split: List[Tuple[str, List[str]]] = []

        # Merge
        clusters = self._find_merge_candidates()
        for cluster in clusters:
            new_id = self.attempt_merge(cluster, episode_id=episode_id)
            if new_id:
                merged.append((cluster, new_id))

        # Split (默认关闭)
        if self.config.enable_split:
            for cell_id in self._find_split_candidates():
                children = self.attempt_split(cell_id, episode_id=episode_id)
                if children:
                    split.append((cell_id, children))

        op_counts = {
            "PROMOTE": len(promoted),
            "DEMOTE": len(demoted),
            "MERGE": len(merged),
            "SPLIT": len(split),
            "ARCHIVE": len(archived),
        }
        return MaintenanceResult(
            promoted=promoted,
            demoted=demoted,
            merged=merged,
            split=split,
            archived=archived,
            op_counts=op_counts,
        )

    def _find_split_candidates(self) -> List[str]:
        """找到可能需要分裂的 cell (默认返回空,由 LLM 主动发现时启用)。"""
        return []

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        """计算余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _ring_for_maturity(maturity: float, default_ring: str) -> str:
        """根据 maturity 确定 ring (使用 MATURITY_RING_RANGES)。"""
        from tree_harness.core.cell_model import MATURITY_RING_RANGES
        for ring, (low, high) in MATURITY_RING_RANGES.items():
            if low <= maturity < high:
                return ring
        if maturity >= 1.0:
            return "L4"
        return default_ring
