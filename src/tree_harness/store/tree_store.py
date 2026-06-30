"""TreeStore —— 统一对外接口层,协调 SQLiteBackend 与 KuzuDB 的读写。

设计原则:
- 不实现业务逻辑,只做双库协调和一致性保障
- 写入顺序: 先 OpLog → 再 SQLite → 再 KuzuDB
- 所有上层模块只通过 TreeStore 操作 tree
对应 spec: docs/specs/tree_store.md
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Optional, List, Tuple

from tree_harness.core.cell_model import Cell, Precondition, VerifyHint, RING_ORDER
from tree_harness.core.oplog import OpLog, OpEnum
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend

_ALL_STATUSES = ["active", "quarantined", "superseded", "archived"]


# ---------------------------------------------------------------------------
# Cell 序列化 (用于 oplog payload,支持 replay 重建)
# ---------------------------------------------------------------------------
def _serialize_cell(cell: Cell) -> dict:
    d = dataclasses.asdict(cell)
    d["created_at"] = cell.created_at.isoformat()
    return d


def _deserialize_cell(d: dict) -> Cell:
    preconds: List[Precondition] = []
    for p in d.get("context_preconditions", []):
        vh = None
        if p.get("verify_hint"):
            vh = VerifyHint(type=p["verify_hint"]["type"], params=p["verify_hint"]["params"])
        preconds.append(Precondition(kind=p["kind"], assertion=p["assertion"], verify_hint=vh))
    return Cell(
        id=d["id"], ring=d["ring"], maturity=d["maturity"], energy=d["energy"],
        context_trigger_task=d["context_trigger_task"],
        context_domain=d["context_domain"],
        context_preconditions=preconds,
        decision=d["decision"], rationale=d["rationale"],
        evidence=d.get("evidence", []), domain_tags=d.get("domain_tags", []),
        status=d["status"], source=d["source"],
        created_at=datetime.fromisoformat(d["created_at"]),
        superseded_by=d.get("superseded_by"),
    )


class TreeStore:
    """统一接口层: 协调 SQLite (细胞档案) 与 KuzuDB (年轮拓扑)。"""

    def __init__(self, sqlite: SQLiteBackend, kuzu: KuzuBackend, oplog: OpLog):
        self.sqlite = sqlite
        self.kuzu = kuzu
        self.oplog = oplog

    # ==================================================================
    # Cell 生命周期
    # ==================================================================
    def insert_cell(
        self, cell: Cell, rays: Optional[List[Tuple[str, float]]] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        """插入新 cell + 可选 ray。"""
        self.oplog.append(OpEnum.INSERT_CELL, {
            "cell_id": cell.id, "ring": cell.ring,
            "decision_summary": cell.decision[:80],
            "_cell": _serialize_cell(cell),
        }, episode_id)
        self.sqlite.insert_cell(cell)
        self.kuzu.add_cell_ref(cell.id, cell.ring)
        for target, weight in (rays or []):
            self.oplog.append(OpEnum.INSERT_RAY, {
                "from_id": cell.id, "to_id": target, "weight": weight,
            }, episode_id)
            self.kuzu.add_ray(cell.id, target, weight)

    def get_cell(self, cell_id: str) -> Optional[Cell]:
        return self.sqlite.get_cell(cell_id)

    def get_cells_batch(self, cell_ids: List[str]) -> List[Cell]:
        return [c for c in (self.sqlite.get_cell(i) for i in cell_ids) if c is not None]

    # ==================================================================
    # 字段更新
    # ==================================================================
    def update_energy(self, cell_id: str, new_energy: float, reason: str,
                      episode_id: Optional[str] = None) -> None:
        old = self.sqlite.get_cell(cell_id)
        old_energy = old.energy if old else 0.0
        self.oplog.append(OpEnum.UPDATE_ENERGY, {
            "cell_id": cell_id, "old_energy": old_energy,
            "new_energy": new_energy, "reason": reason,
        }, episode_id)
        self.sqlite.update_cell(cell_id, energy=new_energy)

    def update_maturity(self, cell_id: str, new_maturity: float,
                        episode_id: Optional[str] = None) -> None:
        old = self.sqlite.get_cell(cell_id)
        old_maturity = old.maturity if old else 0.0
        self.oplog.append(OpEnum.UPDATE_MATURITY, {
            "cell_id": cell_id, "old_maturity": old_maturity,
            "new_maturity": new_maturity,
        }, episode_id)
        self.sqlite.update_cell(cell_id, maturity=new_maturity)

    def promote(self, cell_id: str, from_ring: str, to_ring: str,
                episode_id: Optional[str] = None,
                reason: str = "normal") -> None:
        self.oplog.append(OpEnum.PROMOTE, {
            "cell_id": cell_id, "from_ring": from_ring, "to_ring": to_ring,
            "reason": reason,
        }, episode_id)
        self.sqlite.update_cell(cell_id, ring=to_ring)
        self.kuzu.update_cell_ring(cell_id, to_ring)

    def demote(self, cell_id: str, from_ring: str, to_ring: str,
               episode_id: Optional[str] = None,
               reason: str = "normal") -> None:
        self.oplog.append(OpEnum.DEMOTE, {
            "cell_id": cell_id, "from_ring": from_ring, "to_ring": to_ring,
            "reason": reason,
        }, episode_id)
        self.sqlite.update_cell(cell_id, ring=to_ring)
        self.kuzu.update_cell_ring(cell_id, to_ring)

    # ==================================================================
    # 合并 / 分裂 (木质化)
    # ==================================================================
    def merge_cells(self, source_ids: List[str], merged_cell: Cell,
                    rays: Optional[List[Tuple[str, float]]] = None,
                    episode_id: Optional[str] = None) -> None:
        """合并: source → superseded, merged_cell 接管引用。"""
        self.oplog.append(OpEnum.MERGE, {
            "source_ids": source_ids, "target_id": merged_cell.id,
        }, episode_id)
        # 插入 merged_cell
        self.oplog.append(OpEnum.INSERT_CELL, {
            "cell_id": merged_cell.id, "ring": merged_cell.ring,
            "decision_summary": merged_cell.decision[:80],
            "_cell": _serialize_cell(merged_cell),
        }, episode_id)
        self.sqlite.insert_cell(merged_cell)
        self.kuzu.add_cell_ref(merged_cell.id, merged_cell.ring)
        # 每个 source: supersede + lignified_from + sever outgoing rays
        for source_id in source_ids:
            self.oplog.append(OpEnum.SUPERSEDE, {
                "old_id": source_id, "new_id": merged_cell.id,
            }, episode_id)
            self.sqlite.update_cell(source_id, status="superseded",
                                    superseded_by=merged_cell.id)
            self.kuzu.add_supersedes(merged_cell.id, source_id)
            self.kuzu.add_lignified_from(merged_cell.id, source_id)
            # superseded == 死细胞, sever 其外向 active ray
            for ray in self.kuzu.get_outgoing_rays(source_id):
                if ray["status"] == "active":
                    self.oplog.append(OpEnum.SEVER_RAY, {
                        "from_id": source_id, "to_id": ray["to_id"],
                        "reason": "merged",
                    }, episode_id)
                    self.kuzu.update_ray_status(source_id, ray["to_id"], "severed")
        # 重定向 incoming rays → merged_cell (weight * 0.8)
        for source_id in source_ids:
            for ray in self.kuzu.get_incoming_rays(source_id):
                if ray["from_id"] not in source_ids:
                    new_weight = ray["weight"] * 0.8
                    self.oplog.append(OpEnum.INSERT_RAY, {
                        "from_id": ray["from_id"], "to_id": merged_cell.id,
                        "weight": new_weight,
                    }, episode_id)
                    self.kuzu.add_ray(ray["from_id"], merged_cell.id, new_weight)
        # merged_cell 的出射 ray
        for target, weight in (rays or []):
            self.oplog.append(OpEnum.INSERT_RAY, {
                "from_id": merged_cell.id, "to_id": target, "weight": weight,
            }, episode_id)
            self.kuzu.add_ray(merged_cell.id, target, weight)

    def split_cell(self, source_id: str, child_cells: List[Cell],
                   rays_map: Optional[dict] = None,
                   episode_id: Optional[str] = None) -> None:
        """分裂: source → superseded, children 接管。"""
        self.oplog.append(OpEnum.SPLIT, {
            "source_id": source_id, "target_ids": [c.id for c in child_cells],
        }, episode_id)
        # source supersede 到第一个 child
        self.oplog.append(OpEnum.SUPERSEDE, {
            "old_id": source_id, "new_id": child_cells[0].id,
        }, episode_id)
        self.sqlite.update_cell(source_id, status="superseded",
                                superseded_by=child_cells[0].id)
        # superseded == 死细胞, sever 其外向 active ray
        for ray in self.kuzu.get_outgoing_rays(source_id):
            if ray["status"] == "active":
                self.oplog.append(OpEnum.SEVER_RAY, {
                    "from_id": source_id, "to_id": ray["to_id"],
                    "reason": "split",
                }, episode_id)
                self.kuzu.update_ray_status(source_id, ray["to_id"], "severed")
        # 插入 children
        for child in child_cells:
            self.oplog.append(OpEnum.INSERT_CELL, {
                "cell_id": child.id, "ring": child.ring,
                "decision_summary": child.decision[:80],
                "_cell": _serialize_cell(child),
            }, episode_id)
            self.sqlite.insert_cell(child)
            self.kuzu.add_cell_ref(child.id, child.ring)
            self.kuzu.add_supersedes(child.id, source_id)
        # rays
        for child_id, rays in (rays_map or {}).items():
            for target, weight in rays:
                self.oplog.append(OpEnum.INSERT_RAY, {
                    "from_id": child_id, "to_id": target, "weight": weight,
                }, episode_id)
                self.kuzu.add_ray(child_id, target, weight)

    # ==================================================================
    # 隔离
    # ==================================================================
    def quarantine(self, cell_id: str, reason: str,
                   episode_id: Optional[str] = None) -> None:
        """隔离 cell: status→quarantined + 切断外向 active ray。"""
        self.oplog.append(OpEnum.QUARANTINE, {
            "cell_id": cell_id, "reason": reason,
        }, episode_id)
        self.sqlite.update_cell(cell_id, status="quarantined")
        for ray in self.kuzu.get_outgoing_rays(cell_id):
            if ray["status"] == "active":
                self.oplog.append(OpEnum.SEVER_RAY, {
                    "from_id": ray["from_id"], "to_id": ray["to_id"],
                    "reason": "quarantine",
                }, episode_id)
                self.kuzu.update_ray_status(ray["from_id"], ray["to_id"], "severed")

    def mark_for_review(self, cell_id: str, flag: bool = True,
                        reason: str = "", episode_id: Optional[str] = None) -> None:
        """标记 cell 需要人工/LLM review (DecaySentinel uncertain verdict 用)。

        信号级操作: 不改变 cell.status,只写 oplog MARK_REVIEW 供后续审查。
        """
        self.oplog.append(OpEnum.MARK_REVIEW, {
            "cell_id": cell_id, "flag": flag, "reason": reason,
        }, episode_id)

    # ==================================================================
    # Ray 操作
    # ==================================================================
    def add_ray(self, from_id: str, to_id: str, weight: float,
                episode_id: Optional[str] = None) -> None:
        self.oplog.append(OpEnum.INSERT_RAY, {
            "from_id": from_id, "to_id": to_id, "weight": weight,
        }, episode_id)
        self.kuzu.add_ray(from_id, to_id, weight)

    def update_ray_weight(self, from_id: str, to_id: str, new_weight: float,
                          episode_id: Optional[str] = None) -> None:
        old_weight = None
        for r in self.kuzu.get_outgoing_rays(from_id):
            if r["to_id"] == to_id:
                old_weight = r["weight"]
                break
        self.oplog.append(OpEnum.UPDATE_RAY_WEIGHT, {
            "from_id": from_id, "to_id": to_id,
            "old_weight": old_weight, "new_weight": new_weight,
        }, episode_id)
        self.kuzu.update_ray_weight(from_id, to_id, new_weight)

    def activate_ray(self, from_id: str, to_id: str,
                     episode_id: Optional[str] = None) -> None:
        """标记射线被激活 (更新 last_activated + 写 oplog)。"""
        self.oplog.append(OpEnum.RAY_ACTIVATED, {
            "from_id": from_id, "to_id": to_id,
        }, episode_id)
        self.kuzu.activate_ray(from_id, to_id)

    def sever_ray(self, from_id: str, to_id: str, reason: str,
                  episode_id: Optional[str] = None) -> None:
        self.oplog.append(OpEnum.SEVER_RAY, {
            "from_id": from_id, "to_id": to_id, "reason": reason,
        }, episode_id)
        self.kuzu.update_ray_status(from_id, to_id, "severed")

    # ==================================================================
    # 查询 (路由到合适后端)
    # ==================================================================
    def vec_search(self, query_embedding, top_k: int = 10, min_score: float = 0.5):
        return self.sqlite.vec_search(query_embedding, top_k=top_k, threshold=min_score)

    def query_by_ring(self, ring: str) -> List[Cell]:
        return self.sqlite.query_by_ring(ring)

    def list_by_ring(self, rings: List[str], status: str = "active") -> List[Cell]:
        """返回指定 ring 列表中指定 status 的 cell (before_step pinned 段用)。"""
        return self.sqlite.list_by_ring(rings, status=status)

    def get_incoming_rays(self, cell_id: str) -> List[dict]:
        return self.kuzu.get_incoming_rays(cell_id)

    def get_outgoing_rays(self, cell_id: str) -> List[dict]:
        return self.kuzu.get_outgoing_rays(cell_id)

    def get_in_degree(self, cell_id: str) -> int:
        return self.kuzu.get_in_degree(cell_id)

    def find_orphans(self) -> List[str]:
        return self.kuzu.find_orphans()

    def count_active_by_ring(self, ring: str) -> int:
        """返回指定 ring 中 active cell 的数量 (Lignification 容量检查用)。"""
        return self.sqlite.count_cells(ring=ring, status="active")

    def oldest_active_in_ring(self, ring: str, by: str = "maturity") -> Optional[Cell]:
        """返回指定 ring 中最老的 active cell (Lignification 溢出处理用)。"""
        return self.sqlite.oldest_active_in_ring(ring, by=by)

    # ==================================================================
    # 一致性
    # ==================================================================
    def consistency_check(self) -> List[str]:
        """扫描双库一致性,返回不一致的 cell_id 列表。

        扫描所有 status 的 cell (含 superseded/quarantined),
        因为这些 cell 在 KuzuDB 中仍应保留节点用于版本链回溯。
        """
        inconsistent = set()
        kuzu_refs = {r["id"]: r["ring"] for r in self.kuzu.get_all_refs()}
        # SQLite 任意 status 的 cell 在 KuzuDB 中应存在且 ring 一致
        for cell in self.sqlite.list_all():
            if cell.id not in kuzu_refs:
                inconsistent.add(cell.id)
            elif kuzu_refs[cell.id] != cell.ring:
                inconsistent.add(cell.id)
        # KuzuDB 中存在但 SQLite 不存在
        for kid in kuzu_refs:
            if self.sqlite.get_cell(kid) is None:
                inconsistent.add(kid)
        return sorted(inconsistent)

    def repair(self) -> List[str]:
        """修复双库不一致状态,以 SQLite 为 source of truth。

        策略:
        - SQLite 存在但 KuzuDB 缺失 → 在 KuzuDB 补建节点
        - SQLite 存在但 KuzuDB ring 不一致 → 更新 KuzuDB ring
        - KuzuDB 存在但 SQLite 不存在 → 从 KuzuDB 删除节点
        返回修复的 cell_id 列表。无不一致时返回空列表。
        """
        inconsistent = self.consistency_check()
        if not inconsistent:
            return []

        for cell_id in inconsistent:
            cell = self.sqlite.get_cell(cell_id)
            if cell is None:
                # SQLite 不存在但 KuzuDB 存在 → 从 KuzuDB 删除
                self.kuzu.remove_cell_ref(cell_id)
            else:
                kuzu_ring = self.kuzu.get_ring(cell_id)
                if kuzu_ring is None:
                    self.kuzu.add_cell_ref(cell_id, cell.ring)
                elif kuzu_ring != cell.ring:
                    self.kuzu.update_cell_ring(cell_id, cell.ring)

        # 验证修复结果
        still_bad = set(self.consistency_check())
        repaired = sorted(set(inconsistent) - still_bad)
        return repaired

    # ==================================================================
    # 统计
    # ==================================================================
    def stats(self) -> dict:
        by_ring = {r: self.sqlite.count_cells(ring=r, status="active") for r in RING_ORDER}
        by_status = {s: self.sqlite.count_cells(status=s) for s in _ALL_STATUSES}
        total_rays = 0
        active_rays = 0
        for ref in self.kuzu.get_all_refs():
            for ray in self.kuzu.get_outgoing_rays(ref["id"]):
                total_rays += 1
                if ray["status"] == "active":
                    active_rays += 1
        return {
            "total_cells": self.sqlite.count_cells(),
            "by_ring": by_ring,
            "by_status": by_status,
            "total_rays": total_rays,
            "active_rays": active_rays,
            "oplog_seq": self.oplog.get_latest_seq(),
        }

    # ==================================================================
    # replay 支持 (实现 ReplayTarget)
    # ==================================================================
    def apply_op(self, op: str, payload: dict, episode_id: Optional[str]) -> None:
        """应用单条 op (不写 op log,用于 replay 重建)。"""
        if op == OpEnum.INSERT_CELL:
            cell = _deserialize_cell(payload["_cell"])
            self.sqlite.insert_cell(cell)
            self.kuzu.add_cell_ref(cell.id, cell.ring)
        elif op == OpEnum.INSERT_RAY:
            self.kuzu.add_ray(payload["from_id"], payload["to_id"], payload["weight"])
        elif op == OpEnum.UPDATE_ENERGY:
            self.sqlite.update_cell(payload["cell_id"], energy=payload["new_energy"])
        elif op == OpEnum.UPDATE_MATURITY:
            self.sqlite.update_cell(payload["cell_id"], maturity=payload["new_maturity"])
        elif op == OpEnum.PROMOTE:
            self.sqlite.update_cell(payload["cell_id"], ring=payload["to_ring"])
            self.kuzu.update_cell_ring(payload["cell_id"], payload["to_ring"])
        elif op == OpEnum.DEMOTE:
            self.sqlite.update_cell(payload["cell_id"], ring=payload["to_ring"])
            self.kuzu.update_cell_ring(payload["cell_id"], payload["to_ring"])
        elif op == OpEnum.QUARANTINE:
            self.sqlite.update_cell(payload["cell_id"], status="quarantined")
        elif op == OpEnum.SUPERSEDE:
            self.sqlite.update_cell(payload["old_id"], status="superseded",
                                    superseded_by=payload["new_id"])
            self.kuzu.add_supersedes(payload["new_id"], payload["old_id"])
        elif op == OpEnum.SEVER_RAY:
            self.kuzu.update_ray_status(payload["from_id"], payload["to_id"], "severed")
        elif op == OpEnum.UPDATE_RAY_WEIGHT:
            self.kuzu.update_ray_weight(payload["from_id"], payload["to_id"],
                                        payload["new_weight"])
        elif op == OpEnum.RAY_ACTIVATED:
            self.kuzu.activate_ray(payload["from_id"], payload["to_id"])
        # MERGE / SPLIT / REINFORCE / MARK_REVIEW 是标记 op,实际重建由子 op 完成 → no-op
