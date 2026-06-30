"""KuzuBackend —— 年轮拓扑存储与图查询。

存储 cell 的引用关系 (RAY)、版本关系 (SUPERSEDES)、木质化来源 (LIGNIFIED_FROM)。
对应 spec: docs/specs/kuzu_backend.md
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, List

import kuzu

from tree_harness.core.cell_model import RING_ORDER

logger = logging.getLogger(__name__)


class KuzuBackend:
    """年轮拓扑: KuzuDB 嵌入式图数据库。"""

    def __init__(self, db_path: str):
        # db_path 为目录路径; KuzuDB 以目录形式存储
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self.init_db()

    # ------------------------------------------------------------------
    def init_db(self) -> None:
        """初始化 KuzuDB,创建 schema。"""
        self._execute(
            "CREATE NODE TABLE IF NOT EXISTS CellRef("
            "id STRING, ring STRING, PRIMARY KEY (id))"
        )
        self._execute(
            "CREATE REL TABLE IF NOT EXISTS RAY("
            "FROM CellRef TO CellRef, weight DOUBLE, status STRING, "
            "created_at STRING, last_activated STRING)"
        )
        self._execute(
            "CREATE REL TABLE IF NOT EXISTS SUPERSEDES("
            "FROM CellRef TO CellRef, created_at STRING)"
        )
        self._execute(
            "CREATE REL TABLE IF NOT EXISTS LIGNIFIED_FROM("
            "FROM CellRef TO CellRef, created_at STRING)"
        )

    def _execute(self, cypher: str, params: Optional[dict] = None) -> None:
        self._conn.execute(cypher, params or {})

    def _query_all(self, cypher: str, params: Optional[dict] = None) -> List[dict]:
        result = self._conn.execute(cypher, params or {})
        cols = result.get_column_names()
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append(dict(zip(cols, row)))
        return rows

    # ------------------------------------------------------------------
    # Node 操作
    # ------------------------------------------------------------------
    def add_cell_ref(self, cell_id: str, ring: str) -> None:
        """添加 cell 节点引用 (幂等: 已存在则更新 ring)。"""
        self._execute(
            "MERGE (c:CellRef {id: $id}) SET c.ring = $ring",
            {"id": cell_id, "ring": ring},
        )

    def update_cell_ring(self, cell_id: str, new_ring: str) -> None:
        self._execute(
            "MATCH (c:CellRef {id: $id}) SET c.ring = $ring",
            {"id": cell_id, "ring": new_ring},
        )

    def remove_cell_ref(self, cell_id: str) -> None:
        """移除节点及其所有关系 (仅在 cell 被彻底删除时)。"""
        self._execute(
            "MATCH (c:CellRef {id: $id}) DETACH DELETE c", {"id": cell_id}
        )

    def get_ring(self, cell_id: str) -> Optional[str]:
        rows = self._query_all(
            "MATCH (c:CellRef {id: $id}) RETURN c.ring AS ring", {"id": cell_id}
        )
        return rows[0]["ring"] if rows else None

    def get_all_refs(self) -> List[dict]:
        """返回所有 CellRef 节点 [{id, ring}]。"""
        return self._query_all(
            "MATCH (c:CellRef) RETURN c.id AS id, c.ring AS ring"
        )

    # ------------------------------------------------------------------
    # RAY 操作 (from=外层引用方 → to=内层被引用方)
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_weight(w: float) -> float:
        return max(0.0, min(1.0, float(w)))  # 行为契约 4

    def _check_ray_direction(self, from_id: str, to_id: str) -> None:
        """行为契约 3: from 的 ring 层级应 <= to (外→内),否则 warning。"""
        rows = self._query_all(
            "MATCH (a:CellRef {id: $f}), (b:CellRef {id: $t}) "
            "RETURN a.ring AS fr, b.ring AS tr",
            {"f": from_id, "t": to_id},
        )
        if rows:
            fr, tr = rows[0]["fr"], rows[0]["tr"]
            if fr in RING_ORDER and tr in RING_ORDER:
                if RING_ORDER.index(fr) > RING_ORDER.index(tr):
                    logger.warning(
                        "RAY 方向违反外→内原则: %s(%s) → %s(%s)",
                        from_id, fr, to_id, tr,
                    )

    def add_ray(self, from_id: str, to_id: str, weight: float) -> None:
        """添加射线 (幂等: 已存在则不重建,保留 created_at/last_activated)。

        如需更新已有 ray 的 weight,请用 update_ray_weight。
        """
        weight = self._clip_weight(weight)
        self._check_ray_direction(from_id, to_id)
        # 检查是否已存在 — 存在则不重建,保留时间戳
        existing = self._query_all(
            "MATCH (a:CellRef {id: $f})-[r:RAY]->(b:CellRef {id: $t}) "
            "RETURN r.weight AS w",
            {"f": from_id, "t": to_id},
        )
        if existing:
            return
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            "MATCH (a:CellRef {id: $f}), (b:CellRef {id: $t}) "
            "CREATE (a)-[:RAY {weight: $w, status: 'active', "
            "created_at: $now, last_activated: $now}]->(b)",
            {"f": from_id, "t": to_id, "w": weight, "now": now},
        )

    def update_ray_weight(self, from_id: str, to_id: str, new_weight: float) -> None:
        new_weight = self._clip_weight(new_weight)
        self._execute(
            "MATCH (a:CellRef {id: $f})-[r:RAY]->(b:CellRef {id: $t}) "
            "SET r.weight = $w",
            {"f": from_id, "t": to_id, "w": new_weight},
        )

    def update_ray_status(self, from_id: str, to_id: str, status: str) -> None:
        self._execute(
            "MATCH (a:CellRef {id: $f})-[r:RAY]->(b:CellRef {id: $t}) "
            "SET r.status = $s",
            {"f": from_id, "t": to_id, "s": status},
        )

    def activate_ray(self, from_id: str, to_id: str) -> None:
        """标记射线被激活 (更新 last_activated)。"""
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            "MATCH (a:CellRef {id: $f})-[r:RAY]->(b:CellRef {id: $t}) "
            "SET r.last_activated = $now",
            {"f": from_id, "t": to_id, "now": now},
        )

    def get_outgoing_rays(self, cell_id: str) -> List[dict]:
        """该 cell 指向的所有 ray (该 cell 引用了谁)。"""
        return self._query_all(
            "MATCH (a:CellRef {id: $id})-[r:RAY]->(b:CellRef) "
            "RETURN a.id AS from_id, b.id AS to_id, r.weight AS weight, "
            "r.status AS status, r.created_at AS created_at, "
            "r.last_activated AS last_activated",
            {"id": cell_id},
        )

    def get_incoming_rays(self, cell_id: str) -> List[dict]:
        """指向该 cell 的所有 ray (谁引用了该 cell)。"""
        return self._query_all(
            "MATCH (a:CellRef)-[r:RAY]->(b:CellRef {id: $id}) "
            "RETURN a.id AS from_id, b.id AS to_id, r.weight AS weight, "
            "r.status AS status, r.created_at AS created_at, "
            "r.last_activated AS last_activated",
            {"id": cell_id},
        )

    def get_in_degree(self, cell_id: str, status: str = "active") -> int:
        """入度 (被引用次数,排除 severed)。"""
        rows = self._query_all(
            "MATCH (a:CellRef)-[r:RAY]->(b:CellRef {id: $id}) "
            "WHERE r.status = $s RETURN count(r) AS cnt",
            {"id": cell_id, "s": status},
        )
        return int(rows[0]["cnt"]) if rows else 0

    # ------------------------------------------------------------------
    # SUPERSEDES 操作 (new → old)
    # ------------------------------------------------------------------
    def add_supersedes(self, new_id: str, old_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            "MATCH (a:CellRef {id: $new}), (b:CellRef {id: $old}) "
            "CREATE (a)-[:SUPERSEDES {created_at: $now}]->(b)",
            {"new": new_id, "old": old_id, "now": now},
        )

    def get_supersede_chain(self, cell_id: str) -> List[str]:
        """沿 SUPERSEDES 方向 (新→旧) 用 BFS 收集完整版本链 (按距离排序)。"""
        chain = [cell_id]
        visited = {cell_id}
        queue = [cell_id]
        while queue:
            current = queue.pop(0)
            rows = self._query_all(
                "MATCH (a:CellRef {id: $id})-[:SUPERSEDES]->(b:CellRef) "
                "RETURN b.id AS id",
                {"id": current},
            )
            for r in rows:
                if r["id"] not in visited:
                    visited.add(r["id"])
                    chain.append(r["id"])
                    queue.append(r["id"])
        return chain

    # ------------------------------------------------------------------
    # LIGNIFIED_FROM 操作 (merged → source)
    # ------------------------------------------------------------------
    def add_lignified_from(self, merged_id: str, source_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            "MATCH (a:CellRef {id: $m}), (b:CellRef {id: $s}) "
            "CREATE (a)-[:LIGNIFIED_FROM {created_at: $now}]->(b)",
            {"m": merged_id, "s": source_id, "now": now},
        )

    # ------------------------------------------------------------------
    # 图分析查询
    # ------------------------------------------------------------------
    def find_orphans(self) -> List[str]:
        """找到没有任何 RAY 连接的孤立 cell。"""
        rows = self._query_all(
            "MATCH (c:CellRef) WHERE NOT EXISTS { MATCH (c)-[:RAY]-() } "
            "RETURN c.id AS id"
        )
        return [r["id"] for r in rows]

    def find_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """BFS 找两个 cell 之间的最短路径 (沿出射方向)。"""
        if from_id == to_id:
            return [from_id]
        visited = {from_id}
        queue: List[List[str]] = [[from_id]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for ray in self.get_outgoing_rays(node):
                nxt = ray["to_id"]
                if nxt == to_id:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    def get_connected_component(self, cell_id: str) -> List[str]:
        """获取与 cell 连通的所有节点 (无向,沿 RAY 边)。

        使用单条 Cypher 变长路径查询替代 N 次 BFS 逐跳查询。
        """
        rows = self._query_all(
            "MATCH (c:CellRef {id: $id})-[:RAY*1..]-(n:CellRef) "
            "RETURN DISTINCT n.id AS id",
            {"id": cell_id},
        )
        result = {cell_id}
        result.update(r["id"] for r in rows)
        return list(result)

    def query_by_domain_in_graph(self, seed_ids: List[str]) -> List[str]:
        """从 seed_ids (同 domain 的 cell) 扩展图邻居。"""
        result = set(seed_ids)
        for sid in seed_ids:
            result.update(self.get_connected_component(sid))
        return list(result)
