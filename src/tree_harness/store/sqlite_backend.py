"""SQLiteBackend —— cell 持久化存储、内容检索 (全文 + 向量)、CRUD。

使用单个 SQLite 文件 + sqlite-vec 扩展做向量检索。
对应 spec: docs/specs/sqlite_backend.md
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from typing import Optional, List, Tuple

import sqlite_vec

from tree_harness.core.cell_model import (
    Cell, Precondition, VerifyHint, CellStatus, RingLevel,
)
from tree_harness.core.embedding import Embedder, embed_cell_text

# update_cell 允许更新的字段 (行为契约 2/3: 不可改 decision/rationale/context)
_UPDATABLE_FIELDS = frozenset({"ring", "maturity", "energy", "status", "superseded_by"})

_CELLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS cells (
    id TEXT PRIMARY KEY,
    ring TEXT NOT NULL CHECK(ring IN ('L0','L1','L2','L3','L4')),
    maturity REAL NOT NULL DEFAULT 0.0,
    energy REAL NOT NULL DEFAULT 0.5,

    context_trigger_task TEXT,
    context_domain TEXT,
    context_preconditions_json TEXT,
    decision_text TEXT NOT NULL,
    rationale_text TEXT NOT NULL,

    evidence_json TEXT,
    domain_tags_json TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active','quarantined','superseded','archived')),
    source TEXT NOT NULL DEFAULT 'distilled'
        CHECK(source IN ('distilled','user_directive','seed')),

    created_at TEXT NOT NULL,
    superseded_by TEXT,

    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_cells_ring ON cells(ring);
CREATE INDEX IF NOT EXISTS idx_cells_status ON cells(status);
CREATE INDEX IF NOT EXISTS idx_cells_energy ON cells(energy);
CREATE INDEX IF NOT EXISTS idx_cells_domain ON cells(context_domain);
"""


class SQLiteBackend:
    """cell 档案存储 + 向量检索。"""

    def __init__(self, db_path: str = ":memory:", embedder: Optional[Embedder] = None,
                 conn: Optional[sqlite3.Connection] = None):
        self._embedder = embedder
        self._owns_conn = conn is None
        self._conn = conn if conn is not None else self._connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._dim = embedder.dim if embedder is not None else 64
        self.init_db()

    @property
    def embedder(self) -> Optional[Embedder]:
        """公开 embedder 访问 (CambiumEngine / Connector 需要预计算 embedding)。"""
        return self._embedder

    # ------------------------------------------------------------------
    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def init_db(self) -> None:
        """初始化数据库,创建表结构。"""
        self._conn.executescript(_CELLS_SCHEMA)
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_cells USING vec0("
            f"cell_id TEXT PRIMARY KEY, embedding float[{self._dim}])"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 序列化辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _precondition_to_dict(p: Precondition) -> dict:
        d = {"kind": p.kind, "assertion": p.assertion}
        if p.verify_hint is not None:
            d["verify_hint"] = asdict(p.verify_hint)
        return d

    @staticmethod
    def _dict_to_precondition(d: dict) -> Precondition:
        vh = None
        if d.get("verify_hint"):
            vh = VerifyHint(type=d["verify_hint"]["type"], params=d["verify_hint"]["params"])
        return Precondition(kind=d["kind"], assertion=d["assertion"], verify_hint=vh)

    def _cell_to_row(self, cell: Cell) -> dict:
        embedding = None
        if self._embedder is not None:
            vec = self._embedder.embed(embed_cell_text(cell.decision, cell.rationale))
            embedding = sqlite_vec.serialize_float32(vec)
        return {
            "id": cell.id,
            "ring": cell.ring,
            "maturity": cell.maturity,
            "energy": cell.energy,
            "context_trigger_task": cell.context_trigger_task,
            "context_domain": cell.context_domain,
            "context_preconditions_json": json.dumps(
                [self._precondition_to_dict(p) for p in cell.context_preconditions],
                ensure_ascii=False,
            ),
            "decision_text": cell.decision,
            "rationale_text": cell.rationale,
            "evidence_json": json.dumps(cell.evidence, ensure_ascii=False),
            "domain_tags_json": json.dumps(cell.domain_tags, ensure_ascii=False),
            "status": cell.status,
            "source": cell.source,
            "created_at": cell.created_at.isoformat(),
            "superseded_by": cell.superseded_by,
            "embedding": embedding,
        }

    def _row_to_cell(self, row: sqlite3.Row) -> Cell:
        preconds = []
        if row["context_preconditions_json"]:
            for d in json.loads(row["context_preconditions_json"]):
                preconds.append(self._dict_to_precondition(d))
        return Cell(
            id=row["id"],
            ring=row["ring"],
            maturity=row["maturity"],
            energy=row["energy"],
            context_trigger_task=row["context_trigger_task"] or "",
            context_domain=row["context_domain"] or "",
            context_preconditions=preconds,
            decision=row["decision_text"],
            rationale=row["rationale_text"],
            evidence=json.loads(row["evidence_json"]) if row["evidence_json"] else [],
            domain_tags=json.loads(row["domain_tags_json"]) if row["domain_tags_json"] else [],
            status=row["status"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            superseded_by=row["superseded_by"],
        )

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------
    def insert_cell(self, cell: Cell) -> None:
        """插入新 cell,同时存入 embedding。重复 ID 抛 IntegrityError。"""
        row = self._cell_to_row(cell)
        self._conn.execute(
            """INSERT INTO cells (id, ring, maturity, energy,
                context_trigger_task, context_domain, context_preconditions_json,
                decision_text, rationale_text, evidence_json, domain_tags_json,
                status, source, created_at, superseded_by, embedding)
               VALUES (:id, :ring, :maturity, :energy,
                :context_trigger_task, :context_domain, :context_preconditions_json,
                :decision_text, :rationale_text, :evidence_json, :domain_tags_json,
                :status, :source, :created_at, :superseded_by, :embedding)""",
            row,
        )
        if row["embedding"] is not None:
            self._conn.execute(
                "INSERT INTO vec_cells(cell_id, embedding) VALUES (?, ?)",
                (cell.id, row["embedding"]),
            )
        self._conn.commit()

    def update_cell(self, cell_id: str, **fields) -> None:
        """更新指定字段。只允许 ring/maturity/energy/status/superseded_by。"""
        bad = set(fields) - _UPDATABLE_FIELDS
        if bad:
            raise ValueError(
                f"不允许更新字段: {bad}; 仅允许 {_UPDATABLE_FIELDS} (公理六)"
            )
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [cell_id]
        self._conn.execute(f"UPDATE cells SET {assignments} WHERE id = ?", params)
        self._conn.commit()

    def delete_cell(self, cell_id: str) -> None:
        """彻底删除 cell (含向量),仅在清理时使用。"""
        self._conn.execute("DELETE FROM cells WHERE id = ?", (cell_id,))
        self._conn.execute("DELETE FROM vec_cells WHERE cell_id = ?", (cell_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------
    def get_cell(self, cell_id: str) -> Optional[Cell]:
        row = self._conn.execute("SELECT * FROM cells WHERE id = ?", (cell_id,)).fetchone()
        return self._row_to_cell(row) if row else None

    def query_by_ring(self, ring: str, status: str = "active") -> List[Cell]:
        rows = self._conn.execute(
            "SELECT * FROM cells WHERE ring = ? AND status = ? ORDER BY created_at",
            (ring, status),
        ).fetchall()
        return [self._row_to_cell(r) for r in rows]

    def list_by_ring(self, rings: List[str], status: str = "active") -> List[Cell]:
        """返回指定 ring 列表中指定 status 的所有 cell (before_step pinned 段用)。"""
        if not rings:
            return []
        placeholders = ",".join("?" * len(rings))
        rows = self._conn.execute(
            f"SELECT * FROM cells WHERE ring IN ({placeholders}) AND status = ? "
            "ORDER BY created_at",
            (*rings, status),
        ).fetchall()
        return [self._row_to_cell(r) for r in rows]

    def query_by_domain(self, domain: str, status: str = "active") -> List[Cell]:
        rows = self._conn.execute(
            "SELECT * FROM cells WHERE context_domain = ? AND status = ? ORDER BY created_at",
            (domain, status),
        ).fetchall()
        return [self._row_to_cell(r) for r in rows]

    def query_decay_candidates(
        self, energy_threshold: float, ring_idle_map: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> List[Cell]:
        """查询腐朽候选: energy < threshold 的 active cell。

        limit 用于 OuterHarness after_step 的抽样验证 (funnel_sample_size)。
        (长期未被引用的 idle 判定依赖 ray 拓扑,由 TreeStore/KuzuBackend 侧补充。)
        """
        sql = "SELECT * FROM cells WHERE status = 'active' AND energy < ? ORDER BY energy"
        params: list = [energy_threshold]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_cell(r) for r in rows]

    def list_active(self) -> List[Cell]:
        rows = self._conn.execute(
            "SELECT * FROM cells WHERE status = 'active' ORDER BY created_at"
        ).fetchall()
        return [self._row_to_cell(r) for r in rows]

    def list_by_status(self, status: str) -> List[Cell]:
        """返回指定 status 的所有 cell。"""
        rows = self._conn.execute(
            "SELECT * FROM cells WHERE status = ? ORDER BY created_at", (status,)
        ).fetchall()
        return [self._row_to_cell(r) for r in rows]

    def list_all(self) -> List[Cell]:
        """返回所有 cell (不限 status)。"""
        rows = self._conn.execute(
            "SELECT * FROM cells ORDER BY created_at"
        ).fetchall()
        return [self._row_to_cell(r) for r in rows]

    def vec_search(
        self, query_embedding: List[float], top_k: int = 10, threshold: float = 0.5
    ) -> List[Tuple[Cell, float]]:
        """向量相似度检索,返回 (cell, similarity) 列表,按 similarity 降序。

        只返回 status="active" 的 cell。similarity = 1/(1+L2 distance)。
        """
        q = sqlite_vec.serialize_float32(query_embedding)
        rows = self._conn.execute(
            "SELECT cell_id, distance FROM vec_cells "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            [q, top_k],
        ).fetchall()
        result: List[Tuple[Cell, float]] = []
        for row in rows:
            cell = self.get_cell(row["cell_id"])
            if cell is not None and cell.status == "active":
                similarity = 1.0 / (1.0 + row["distance"])
                if similarity >= threshold:
                    result.append((cell, similarity))
        return result

    def count_cells(self, ring: Optional[str] = None, status: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) AS c FROM cells WHERE 1=1"
        params: list = []
        if ring is not None:
            sql += " AND ring = ?"
            params.append(ring)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        return int(self._conn.execute(sql, params).fetchone()["c"])

    def oldest_active_in_ring(self, ring: str, by: str = "maturity") -> Optional[Cell]:
        """返回指定 ring 中最老 (by=maturity) 或最早创建 (by=created_at) 的 active cell。"""
        if by == "maturity":
            order = "maturity ASC, created_at ASC"
        elif by == "created_at":
            order = "created_at ASC"
        else:
            order = "created_at ASC"
        row = self._conn.execute(
            f"SELECT * FROM cells WHERE ring = ? AND status = 'active' ORDER BY {order} LIMIT 1",
            (ring,),
        ).fetchone()
        return self._row_to_cell(row) if row else None

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()
