"""OpLog —— append-only 操作日志。

记录所有对 tree 的写操作,用于版本控制、状态重建、实验分析。
对应 spec: docs/specs/oplog.md
"""
from __future__ import annotations

import enum
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# 操作类型枚举 (底层 op_type,粒度细便于回放/调试)
# 语义层聚合到 5 算符见 OP_TO_OPERATOR。
# ---------------------------------------------------------------------------
class OpEnum(str, enum.Enum):
    """底层操作类型。枚举仅扩不删——新增 op 必须同步更新 OP_TO_OPERATOR。"""
    # CRYSTALLIZE
    INSERT_CELL = "INSERT_CELL"
    # CONNECT (含权重变化、激活、反向特例 REINFORCE)
    INSERT_RAY = "INSERT_RAY"
    UPDATE_RAY_WEIGHT = "UPDATE_RAY_WEIGHT"
    SEVER_RAY = "SEVER_RAY"
    REINFORCE = "REINFORCE"
    RAY_ACTIVATED = "RAY_ACTIVATED"
    # PROMOTE (含复合算符 MERGE / SPLIT、降层 DEMOTE)
    PROMOTE = "PROMOTE"
    DEMOTE = "DEMOTE"
    MERGE = "MERGE"
    SPLIT = "SPLIT"
    # QUARANTINE (含版本替代、review 标记、容量淘汰)
    QUARANTINE = "QUARANTINE"
    SUPERSEDE = "SUPERSEDE"
    MARK_REVIEW = "MARK_REVIEW"
    ARCHIVE = "ARCHIVE"
    # DECAY
    UPDATE_ENERGY = "UPDATE_ENERGY"
    UPDATE_MATURITY = "UPDATE_MATURITY"


# ---------------------------------------------------------------------------
# 算符映射 (I-Op2: 每个 op 必须归属一个算符)
#
# 设计决策: OpLog 走 "底层 op_type + 单射算符映射表",不在存储层引入语义级
# op_type。理由见 docs/specs/oplog.md "OP_TO_OPERATOR 映射" 节。
# ---------------------------------------------------------------------------
OP_TO_OPERATOR: dict[str, str] = {
    # CRYSTALLIZE
    OpEnum.INSERT_CELL.value: "CRYSTALLIZE",
    # CONNECT
    OpEnum.INSERT_RAY.value: "CONNECT",
    OpEnum.UPDATE_RAY_WEIGHT.value: "CONNECT",
    OpEnum.SEVER_RAY.value: "CONNECT",
    OpEnum.REINFORCE.value: "CONNECT",
    OpEnum.RAY_ACTIVATED.value: "CONNECT",
    # PROMOTE
    OpEnum.PROMOTE.value: "PROMOTE",
    OpEnum.DEMOTE.value: "PROMOTE",
    OpEnum.MERGE.value: "PROMOTE",
    OpEnum.SPLIT.value: "PROMOTE",
    # QUARANTINE
    OpEnum.QUARANTINE.value: "QUARANTINE",
    OpEnum.SUPERSEDE.value: "QUARANTINE",
    OpEnum.MARK_REVIEW.value: "QUARANTINE",
    OpEnum.ARCHIVE.value: "QUARANTINE",
    # DECAY
    OpEnum.UPDATE_ENERGY.value: "DECAY",
    OpEnum.UPDATE_MATURITY.value: "DECAY",
}

# I-Op2 强制校验: 枚举里每个 op 必须在映射表里,新增 op 忘了归类直接 import 失败
assert set(OP_TO_OPERATOR.keys()) == {e.value for e in OpEnum}, \
    "I-Op2 violated: op_type without operator binding"

# 5 算符常量 (count_by_op_type 保证返回这 5 个 key)
_OPERATORS = ("CRYSTALLIZE", "CONNECT", "PROMOTE", "QUARANTINE", "DECAY")

# 向后兼容别名 (str mixin 使得 OpEnum.INSERT_CELL == "INSERT_CELL" 恒为 True)
OpType = OpEnum


@dataclass
class OpLogEntry:
    """单条操作日志。"""
    seq: int
    timestamp: str            # ISO 8601
    op: str                   # 底层 op_type (OpEnum 之一)
    operator: str             # 派生算符: CRYSTALLIZE/CONNECT/PROMOTE/QUARANTINE/DECAY
    payload: dict
    episode_id: Optional[str] = None


# ---------------------------------------------------------------------------
# replay 目标协议: TreeStore 实现该方法以支持回放重建
# ---------------------------------------------------------------------------
@runtime_checkable
class ReplayTarget(Protocol):
    def apply_op(self, op: str, payload: dict, episode_id: Optional[str]) -> None:
        """应用单条 op (不写 op log,用于 replay)。"""
        ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS oplog (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    op TEXT NOT NULL,                -- 底层 op_type (OpEnum 之一)
    operator TEXT NOT NULL,         -- 派生算符: CRYSTALLIZE/CONNECT/PROMOTE/QUARANTINE/DECAY
    payload_json TEXT NOT NULL,
    episode_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_oplog_op       ON oplog(op);
CREATE INDEX IF NOT EXISTS idx_oplog_operator ON oplog(operator);
CREATE INDEX IF NOT EXISTS idx_oplog_episode  ON oplog(episode_id);

-- cell_id -> seq 映射表 (一条 op 可关联多个 cell_id)
CREATE TABLE IF NOT EXISTS oplog_cell_map (
    seq INTEGER NOT NULL,
    cell_id TEXT NOT NULL,
    PRIMARY KEY (seq, cell_id)
);
CREATE INDEX IF NOT EXISTS idx_map_cell ON oplog_cell_map(cell_id);
"""

# payload 中可能包含 cell_id 的字段名
_CELL_ID_FIELDS = (
    "cell_id", "old_id", "new_id", "source_id", "target_id",
    "from_id", "to_id",
)
_CELL_ID_LIST_FIELDS = ("source_ids", "target_ids")


def _extract_cell_ids(payload: dict) -> list[str]:
    """从 payload 中提取所有 cell_id (用于索引映射)。"""
    ids: set[str] = set()
    for field_name in _CELL_ID_FIELDS:
        val = payload.get(field_name)
        if isinstance(val, str) and val:
            ids.add(val)
    for field_name in _CELL_ID_LIST_FIELDS:
        vals = payload.get(field_name)
        if isinstance(vals, list):
            ids.update(v for v in vals if isinstance(v, str) and v)
    return list(ids)


class OpLog:
    """append-only 操作日志,SQLite 存储。"""

    def __init__(self, db_path: str = ":memory:", conn: Optional[sqlite3.Connection] = None):
        self._owns_conn = conn is None
        self._conn = conn if conn is not None else sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self.init_db()

    def init_db(self) -> None:
        """初始化数据库,创建表结构。"""
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def clear(self) -> None:
        """清空所有 oplog 记录 (Runner reset 用)。"""
        self._conn.execute("DELETE FROM oplog_cell_map")
        self._conn.execute("DELETE FROM oplog")
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def append(self, op: str, payload: dict, episode_id: Optional[str] = None) -> int:
        """追加一条日志,返回 seq。OpLog 只追加,永不修改已有条目。

        operator 字段由 OP_TO_OPERATOR[op] 自动派生,调用方不允许传入。
        若 op 不在映射表中,抛 KeyError (I-Op2 运行时兜底)。
        """
        operator = OP_TO_OPERATOR[op]  # KeyError if not registered
        cur = self._conn.execute(
            "INSERT INTO oplog (timestamp, op, operator, payload_json, episode_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), op, operator,
             json.dumps(payload, ensure_ascii=False), episode_id),
        )
        seq = cur.lastrowid
        # 提取所有 cell_id 写入映射表 (索引加速 get_cell_history)
        for cid in _extract_cell_ids(payload):
            self._conn.execute(
                "INSERT OR IGNORE INTO oplog_cell_map (seq, cell_id) VALUES (?, ?)",
                (seq, cid),
            )
        self._conn.commit()
        return seq

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_latest_seq(self) -> int:
        """获取当前最新 seq,空库返回 0。"""
        row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) AS s FROM oplog").fetchone()
        return int(row["s"])

    def get_entries(
        self, since_seq: int = 0, op_filter: Optional[str] = None
    ) -> List[OpLogEntry]:
        """查询日志条目 (seq > since_seq)。"""
        if op_filter is not None:
            rows = self._conn.execute(
                "SELECT * FROM oplog WHERE seq > ? AND op = ? ORDER BY seq",
                (since_seq, op_filter),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM oplog WHERE seq > ? ORDER BY seq", (since_seq,)
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_cell_history(self, cell_id: str) -> List[OpLogEntry]:
        """获取某 cell 的完整操作历史 (通过 oplog_cell_map 索引加速)。"""
        rows = self._conn.execute(
            "SELECT o.* FROM oplog o "
            "JOIN oplog_cell_map m ON o.seq = m.seq "
            "WHERE m.cell_id = ? ORDER BY o.seq",
            (cell_id,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_episode_ops(self, episode_id: str) -> List[OpLogEntry]:
        """获取某 episode 产生的所有操作。"""
        rows = self._conn.execute(
            "SELECT * FROM oplog WHERE episode_id = ? ORDER BY seq", (episode_id,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # 聚合统计 (供 metrics & EpisodeReport 消费)
    # ------------------------------------------------------------------
    def count_by_op_type(self, episode_id: Optional[str] = None) -> dict[str, int]:
        """按 5 算符聚合 (不是底层 op)。

        返回 {"CRYSTALLIZE", "CONNECT", "PROMOTE", "QUARANTINE", "DECAY"} -> int。
        缺失的算符 key 用 0 填充,保证下游永远拿到 5 个 key。
        """
        result = {op: 0 for op in _OPERATORS}
        if episode_id is not None:
            rows = self._conn.execute(
                "SELECT operator, COUNT(*) AS cnt FROM oplog "
                "WHERE episode_id = ? GROUP BY operator",
                (episode_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT operator, COUNT(*) AS cnt FROM oplog GROUP BY operator"
            ).fetchall()
        for row in rows:
            result[row["operator"]] = int(row["cnt"])
        return result

    def count_by_raw_op(self, episode_id: Optional[str] = None) -> dict[str, int]:
        """按底层 op_type 统计 (不聚合到 5 算符)。

        用于区分 PROMOTE 算符下的真实 ring promotion vs MERGE/DEMOTE/SPLIT。
        返回所有 OpEnum 值 -> int, 缺失的用 0 填充。
        """
        result = {e.value: 0 for e in OpEnum}
        if episode_id is not None:
            rows = self._conn.execute(
                "SELECT op, COUNT(*) AS cnt FROM oplog "
                "WHERE episode_id = ? GROUP BY op",
                (episode_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT op, COUNT(*) AS cnt FROM oplog GROUP BY op"
            ).fetchall()
        for row in rows:
            result[row["op"]] = int(row["cnt"])
        return result

    def count_promotes_by_reason(self, episode_id: Optional[str] = None) -> dict[str, int]:
        """统计 PROMOTE / DEMOTE 中各 reason 的次数。

        返回 {"normal", "overflow_force", "overflow_demote"} -> int。
        """
        result = {"normal": 0, "overflow_force": 0, "overflow_demote": 0}
        sql = "SELECT payload_json FROM oplog WHERE op IN (?, ?)"
        params: list = [OpEnum.PROMOTE.value, OpEnum.DEMOTE.value]
        if episode_id is not None:
            sql += " AND episode_id = ?"
            params.append(episode_id)
        rows = self._conn.execute(sql, params).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            reason = payload.get("reason", "normal")
            if reason in result:
                result[reason] += 1
        return result

    # ------------------------------------------------------------------
    # 回放
    # ------------------------------------------------------------------
    def replay(
        self,
        tree_store: ReplayTarget,
        from_seq: int = 0,
        to_seq: Optional[int] = None,
    ) -> None:
        """从 op log 重建 tree 状态: 逐条调用 tree_store.apply_op。"""
        entries = self.get_entries(since_seq=from_seq)
        for entry in entries:
            if to_seq is not None and entry.seq > to_seq:
                break
            tree_store.apply_op(entry.op, entry.payload, entry.episode_id)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> OpLogEntry:
        return OpLogEntry(
            seq=int(row["seq"]),
            timestamp=row["timestamp"],
            op=row["op"],
            operator=row["operator"],
            payload=json.loads(row["payload_json"]),
            episode_id=row["episode_id"],
        )
