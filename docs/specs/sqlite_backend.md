# SQLite Backend Spec

## 概述

SQLiteBackend 是 Self-Evolution Operator Set 中 C 集合（cells）、E（energy）、ρ（ring 分派）以及 cell.status 的关系存储后端——`crystallize` 算符的新 cell 落盘、`decay` 算符的 energy / maturity 更新、`promote` 算符的 ring 字段更新、`quarantine` 算符的 status 翻转全部由 SQLiteBackend 持久化。它使用单个 SQLite 文件 + sqlite-vec 扩展，承担 cell 的持久化存储、内容检索（向量）、CRUD 操作。

定位说明：SQLiteBackend 不感知任何算符语义，只暴露表级原语（insert_cell / update_cell / vec_search / query_by_ring 等）；所有算符调用必须先经 `TreeStore` 外观，SQLiteBackend 不允许被算法服务（Cambium / EnergySystem / Decay Sentinel / Lignification）直接访问——这条对齐 outer_harness.md I-Op1 与 tree_store.md 的双库一致性约束。

## Schema

```sql
CREATE TABLE cells (
    id TEXT PRIMARY KEY,
    ring TEXT NOT NULL CHECK(ring IN ('L0','L1','L2','L3','L4')),
    maturity REAL NOT NULL DEFAULT 0.0,
    energy REAL NOT NULL DEFAULT 0.5,
    
    context_trigger_task TEXT,
    context_domain TEXT,
    context_preconditions_json TEXT,  -- JSON array of Precondition
    decision_text TEXT NOT NULL,
    rationale_text TEXT NOT NULL,
    
    evidence_json TEXT,       -- JSON array of strings
    domain_tags_json TEXT,    -- JSON array of strings
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','quarantined','superseded','archived')),
    source TEXT NOT NULL DEFAULT 'distilled' CHECK(source IN ('distilled','user_directive','seed')),
    needs_review INTEGER NOT NULL DEFAULT 0,  -- 0/1 bool; Decay Sentinel 在 Verdict=uncertain 时置 1
    
    created_at TEXT NOT NULL,  -- ISO 8601
    superseded_by TEXT,
    
    -- embedding for vec search
    embedding BLOB  -- float32 vector, dimension depends on model
);

CREATE INDEX idx_cells_ring ON cells(ring);
CREATE INDEX idx_cells_status ON cells(status);
CREATE INDEX idx_cells_energy ON cells(energy);
CREATE INDEX idx_cells_domain ON cells(context_domain);
CREATE INDEX idx_cells_needs_review ON cells(needs_review);
```

## 接口定义

```python
from typing import Protocol, Optional

class SQLiteBackendProtocol(Protocol):
    def init_db(self, db_path: str) -> None:
        """初始化数据库，创建表结构"""
        ...
    
    def insert_cell(self, cell: Cell) -> None:
        """插入新 cell，同时存入 embedding"""
        ...
    
    def get_cell(self, cell_id: str) -> Optional[Cell]:
        """按 ID 获取 cell"""
        ...
    
    def update_cell(self, cell_id: str, **fields) -> None:
        """更新指定字段（ring, maturity, energy, status, superseded_by）"""
        ...
    
    def query_by_ring(self, ring: str, status: str = "active") -> list[Cell]:
        """按 ring 层查询"""
        ...
    
    def query_by_domain(self, domain: str, status: str = "active") -> list[Cell]:
        """按 domain 查询"""
        ...
    
    def query_decay_candidates(self, energy_threshold: float, ring_idle_map: dict) -> list[Cell]:
        """查询腐朽候选：energy < threshold OR 长期未被引用"""
        ...
    
    def vec_search(self, query_embedding: list[float], top_k: int = 10, threshold: float = 0.5) -> list[tuple[Cell, float]]:
        """向量相似度检索，返回 (cell, similarity_score) 列表"""
        ...
    
    def count_cells(self, ring: Optional[str] = None, status: Optional[str] = None) -> int:
        """统计 cell 数量"""
        ...
```

## 行为契约

1. `insert_cell` 时必须同时计算并存入 embedding
2. `update_cell` 只允许更新以下字段：`ring`, `maturity`, `energy`, `status`, `superseded_by`, `needs_review`
3. `update_cell` 不允许修改 `decision_text`, `rationale_text`, `context_*`（公理六）
4. `vec_search` 返回结果按 similarity 降序排列
5. `vec_search` 只返回 status="active" 的 cell
6. 所有写操作在单个事务内完成

## Embedding 模型

初始选型：`text-embedding-3-small`（OpenAI）或本地 `all-MiniLM-L6-v2`

Embedding 输入：`cell.decision + " | " + cell.rationale`

维度：取决于模型（384 / 1536）

## 测试用例

1. 创建数据库 → 验证表结构存在
2. 插入 cell → get_cell 取回 → 字段一致
3. 插入重复 ID → 抛出异常
4. update_cell(energy=0.3) → 验证只有 energy 变了
5. update_cell(decision_text=...) → 抛出异常（不允许改内容）
6. 插入 5 个不同 domain 的 cell → query_by_domain 只返回匹配的
7. 插入 10 个 cell → vec_search 返回按相似度排序的 top-5
8. quarantined cell 不出现在 vec_search 结果中
9. query_decay_candidates 正确识别 energy < threshold 的 cell
