# KuzuDB Backend Spec

## 概述

KuzuBackend 是 Self-Evolution Operator Set 中 R 集合（ray 拓扑）与版本/木质化关系的图存储后端——`connect` 算符的 RAY 落盘、`quarantine` 算符触发的 SEVER_RAY 状态变更、`promote` 算符的 LIGNIFIED_FROM 链路全部由 KuzuBackend 持久化。它存储 cell 的引用关系（RAY）、版本关系（SUPERSEDES）、木质化来源（LIGNIFIED_FROM）。

定位说明：KuzuBackend 不感知任何算符语义，只暴露图原语（add_ray / update_ray_status / get_in_degree / find_path 等）；所有算符调用必须先经 `TreeStore` 外观，KuzuBackend 不允许被算法服务（Cambium / Decay Sentinel / Lignification）直接访问——这条对齐 outer_harness.md I-Op1 与 tree_store.md 的双库一致性约束。

## Schema

```cypher
CREATE NODE TABLE CellRef (
    id STRING,
    ring STRING,
    PRIMARY KEY (id)
)

CREATE REL TABLE RAY (
    FROM CellRef TO CellRef,
    weight DOUBLE,
    status STRING,       -- active | weakened | severed
    created_at STRING,
    last_activated STRING
)

CREATE REL TABLE SUPERSEDES (
    FROM CellRef TO CellRef,
    created_at STRING
)

CREATE REL TABLE LIGNIFIED_FROM (
    FROM CellRef TO CellRef,
    created_at STRING
)
```

## 接口定义

```python
class KuzuBackendProtocol(Protocol):
    def init_db(self, db_path: str) -> None:
        """初始化 KuzuDB，创建 schema"""
        ...
    
    # --- Node 操作 ---
    def add_cell_ref(self, cell_id: str, ring: str) -> None:
        """添加 cell 节点引用"""
        ...
    
    def update_cell_ring(self, cell_id: str, new_ring: str) -> None:
        """更新节点的 ring 字段"""
        ...
    
    def remove_cell_ref(self, cell_id: str) -> None:
        """移除节点（仅在 cell 被彻底删除时）"""
        ...
    
    # --- RAY 操作 ---
    def add_ray(self, from_id: str, to_id: str, weight: float) -> None:
        """添加射线（方向：from=外层 → to=内层）"""
        ...
    
    def update_ray_weight(self, from_id: str, to_id: str, new_weight: float) -> None:
        """更新射线权重"""
        ...
    
    def update_ray_status(self, from_id: str, to_id: str, status: str) -> None:
        """更新射线状态 (active/weakened/severed)"""
        ...
    
    def activate_ray(self, from_id: str, to_id: str) -> None:
        """标记射线被激活（更新 last_activated）"""
        ...
    
    def get_outgoing_rays(self, cell_id: str) -> list[dict]:
        """获取该 cell 指向的所有 ray（该 cell 引用了谁）"""
        ...
    
    def get_incoming_rays(self, cell_id: str) -> list[dict]:
        """获取指向该 cell 的所有 ray（谁引用了该 cell）"""
        ...
    
    def get_in_degree(self, cell_id: str, status: str = "active") -> int:
        """获取入度（被引用次数）"""
        ...
    
    # --- SUPERSEDES 操作 ---
    def add_supersedes(self, new_id: str, old_id: str) -> None:
        """标记 new supersedes old"""
        ...
    
    def get_supersede_chain(self, cell_id: str) -> list[str]:
        """获取完整的版本链"""
        ...
    
    # --- LIGNIFIED_FROM 操作 ---
    def add_lignified_from(self, merged_id: str, source_id: str) -> None:
        """标记合并来源"""
        ...
    
    # --- 图分析查询 ---
    def find_orphans(self) -> list[str]:
        """找到没有任何 ray 连接的孤立 cell"""
        ...
    
    def find_path(self, from_id: str, to_id: str) -> Optional[list[str]]:
        """找两个 cell 之间的最短路径"""
        ...
    
    def get_connected_component(self, cell_id: str) -> list[str]:
        """获取与 cell 连通的所有节点"""
        ...
    
    def query_by_domain_in_graph(self, domain_tags: list[str]) -> list[str]:
        """通过图遍历找同 domain 的 cell（从 SQLite 侧查 domain 后在图侧扩展邻居）"""
        ...
```

## 行为契约

1. `add_ray` 前必须确认 from_id 和 to_id 都存在于 CellRef 表
2. RAY 的方向语义：from = 外层（引用方），to = 内层（被引用方）
3. `add_ray` 中 from 的 ring 层级应 <= to 的 ring 层级（外→内原则），否则 log warning 但不阻止
4. `update_ray_weight` 中 weight 被 clip 到 [0, 1]
5. severed 状态的 ray 不删除，保留历史
6. `add_cell_ref` 只存 id 和 ring，完整 cell 数据在 SQLite 侧

## 与 SQLite 的协调

- 写入顺序：先 SQLite（存完整 cell）→ 后 KuzuDB（存拓扑引用）
- KuzuDB 中 CellRef.id 必须在 SQLite cells.id 中存在
- 一致性检查：每 100 episode 扫描 KuzuDB 所有 id 是否在 SQLite 中存在

## 测试用例

1. 初始化 DB → schema 正确创建
2. add_cell_ref + add_ray → get_outgoing_rays 正确返回
3. add_ray → get_incoming_rays 从被引用方能查到
4. get_in_degree 正确计数（排除 severed ray）
5. update_ray_weight 超过 1.0 → clip 到 1.0
6. add_supersedes → get_supersede_chain 返回完整链
7. find_orphans 正确识别无连接节点
8. 添加 3 个 cell 形成链 A→B→C → find_path(A,C) 返回 [A,B,C]
