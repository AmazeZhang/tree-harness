# TreeStore Spec

## 概述

TreeStore 是 Self-Evolution Operator Set 五算符的**统一持久化外观**，协调 SQLiteBackend（cell 档案）与 KuzuBackend（ray 拓扑）的读写。算符的所有副作用（向 C 插入元素、向 R 插入元素、修改 ρ / E / cell.status）都必须经 TreeStore 落盘，并由 OuterHarness 在调用前/后写 OpLog——这条规则是 outer_harness.md 不变量 I-Op1 的物理实现。

定位说明：算法服务（Cambium / EnergySystem / Decay Sentinel / Lignification）通过 TreeStore 读写，**不允许绕过 TreeStore 直接访问底层后端**（不允许 `tree_store.sqlite.xxx` 或 `tree_store.kuzu.xxx` 形式的直接调用）。TreeStore 自身不实现任何业务逻辑、不感知算符语义，只做双库一致性保障与读写路由。

## 设计原则

- TreeStore 不实现业务逻辑（能量计算、蒸馏等），只做双库协调和一致性保障
- 写入顺序固定：先 SQLite → 后 KuzuDB
- 所有写操作同步写 OpLog（先 OpLog → 再实际写入）
- 读操作根据需要路由到合适的后端

## 接口定义

```python
from typing import Protocol, Optional

class TreeStoreProtocol(Protocol):
    def __init__(self, sqlite: SQLiteBackend, kuzu: KuzuBackend, oplog: OpLog):
        ...

    # --- Cell 生命周期 ---
    def insert_cell(self, cell: Cell, rays: list[tuple[str, float]] = None) -> None:
        """
        插入新 cell + 可选 ray。
        执行顺序：
        1. oplog.append(INSERT_CELL)
        2. sqlite.insert_cell(cell)
        3. kuzu.add_cell_ref(cell.id, cell.ring)
        4. for target, weight in rays:
               oplog.append(INSERT_RAY)
               kuzu.add_ray(cell.id, target, weight)
        """
        ...

    def get_cell(self, cell_id: str) -> Optional[Cell]:
        """从 SQLite 获取完整 cell"""
        ...

    def get_cells_batch(self, cell_ids: list[str]) -> list[Cell]:
        """批量获取 cell"""
        ...

    # --- 字段更新 ---
    def update_energy(self, cell_id: str, new_energy: float, reason: str, episode_id: str) -> None:
        """
        更新 energy。
        1. oplog.append(UPDATE_ENERGY, {old, new, reason})
        2. sqlite.update_cell(cell_id, energy=new_energy)
        """
        ...

    def update_maturity(self, cell_id: str, new_maturity: float, episode_id: str) -> None:
        """更新 maturity"""
        ...

    def mark_for_review(self, cell_id: str, flag: bool, episode_id: str, reason: str = "uncertain_verdict") -> None:
        """
        切换 cell.needs_review 标志位。
        1. oplog.append(MARK_REVIEW, {cell_id, flag, reason})
        2. sqlite.update_cell(cell_id, needs_review=flag)

        由 Decay Sentinel 在 Verdict=`uncertain` 时调用置 True；funnel verification
        在下次抽样时优先选中 needs_review=True 的 cell。该字段不会被 quarantine 算符自动清零。
        """
        ...

    def promote(
        self,
        cell_id: str,
        from_ring: str,
        to_ring: str,
        episode_id: str,
        reason: str = "normal",
    ) -> None:
        """
        升层（原地改字段）。
        1. oplog.append(PROMOTE, {cell_id, from_ring, to_ring, reason})
        2. sqlite.update_cell(cell_id, ring=to_ring)
        3. kuzu.update_cell_ring(cell_id, to_ring)
        reason ∈ {"normal", "overflow_force", "overflow_demote"}（容量策略见 ring_promotion.md）。
        """
        ...

    def demote(
        self,
        cell_id: str,
        from_ring: str,
        to_ring: str,
        episode_id: str,
        reason: str = "normal",
    ) -> None:
        """降层；reason 语义与 promote 对齐。"""
        ...

    # --- 合并/分裂 ---
    def merge_cells(self, source_ids: list[str], merged_cell: Cell, rays: list[tuple[str, float]], episode_id: str) -> None:
        """
        合并操作（木质化）。
        1. oplog.append(MERGE, {source_ids, target_id})
        2. 插入 merged_cell (SQLite + KuzuDB)
        3. 对每个 source: quarantine + SUPERSEDES + LIGNIFIED_FROM
        4. 重定向 incoming rays → merged_cell
        5. 建立新 ray 连接
        """
        ...

    def split_cell(self, source_id: str, child_cells: list[Cell], rays_map: dict[str, list[tuple[str, float]]], episode_id: str) -> None:
        """
        分裂操作。
        1. oplog.append(SPLIT, {source_id, target_ids})
        2. quarantine source
        3. 插入所有 child_cells
        4. SUPERSEDES source → children
        5. 建立 ray 连接
        """
        ...

    # --- 隔离 ---
    def quarantine(self, cell_id: str, reason: str, episode_id: str) -> None:
        """
        隔离 cell（腐朽裁定后）。
        1. oplog.append(QUARANTINE)
        2. sqlite.update_cell(cell_id, status='quarantined')
        3. 切断所有外向 active ray → severed
        """
        ...

    # --- Ray 操作 ---
    def add_ray(self, from_id: str, to_id: str, weight: float, episode_id: str) -> None:
        """新建射线"""
        ...

    def update_ray_weight(self, from_id: str, to_id: str, new_weight: float, episode_id: str) -> None:
        """更新射线权重"""
        ...

    def activate_ray(self, from_id: str, to_id: str, episode_id: str) -> None:
        """标记射线被激活：更新 last_activated + oplog.append(RAY_ACTIVATED)。
        Ray 激活计数是 control lag 指标（ray 一跳传播延迟）的来源之一。"""
        ...

    def sever_ray(self, from_id: str, to_id: str, reason: str, episode_id: str) -> None:
        """切断射线"""
        ...

    # --- 查询 ---
    def vec_search(self, query_embedding: list[float], top_k: int = 10, min_score: float = 0.5) -> list[tuple[Cell, float]]:
        """向量检索（路由到 SQLite）"""
        ...

    def list_by_ring(
        self,
        rings: list[str],
        status: str = "active",
    ) -> list[Cell]:
        """按 ring 列表 + status 过滤批量取 cell。

        Pinned 段（outer_harness before_step）调用此方法拉取所有 L3/L4 active cell。
        rings 为单元素 list 时退化为单 ring 查询；status 取 active / quarantined / superseded / archived 之一。
        """
        ...

    def list_active_cells(self) -> list[Cell]:
        """返回所有 status=='active' 的 cell（跨 ring）。

        EnergySystem.decay_all / update_all_maturity 在 after_episode 内调用，
        遍历所有活跃 cell 执行能量/成熟度更新。
        """
        ...

    def count_active_by_ring(self, ring: str) -> int:
        """统计指定 ring 中 active 状态的 cell 数量（容量执行用，见 lignification.md）。"""
        ...

    def oldest_active_in_ring(self, ring: str, by: str = "maturity") -> Optional[Cell]:
        """按指定字段升序返回 ring 中最旧/最弱的 active cell。

        by ∈ {"maturity", "last_referenced_episode", "created_at"}。
        容量溢出策略 demote_oldest 使用 by="last_referenced_episode"。
        无符合条件 cell 返回 None。
        """
        ...

    def get_incoming_rays(self, cell_id: str) -> list[dict]:
        """获取入射线（路由到 KuzuDB）"""
        ...

    def get_outgoing_rays(self, cell_id: str) -> list[dict]:
        """获取出射线"""
        ...

    def get_in_degree(self, cell_id: str) -> int:
        """获取入度"""
        ...

    def find_orphans(self) -> list[str]:
        """查找孤立节点"""
        ...

    # --- 一致性 ---
    def consistency_check(self) -> list[str]:
        """
        扫描双库一致性，返回不一致的 cell_id 列表。
        检查项：
        1. KuzuDB 中存在但 SQLite 中不存在的 id
        2. KuzuDB 中 ring 字段与 SQLite 不一致的 id
        3. SQLite 中 active 但 KuzuDB 中不存在的 id
        """
        ...

    # --- 统计 ---
    def stats(self) -> dict:
        """返回 tree 统计摘要"""
        ...
```

## 写入序列示例

### insert_cell 完整序列

```
1. validate(cell)                          # 字段合法性
2. oplog.append("INSERT_CELL", {...})      # 先写日志
3. sqlite.insert_cell(cell)                # 存档案
4. kuzu.add_cell_ref(cell.id, cell.ring)   # 存拓扑
5. for (target, weight) in rays:
     oplog.append("INSERT_RAY", {...})
     kuzu.add_ray(cell.id, target, weight)
```

### quarantine 完整序列

```
1. oplog.append("QUARANTINE", {cell_id, reason})
2. sqlite.update_cell(cell_id, status="quarantined")
3. outgoing = kuzu.get_outgoing_rays(cell_id)
4. for ray in outgoing:
     if ray.status == "active":
       oplog.append("SEVER_RAY", {from, to, reason="quarantine"})
       kuzu.update_ray_status(from, to, "severed")
```

### merge_cells 完整序列

```
1. oplog.append("MERGE", {source_ids, target_id=merged_cell.id})
2. sqlite.insert_cell(merged_cell)
3. kuzu.add_cell_ref(merged_cell.id, merged_cell.ring)
4. for source_id in source_ids:
     oplog.append("QUARANTINE", {source_id, reason="merged"})
     sqlite.update_cell(source_id, status="superseded", superseded_by=merged_cell.id)
     kuzu.add_supersedes(merged_cell.id, source_id)
     kuzu.add_lignified_from(merged_cell.id, source_id)
5. # 重定向：将指向 source 的 active incoming ray 复制到 merged_cell
   for source_id in source_ids:
     incoming = kuzu.get_incoming_rays(source_id)
     for ray in incoming:
       if ray.from_id not in source_ids:  # 不含内部互指
         kuzu.add_ray(ray.from_id, merged_cell.id, ray.weight * 0.8)
6. # 建立 merged_cell 的 outgoing ray
   for (target, weight) in rays:
     oplog.append("INSERT_RAY", {merged_cell.id, target, weight})
     kuzu.add_ray(merged_cell.id, target, weight)
```

## 错误处理

- SQLite 写入失败 → 整个操作回滚（利用 SQLite 事务），KuzuDB 不写
- KuzuDB 写入失败 → SQLite 已提交无法回滚 → 标记 inconsistency flag → 下次 consistency_check 修复
- OpLog 写入失败 → 操作不执行（op log 是 write-ahead log）

## stats() 返回结构

```python
{
    "total_cells": int,
    "by_ring": {"L0": int, "L1": int, ...},
    "by_status": {"active": int, "quarantined": int, ...},
    "total_rays": int,
    "active_rays": int,
    "oplog_seq": int,
}
```

## 测试用例

1. insert_cell → SQLite 和 KuzuDB 中都能查到
2. insert_cell 带 rays → ray 正确建立在 KuzuDB 中
3. quarantine → cell status 变更 + 外向 ray 全部 severed
4. merge_cells → source 变 superseded + merged_cell 存在 + incoming ray 重定向
5. promote → SQLite ring 字段和 KuzuDB ring 字段同步变更
6. oplog 记录完整（insert_cell 产生 INSERT_CELL + N 条 INSERT_RAY）
7. consistency_check 在正常状态下返回空列表
8. 手动删除 KuzuDB 中一个节点 → consistency_check 检出不一致
9. stats 返回正确的统计数字
