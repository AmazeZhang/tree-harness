# Phase 0+1 Code Review — 2026-06-24

**审查范围**：
- Phase 0：cell_model / oplog / sqlite_backend / kuzu_backend / tree_store
- Phase 1：energy_system / ring_promotion

**测试状态**：66 passed（含 100 episode 端到端模拟）。

**整体评价**：spec 还原度高，关键不变量（公理六不可变、半衰期梯度、滞回带、user_directive 免疫）都有运行时保护或断言覆盖。下面是发现的 bug 与改进建议，按优先级排列。

---

## P0 — 必须修复

### Bug 1: `merge_cells` 未处理 source 的外向 ray

**位置**：`src/tree_harness/store/tree_store.py: merge_cells`

**问题**：merge 时把 source cell status 改成 `superseded`，但没切断它原来的 outgoing active ray。结果是 superseded cell 仍以 active ray 出发指向其它节点，污染 `find_orphans`、`stats.active_rays`、连通分量计算。

**对照 spec**（tree_store.md merge_cells 完整序列）：spec 实际也没显式写这一步，但语义上 superseded == 死细胞，外向 ray 必须 sever，与 quarantine 一致。

**修复方案**：

```python
# merge_cells 中,在每个 source supersede 后加:
for ray in self.kuzu.get_outgoing_rays(source_id):
    if ray["status"] == "active":
        self.oplog.append(OpType.SEVER_RAY, {
            "from_id": source_id, "to_id": ray["to_id"],
            "reason": "merged",
        }, episode_id)
        self.kuzu.update_ray_status(source_id, ray["to_id"], "severed")
```

`split_cell` 同理需要补。

**spec 同步**：在 `docs/specs/tree_store.md` 的 merge_cells 序列里补这一步。

---

### Bug 2: `KuzuBackend.add_ray` 先删后建会丢失历史时间戳

**位置**：`src/tree_harness/store/kuzu_backend.py: add_ray`

**问题**：当前实现是先 DELETE 后 CREATE 来做幂等，副作用是已有 ray 的 `created_at` 和 `last_activated` 会被覆盖为 "now"。

**潜在影响**：
- 违反 Connector spec 契约 "connect 只建立新 ray，不修改已有 ray"
- 影响 idle 判定（DecaySentinel 后续会基于 last_activated 判 idle）
- 影响实验日志的时间序列分析

**修复方案**：检查存在性，存在则不做或仅更新 weight。

```python
def add_ray(self, from_id: str, to_id: str, weight: float) -> None:
    weight = self._clip_weight(weight)
    self._check_ray_direction(from_id, to_id)
    # 检查是否已存在
    existing = self._query_all(
        "MATCH (a:CellRef {id: $f})-[r:RAY]->(b:CellRef {id: $t}) "
        "RETURN r.weight AS w, r.status AS s",
        {"f": from_id, "t": to_id},
    )
    if existing:
        # 已存在:不重建,保留时间戳
        return
    now = datetime.now(timezone.utc).isoformat()
    self._execute(
        "MATCH (a:CellRef {id: $f}), (b:CellRef {id: $t}) "
        "CREATE (a)-[:RAY {weight: $w, status: 'active', "
        "created_at: $now, last_activated: $now}]->(b)",
        {"f": from_id, "t": to_id, "w": weight, "now": now},
    )
```

如果调用方真要更新已有 ray 的 weight，应改用 `update_ray_weight`，而不是依赖 add_ray 的"幂等"语义。

---

## P1 — 应该修复

### Bug 3: SQLite 与 KuzuDB 的写操作不在同一事务

**位置**：`src/tree_harness/store/{sqlite_backend.py, kuzu_backend.py}` + `tree_store.py`

**问题**：当前每个写操作都各自 commit，三库（SQLite cells、KuzuDB、OpLog）之间无原子性保证。如果 sqlite 提交后 kuzu 失败，残留状态需要靠 `consistency_check` 检出，但目前只能检出，没有自动修复路径。

**修复方案**（轻量版）：

1. 给 `TreeStore` 增加 `repair()` 方法：根据 oplog replay 缺失部分。
2. `SQLiteBackend` 暴露 `begin/commit/rollback`，TreeStore 在多步操作里把 sqlite 写入推迟到全部成功后再 commit。
3. KuzuDB 失败时，主动 rollback SQLite。

短期可以只做（1），保证有恢复能力即可。

**spec 同步**：在 `tree_store.md` 错误处理章节补 `repair()` 接口定义。

---

### Bug 4: `EnergySystem.reference / challenge` 不返回是否生效

**位置**：`src/tree_harness/modules/energy_system.py`

**问题**：当 cell 不存在或非 active 时静默 return None。调用方（未来 ContextInjector 的强化阶段）无法区分"成功强化"和"目标已死"。

**修复方案**：

```python
def reference(self, cell_id: str, episode_id: str) -> bool:
    """返回 True 表示成功强化,False 表示目标不存在或非 active。"""
    cell = self.tree_store.get_cell(cell_id)
    if cell is None or cell.status != "active":
        return False
    new_energy = cell.energy + self.config.delta_reference
    self.tree_store.update_energy(cell_id, new_energy, "reference", episode_id)
    return True
```

challenge 同理。

---

### Bug 5: `consistency_check` 漏检非 active cell

**位置**：`src/tree_harness/store/tree_store.py: consistency_check`

**问题**：当前只扫描 SQLite active cell。但 superseded / quarantined cell 在 KuzuDB 中应该仍存在节点（用于 SUPERSEDES 链回溯），它们的 ring 是否同步、id 是否匹配都没检查。

**修复方案**：扫描所有 status 的 cell，不仅 active。

```python
def consistency_check(self) -> List[str]:
    inconsistent = set()
    kuzu_refs = {r["id"]: r["ring"] for r in self.kuzu.get_all_refs()}
    # 扫描所有 status 的 cell (含 superseded/quarantined)
    for status in _ALL_STATUSES:
        for cell in self.sqlite.list_by_status(status):  # 需新增
            if cell.id not in kuzu_refs:
                inconsistent.add(cell.id)
            elif kuzu_refs[cell.id] != cell.ring:
                inconsistent.add(cell.id)
    for kid in kuzu_refs:
        if self.sqlite.get_cell(kid) is None:
            inconsistent.add(kid)
    return sorted(inconsistent)
```

---

## P2 — 可读性 / 性能 / 长期改进

### 改进 1: `RING_THRESHOLDS` 命名歧义

**位置**：`src/tree_harness/core/cell_model.py`

**问题**：`RING_THRESHOLDS["L1"]` 表示"maturity 落在 [0.15, 0.40) 时归 L1"，但代码里很多地方用 `RING_THRESHOLDS[next_ring][0]` 取"升入 next_ring 的阈值"——含义混淆。

**建议**：拆成两个命名清晰的字典：

```python
MATURITY_RING_RANGES = {  # "落在哪个区间归哪个 ring"
    "L0": (0.00, 0.15),
    ...
}

PROMOTE_THRESHOLDS = {  # "升入此 ring 需要的最小 maturity"
    "L0→L1": 0.15,
    "L1→L2": 0.40,
    ...
}
```

升降逻辑里全部用 PROMOTE_THRESHOLDS，结果能直接对应 spec 文档。

---

### 改进 2: `OpLog._payload_contains` 全表递归扫描

**位置**：`src/tree_harness/core/oplog.py: get_cell_history`

**问题**：在大 oplog 下 O(N×M) 慢。SWE-bench 跑 500 task × 10 op/task = 5000 条 oplog 还能跑，但跑全 SWE-bench Verified 时会成为瓶颈。

**建议**：在 INSERT_CELL/UPDATE_*/PROMOTE 等已知字段位置做关键字索引列：

```sql
ALTER TABLE oplog ADD COLUMN cell_id_indexed TEXT;
CREATE INDEX idx_oplog_cell ON oplog(cell_id_indexed);
```

append 时按 op type 显式提取 cell_id 写入。

---

### 改进 3: `KuzuBackend.get_connected_component` 查询次数过多

**位置**：`src/tree_harness/store/kuzu_backend.py`

**问题**：每个节点 2 次查询，N 节点 2N 次。

**建议**：一次性 Cypher 路径查询：

```cypher
MATCH (c:CellRef {id: $id})-[:RAY*]-(n:CellRef)
RETURN DISTINCT n.id AS id
```

---

### 改进 4: `DeterministicEmbedder` 死代码

**位置**：`src/tree_harness/core/embedding.py: DeterministicEmbedder.embed`

**问题**：`if i >= len(seed)` 分支永远不会触发（外层 `seed[i % len(seed)]` 已 wrap）。代码无害但读起来困惑。

**建议**：直接删掉那段 `if`，简化为：

```python
def embed(self, text: str) -> List[float]:
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    return [(seed[i % len(seed)] / 255.0) * 2.0 - 1.0 for i in range(self._dim)]
```

如果担心维度过大时单调，可以引入 chained hash：

```python
def embed(self, text: str) -> List[float]:
    vec, buf = [], hashlib.sha256(text.encode("utf-8")).digest()
    while len(vec) < self._dim:
        for b in buf:
            vec.append((b / 255.0) * 2.0 - 1.0)
            if len(vec) >= self._dim:
                break
        buf = hashlib.sha256(buf).digest()
    return vec
```

---

## 修复优先级建议

| 优先级 | 项 | 阻塞下个 Phase 吗 |
|--------|----|---|
| P0 | Bug 1 (merge sever) | 阻塞 Phase 4 木质化测试 |
| P0 | Bug 2 (add_ray 时间戳) | 阻塞 Phase 3 idle 判定 |
| P1 | Bug 3 (事务原子性) | 不阻塞,但生产前必修 |
| P1 | Bug 4 (return bool) | 不阻塞,Phase 5 ContextInjector 时再补 |
| P1 | Bug 5 (consistency 全量) | 不阻塞 |
| P2 | 改进 1-4 | 不阻塞,有时间再清理 |

**建议路径**：先修 Bug 1 + Bug 2（影响后续 Phase 的语义正确性），其它在进入对应 Phase 前再处理。

---

## 待确认的设计问题（不是 bug,需要决策）

1. **superseded cell 的 KuzuDB 节点保留多久？** 当前永久保留以维护 SUPERSEDES 链。是否需要"超过 N 个 episode 后归档"机制？
2. **OpLog 大小上限？** 当前无限增长。是否需要按 episode_id 分片或定期 archive？
3. **consistency_check 的触发时机？** spec 里说"每 100 episode 扫描一次"，目前没有自动触发——需要 Runner 层加调度。

这些可以放到下次 review 或 design decision 里讨论。
