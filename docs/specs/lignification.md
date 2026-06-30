# Lignification Scheduler Spec

## 概述

Lignification Scheduler 是 Self-Evolution Operator Set 中 `promote` 算符的实现策略（同时承担 `merge` / `split` 复合算符的子例程，二者按 outer_harness.md 定义等价于 `crystallize + quarantine` 的策略性组合）。它本身**不是 harness 状态的所有者**，只是一个无状态算法服务，由 `OuterHarness.after_episode()` 在 episode 末调用。

定位说明：Scheduler 直接对应 Harness Card **stability** 属性——promote 的滞回控制（ring hysteresis 0.10 dead zone）与 episode 节拍是稳定性的物理基础。Scheduler 不自行决定何时跑维护周期，节奏完全由 OuterHarness 控制；也不写 warning（quarantine 信号传播在 after_step 里已完成）。Maturity 在 framing 中被重定位为 promote 的滞后窗口，不是独立健康轴。

## 配置

```python
@dataclass
class LignificationConfig:
    """Lignification 自身只持有维护周期相关参数；ring 阈值不在此处定义。

    升降层阈值的唯一权威源是 `cell_model.RING_THRESHOLDS` /
    `RING_DEMOTE_THRESHOLDS`（箭头格式 key），本模块 import 使用，
    不允许在 Config 中重复定义同名字段。滞回带 = 0.10（任意层），
    由 cell_model.md 的不变量保证。
    """
    # Ring capacity（与 ring_promotion.md 共享，由 ring_promotion 决策端写入）
    ring_capacity: dict[str, int] = field(default_factory=lambda: {"L3": 60, "L4": 20})
    overflow_policy: str = "force_promote"   # "force_promote" | "demote_oldest" | "block_new"

    # Merge / split
    merge_similarity_threshold: float = 0.80
    merge_max_cluster_size: int = 5
    enable_split: bool = False               # 默认关闭，仅 LLM 主动发现时启用
```

## 数据类

```python
@dataclass
class MaintenanceResult:
    """run_maintenance_cycle 的返回值，字段与 EpisodeReport 对齐。"""
    promoted: list[tuple[str, str, str]]    # (cell_id, from_ring, to_ring)
    demoted: list[tuple[str, str, str]]
    merged: list[tuple[list[str], str]]     # (source_ids, merged_cell_id)
    split: list[tuple[str, list[str]]]      # (source_id, child_ids)
    op_counts: dict[str, int]               # 本次 cycle 触发的算符调用计数（PROMOTE / DEMOTE / MERGE / SPLIT）
```

## 接口定义

```python
class LignificationSchedulerProtocol(Protocol):
    def __init__(self, tree_store: TreeStore, energy_system: EnergySystem,
                 llm_client: LLMClient, oplog: OpLog,
                 config: "LignificationConfig"):
        ...

    def check_promotions(self) -> list[tuple[str, str, str]]:
        """检查并执行所有待升层的 cell，返回 [(cell_id, from_ring, to_ring), ...]"""
        ...

    def check_demotions(self) -> list[tuple[str, str, str]]:
        """检查并执行所有待降层的 cell"""
        ...

    def attempt_merge(self, candidate_ids: list[str]) -> Optional[str]:
        """尝试合并一组 cell，返回新 cell id（如果成功）"""
        ...

    def attempt_split(self, cell_id: str) -> Optional[list[str]]:
        """尝试分裂一个 cell，返回新 cell id 列表（如果成功）"""
        ...

    def run_maintenance_cycle(self, episode_id: str) -> MaintenanceResult:
        """执行一轮完整的木质化维护，返回结构化结果"""
        ...
```

注意：`tree_store.update_cell` / `tree_store.kuzu.xxx` 形式的直接访问在本 spec 内的伪代码中尚有遗留，**实现时必须改走 TreeStore facade**（`tree_store.promote / merge_cells / split_cell` 等专用方法），以满足 tree_store.md 不变量。下文伪代码将逐步迁移。

## Promote（纯升层）

触发条件：`maturity >= RING_THRESHOLDS[f"{current_ring}→{next_ring}"]`（从 cell_model.md import）

操作：
```python
def promote(self, cell_id: str, target_ring: str, reason: str = "normal"):
    self.tree_store.update_cell(cell_id, ring=target_ring)
    self.tree_store.kuzu.update_cell_ring(cell_id, target_ring)
    self.oplog.append("PROMOTE", {
        "cell_id": cell_id,
        "from_ring": old_ring,
        "to_ring": target_ring,
        "reason": reason,            # "normal" | "overflow_force" | "overflow_demote"
    })
```

- 原地修改 ring 字段
- 不创建新 cell
- 不影响 ray 拓扑
- energy/maturity 连续，不重置

### 容量执行

每次 promote 进入 L3 / L4 前，Scheduler 必须查 `RingPromotion.ring_capacity` 并按 `overflow_policy` 处理（见 `ring_promotion.md` Ring Capacity Policy）：

```python
def _enforce_capacity(self, target_ring: str) -> Optional[str]:
    """返回触发的 overflow reason，若无溢出返回 None"""
    if target_ring not in self.config.ring_capacity:
        return None
    active_count = self.tree_store.count_active_by_ring(target_ring)
    if active_count < self.config.ring_capacity[target_ring]:
        return None
    policy = self.config.overflow_policy
    if policy == "force_promote" and target_ring == "L3":
        return "overflow_force"     # 调用方应直接升 L4
    if policy in ("force_promote", "demote_oldest"):
        victim = self.tree_store.oldest_active_in_ring(target_ring, by="maturity")
        self.promote(victim, demote_one_level(target_ring), reason="overflow_demote")
        return "overflow_demote"
    if policy == "block_new":
        return "block_new"
    raise ValueError(policy)
```

Scheduler 不允许在不写 OpLog 的情况下修改 ring（不变量 I-Cap）。Capacity 溢出导致的所有移动都视为 promote 算符调用，oplog 中 reason 字段记录溢出根因，供 metrics.md H5 统计 `promote.reason` 维度。

## Demote（降层）

触发条件：`maturity < RING_DEMOTE_THRESHOLDS[f"{current_ring}→{prev_ring}"]`（从 cell_model.md import）

操作同 promote，方向相反。

## Merge（合并）

触发条件：
- 多个同 ring 的 cell，embedding similarity > 0.80
- 共享同一 domain_tag
- 都处于 active 状态
- 由 LLM 确认"它们表达的是同一件事的不同方面"

操作：
```python
def merge(self, source_ids: list[str]) -> str:
    # 1. LLM 生成合并后的 decision/rationale
    merged_content = self.llm_merge(source_cells)
    
    # 2. 计算继承属性
    merged_energy = max(source_energies) * 0.8
    merged_maturity = mean(source_maturities)
    merged_ring = ring_of(merged_maturity)
    
    # 3. 创建新 cell
    new_cell = Cell(
        id=generate_id(),
        ring=merged_ring,
        energy=merged_energy,
        maturity=merged_maturity,
        ...merged_content
    )
    self.tree_store.insert_cell(new_cell)
    
    # 4. 建立 SUPERSEDES 边
    for src_id in source_ids:
        self.tree_store.kuzu.add_supersedes(new_cell.id, src_id)
        self.tree_store.update_cell(src_id, status="superseded", superseded_by=new_cell.id)
    
    # 5. 迁移 incoming ray 到新 cell
    for src_id in source_ids:
        incoming = self.tree_store.kuzu.get_incoming_rays(src_id)
        for ray in incoming:
            self.tree_store.kuzu.add_ray(ray["from_id"], new_cell.id, ray["weight"])
    
    # 6. op log
    self.oplog.append("MERGE", {"source_ids": source_ids, "target_id": new_cell.id})
    
    return new_cell.id
```

## Split（分裂）

触发条件：LLM 在 merge 尝试中发现一个 cell 实际包含多个独立知识。

操作：
```python
def split(self, source_id: str, split_contents: list[dict]) -> list[str]:
    source = self.tree_store.get_cell(source_id)
    new_ids = []
    
    for content in split_contents:
        child = Cell(
            id=generate_id(),
            ring=ring_of(source.maturity * 0.8),
            energy=source.energy * 0.6,
            maturity=source.maturity * 0.8,
            ...content
        )
        self.tree_store.insert_cell(child)
        self.tree_store.kuzu.add_supersedes(child.id, source_id)
        new_ids.append(child.id)
    
    self.tree_store.update_cell(source_id, status="superseded")
    self.oplog.append("SPLIT", {"source_id": source_id, "target_ids": new_ids})
    return new_ids
```

## Maintenance Cycle

```python
def run_maintenance_cycle(self, episode_id: str) -> MaintenanceResult:
    promoted = self.check_promotions()       # [(cell_id, from_ring, to_ring), ...]
    demoted = self.check_demotions()
    merged: list[tuple[list[str], str]] = []
    split: list[tuple[str, list[str]]] = []

    clusters = self._find_merge_candidates()
    for cluster in clusters:
        new_id = self.attempt_merge(cluster)
        if new_id:
            merged.append((cluster, new_id))

    if self.config.enable_split:
        for cell_id in self._find_split_candidates():
            children = self.attempt_split(cell_id)
            if children:
                split.append((cell_id, children))

    op_counts = {
        "PROMOTE": sum(1 for _ in promoted),
        "DEMOTE": sum(1 for _ in demoted),
        "MERGE": len(merged),
        "SPLIT": len(split),
    }
    return MaintenanceResult(
        promoted=promoted, demoted=demoted,
        merged=merged, split=split,
        op_counts=op_counts,
    )
```

## 测试用例

1. maturity=0.41 的 L1 cell → promote 到 L2，ring 字段变化，其他不变
2. maturity=0.04 的 L1 cell → demote 到 L0
3. maturity=0.35 的 L1 cell → 不触发 promote 也不触发 demote（滞回带内）
4. 3 个相似 L1 cell merge → 产出 1 个新 cell + 3 条 SUPERSEDES 边
5. merge 后新 cell 的 energy = max(sources) * 0.8
6. merge 后新 cell 的 maturity = mean(sources)
7. merge 后源 cell 的 status = "superseded"
8. split 后子 cell 的 energy = parent * 0.6
9. split 后子 cell 的 maturity = parent * 0.8
