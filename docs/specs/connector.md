# Connector Spec

## 概述

Connector 是 Self-Evolution Operator Set 中 `connect` 算符的实现策略——具体承担 Cambium Engine 三步管线中的 Step C：为新创建的 cell 建立 RAY 连接。RAY 方向为外→内（新 cell 指向已有的同层或更内层 cell），表示"我引用/依赖了你"。

定位说明：Connector 不是独立的 harness 状态所有者，只是一个无状态子例程。它由 `CambiumEngine.crystallize_step()` 在 Step B 通过 INSERT_NEW 后调用一次，由 `OuterHarness.after_step()` 通过 `cambium.connect_new_cells()` 调用作为批量再连边入口。它不调任何其他算符、不写 oplog（oplog 写入由 TreeStore 在 add_ray 时附带完成）、不感知 ring 升降级。这一切分让 `connect` 算符的所有边可被 OpLog 完整审计——RAY_ADDED 事件总能精确归因到一次 Connector 调用。

## 接线流程

```
new_cell (刚从 Step B INSERT_NEW 出来)
    ↓
vec_search(new_cell.embedding, top_k=10, threshold=0.5)
    ↓
过滤：只保留 ring >= new_cell.ring 的候选（外→内原则）
    ↓
对每个候选计算 ray_weight = similarity * (1 + domain_bonus)
    ↓
取 top-5 by ray_weight
    ↓
建立 RAY 边
```

## 接口定义

```python
@dataclass
class ConnectorConfig:
    search_top_k: int = 10           # vec_search 返回数量
    search_threshold: float = 0.5    # 最低相似度
    max_rays_per_cell: int = 5       # 每个新 cell 最多建 5 条 ray
    domain_overlap_bonus: float = 0.2  # domain 重叠时的权重加成


class ConnectorProtocol(Protocol):
    def __init__(self, tree_store: TreeStore, config: ConnectorConfig):
        ...

    def connect(self, new_cell: Cell) -> list[tuple[str, float]]:
        """
        为 new_cell 寻找并建立 ray 连接。
        返回实际建立的 ray 列表: [(target_cell_id, weight)]
        """
        ...

    def _find_candidates(self, new_cell: Cell) -> list[tuple[Cell, float]]:
        """向量检索 + ring 过滤"""
        ...

    def _compute_weight(self, new_cell: Cell, target: Cell, similarity: float) -> float:
        """计算 ray weight"""
        ...

    def _select_top(self, candidates: list[tuple[Cell, float, float]]) -> list[tuple[Cell, float]]:
        """排序并取 top-k"""
        ...
```

## Ray Weight 计算

```python
def _compute_weight(self, new_cell, target, similarity):
    # domain 重叠检测
    overlap = len(set(new_cell.domain_tags) & set(target.domain_tags))
    domain_bonus = min(overlap * self.config.domain_overlap_bonus, 0.4)  # cap at 0.4

    weight = similarity * (1 + domain_bonus)
    return min(weight, 1.0)  # clip to [0, 1]
```

## Ring 过滤规则

```python
RING_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}

def _ring_filter(self, new_cell_ring: str, target_ring: str) -> bool:
    """外→内原则：target ring 必须 >= new_cell ring"""
    return RING_ORDER[target_ring] >= RING_ORDER[new_cell_ring]
```

允许同层连接（横向 ray），但不允许向更外层连接。这保证了 ray 网络的整体方向性：知识溯源总是从表层指向深层。

## 特殊情况

### Tree 初期（cell 总量少）

当可用的过滤后候选不足 max_rays_per_cell 时，有多少接多少。最极端情况（tree 只有 1 个 cell）只建 0~1 条 ray。

### 新 cell 位于 L0

L0 cell 可以连接到任何层（L0~L4），因为 L0 是最外层。这是最常见的情况，新蒸馏的 cell 初始都在 L0。

### 新 cell 由 merge 产生（位于 L2+）

merge 产出的 cell 位于较高层级，此时只搜索同层或更内层的目标。这确保合并后的抽象知识连接到同等或更抽象的已有知识。

## 行为契约

1. connect 只建立新 ray，不修改已有 ray
2. 不连接到 status != "active" 的 cell
3. 不连接到自身（self-loop 禁止）
4. 返回的 weight 在 (0, 1] 范围内
5. ray 建立后立即在 KuzuDB 中可查询

## 测试用例

1. 新 L0 cell + tree 中有 5 个 L1-L3 cell → 建立 ray，方向全部正确（新→旧）
2. 新 L0 cell + tree 为空 → 返回空列表（无 ray）
3. 新 L2 cell → 不连接到 L0/L1 的 cell（ring 过滤生效）
4. domain_tags 有重叠 → weight 比无重叠的高
5. similarity=0.45 的候选 → 被过滤掉（低于 threshold=0.5）
6. 10 个候选中选出 top-5 → 按 weight 降序取
7. quarantined cell 不被选为 ray target
8. ray weight 超过 1.0 → clip 到 1.0
