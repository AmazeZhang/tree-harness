# Cell Model Spec

## 概述

Cell 是 Self-Evolution Operator Set 五算符 `{crystallize, connect, promote, quarantine, decay}` 作用的**最小状态单元**——即 outer_harness.md 中 `H = (C, R, E, ρ)` 形式化里的 C 集合元素。每个 cell 承载一条 (Context, Decision, Rationale) 三元组，是算符可被实施、可被回放的最小颗粒度。

定位说明：Cell 不是"记忆条目"，而是算符 domain 的代数对象。所有 cell 字段（status / ring / energy / maturity / source_type / verify_hints …）的存在都必须可追溯到某个算符的输入或输出；不被任何算符消费的字段视为冗余，应在 spec 评审中删除。

## 数据模型

```python
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime

@dataclass
class VerifyHint:
    type: Literal["file_grep", "ast_query", "lockfile_query", "test_id_lookup", "env_check"]
    params: dict  # type-specific params

@dataclass
class Precondition:
    kind: Literal["fact", "config", "dependency", "code_invariant", "test_existence", "convention"]
    assertion: str  # NL 断言，永远保留
    verify_hint: Optional[VerifyHint] = None  # 可选的机器验证提示

@dataclass
class Cell:
    id: str
    ring: Literal["L0", "L1", "L2", "L3", "L4"]
    maturity: float  # 0.0 ~ 1.0
    energy: float    # 可为负
    
    # 内容三元组
    context_trigger_task: str       # 触发 task id
    context_domain: str             # 领域标签
    context_preconditions: list[Precondition]
    decision: str                   # NL 决策描述
    rationale: str                  # NL 理由
    
    # 元数据
    evidence: list[str]             # ["test_id:xxx", "commit:yyy", "file:zzz"]
    domain_tags: list[str]          # 领域标签列表
    status: Literal["active", "quarantined", "superseded", "archived"]
    source: Literal["distilled", "user_directive", "seed"]
    needs_review: bool = False      # Verdict=uncertain 时由 Decay Sentinel 置位；下次 funnel verification 优先抽样
    
    # 向量检索字段（由 SQLiteBackend 落盘到 BLOB 列；in-memory Cell 对象上可为 None）
    embedding: Optional[list[float]] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    superseded_by: Optional[str] = None  # 指向新 cell id
```

字段补充说明：

- `embedding`：on-disk 形态为 sqlite-vec BLOB（见 sqlite_backend.md schema）；in-memory Cell 对象上 `embedding` 可为 None（按需 lazy-load）。CandidateCell（dedup.md）携带的 embedding 在 INSERT_NEW 时由 TreeStore.insert_cell 写入 Cell 与 SQLite。
- `needs_review`：仅由 Decay Sentinel 在 Verdict=`uncertain` 时通过 `TreeStore.update_cell(cell_id, needs_review=True)` 置位；由 funnel verification 在下次抽样时降权读取（实现细节见 decay_sentinel.md Step 0）。被 `quarantine` 算符执行后该字段保留以便 oplog 审计，不主动清零。

## ID 生成规则

格式：`cell-{timestamp}-{random_suffix}`

```python
import uuid
from datetime import datetime

def generate_cell_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"cell-{ts}-{suffix}"
```

## Ring 与 Maturity 映射

本节是 ring 阈值的**唯一权威源**。`ring_promotion.md`（决策子模块）与 `lignification.md`（执行子模块）均从此处 import，不允许在其他 spec / Config 中重复定义同名常量。

```python
# 升层阈值：跨越事件的 maturity 下沿
RING_THRESHOLDS: dict[str, float] = {
    "L0→L1": 0.15,
    "L1→L2": 0.40,
    "L2→L3": 0.65,
    "L3→L4": 0.85,
}

# 降层阈值：从当前 ring 降出的 maturity 上沿
RING_DEMOTE_THRESHOLDS: dict[str, float] = {
    "L1→L0": 0.05,
    "L2→L1": 0.30,
    "L3→L2": 0.55,
    "L4→L3": 0.75,
}

# 滞回带：promote - demote = 0.10（任意层）
# 这条等式由 ring_promotion.md 的 hysteresis_min_gap 不变量校验

# 派生：由 maturity 反查 ring 归属（仅用于初值映射 / 调试可视化，不参与升降层判定——
# 升降层一律走 RING_THRESHOLDS / RING_DEMOTE_THRESHOLDS 的跨越事件判定，避免双轨）
def ring_of(maturity: float) -> str:
    if maturity < 0.15: return "L0"
    if maturity < 0.40: return "L1"
    if maturity < 0.65: return "L2"
    if maturity < 0.85: return "L3"
    return "L4"
```

## 初始值

| 字段 | 新蒸馏 cell | 用户指令 cell |
|------|------------|-------------|
| ring | L0 | L4 |
| maturity | 0.0 | 0.85 |
| energy | 0.5 | 1.0 |
| status | active | active |
| source | distilled | user_directive |

## 行为契约

1. `id` 全局唯一，一旦生成不可更改
2. `decision` 和 `rationale` 创建后不可修改（公理六）
3. `ring`、`maturity`、`energy`、`status`、`needs_review` 可更新，每次更新必须写 op log
4. `maturity` 硬截断在 [0, 1]
5. `energy` 无上下限（但建议实现中用 soft cap）
6. `embedding` 在 INSERT_NEW 时由 SQLiteBackend 计算并写入；之后视为不可变（embedding 模型升级走全库重建流程，不在线改写）
7. `needs_review` 只由 Decay Sentinel 通过 `TreeStore.mark_for_review` 翻转；其他模块（Cambium / EnergySystem / Lignification）不得读写该字段

## 测试用例

1. 创建一个标准 cell，验证所有字段有值（包括 needs_review=False、embedding 维度匹配模型）
2. 创建 user_directive cell，验证初始 ring=L4, energy=1.0
3. 验证 ID 格式正确且唯一（生成 1000 个无重复）
4. 验证 maturity 超过 1.0 时被截断为 1.0
5. 验证 maturity 低于 0.0 时被截断为 0.0
6. `mark_for_review(cell_id, True)` → 该 cell needs_review 变为 True；之后 `quarantine(cell_id)` 不会自动清零 needs_review
7. 尝试修改 embedding 字段 → 应抛异常或被 SQLiteBackend 拒绝
