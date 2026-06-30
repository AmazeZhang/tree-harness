# OpLog Spec

## 概述

OpLog 是 Self-Evolution Operator Set 五算符 `{CRYSTALLIZE, CONNECT, PROMOTE, QUARANTINE, DECAY}` 的 append-only 事件日志（含 `MERGE` / `SPLIT` 复合算符与 `REFERENCE` / `CHALLENGE` 反向特例）。所有对 TreeStore 的写操作必须先经过 OpLog 落 op_type，违反者视为 bug（outer_harness.md I-Op2）。

定位说明：OpLog 是 harness-level metrics 的测量底座——`op_count_distribution`（算符封闭性回归）、`control_lag`（quarantine→warning 注入 step 距离）、`entropy_release_per_episode`、`ring_oscillation_rate` 全部从 OpLog 推导。同时支撑确定性回放（重放 op 序列 → 重建 H 状态），是 outer_harness.md 中 before_step 只读约束的实现保障。

## OP_TO_OPERATOR 映射（I-Op2 实现）

**设计决策**：OpLog 走 **"底层 op_type + 单射算符映射表"**，不在存储层引入语义级 op_type。理由：

1. **回放确定性**：底层 op_type 与 TreeStore 方法严格一对一，replay 引擎拿到 `op` 直接 dispatch，不需要解析 payload 二次判别。
2. **信息量不可逆压缩**：`INSERT_RAY` 与 `SEVER_RAY` 都归 `CONNECT` 算符却是互逆操作；`REINFORCE` 与 `INSERT_RAY` 语义不同但同属 `CONNECT`——压成单一 op_type 会损失 control lag / debug 精度。
3. **store low, query high**：5 算符聚合属查询层职责（`count_by_op_type` 内部 GROUP BY），不是存储层职责。
4. **I-Op2 从文档承诺升级为代码约束**：下面这张单射表在模块加载时通过 `assert set(OP_TO_OPERATOR.keys()) == set(OpEnum)` 强制全覆盖，新增 op 忘了归类直接 import 失败。

```python
# oplog.py 顶层；唯一权威映射
OP_TO_OPERATOR: dict[str, Literal["CRYSTALLIZE","CONNECT","PROMOTE","QUARANTINE","DECAY"]] = {
    # CRYSTALLIZE
    "INSERT_CELL":       "CRYSTALLIZE",
    # CONNECT（含权重变化、激活、反向特例 REINFORCE）
    "INSERT_RAY":        "CONNECT",
    "UPDATE_RAY_WEIGHT": "CONNECT",
    "SEVER_RAY":         "CONNECT",
    "REINFORCE":         "CONNECT",
    "RAY_ACTIVATED":     "CONNECT",
    # PROMOTE（含复合算符 MERGE / SPLIT、降层 DEMOTE）
    "PROMOTE":           "PROMOTE",
    "DEMOTE":            "PROMOTE",
    "MERGE":             "PROMOTE",
    "SPLIT":             "PROMOTE",
    # QUARANTINE（含版本替代、review 标记）
    "QUARANTINE":        "QUARANTINE",
    "SUPERSEDE":         "QUARANTINE",
    "MARK_REVIEW":       "QUARANTINE",
    # DECAY
    "UPDATE_ENERGY":     "DECAY",
    "UPDATE_MATURITY":   "DECAY",
}

# 模块加载时强制校验 I-Op2：枚举里每个 op 必须在映射表里
assert set(OP_TO_OPERATOR.keys()) == set(OpEnum), \
    "I-Op2 violated: op_type without operator binding"
```

归类说明：

- `MERGE` / `SPLIT` 归入 `PROMOTE` 算符是因为它们都改变 ρ 映射（cell 在 ring 中的归属或被新 cell 取代）。
- `REINFORCE` 与 `RAY_ACTIVATED` 归入 `CONNECT` 是因为它们都作用在 R 集合的权重维度。
- `MARK_REVIEW` 归入 `QUARANTINE` 算符是因为它是 funnel verification 在 Verdict=`uncertain` 时为下次抽样置位的标志——属 quarantine 的预备信号，但本身不切换 cell.status。
- `UPDATE_MATURITY` 归入 `DECAY` 而非 `PROMOTE`：maturity 是 promote 算符的滞后窗口/cooldown 计数器（见 ring_promotion.md），其更新源（reference 加成 / idle 衰减）本质都属 `decay` 算符的反向特例。

## 日志条目格式

```python
@dataclass
class OpLogEntry:
    seq: int              # 自增序号
    timestamp: str        # ISO 8601
    op: str               # 底层 op_type（OpEnum 之一）
    operator: Literal["CRYSTALLIZE","CONNECT","PROMOTE","QUARANTINE","DECAY"]
                          # 派生字段；append 时由 OP_TO_OPERATOR[op] 自动填充，不可由调用方传入
    payload: dict         # 操作参数
    episode_id: Optional[str] = None  # 关联的 episode
```

`operator` 字段在写入时自动派生并落盘到 SQLite oplog 表的索引列，`count_by_op_type` 直接 `GROUP BY operator` 即可——避免每次 metrics 查询都重算映射。该字段是 I-Op2 在存储层的物理体现。

## 操作类型枚举

底层 op 一览（粒度细，便于回放/调试；语义层聚合到 5 算符见上节）。枚举仅扩不删——新增 op 必须同步更新"5 算符 → 底层 op 映射"。

| op | payload | 语义 |
|----|---------|------|
| `INSERT_CELL` | `{cell_id, ring, decision_summary}` | 新 cell 入树 |
| `INSERT_RAY` | `{from_id, to_id, weight}` | 新建射线 |
| `UPDATE_ENERGY` | `{cell_id, old_energy, new_energy, reason}` | 能量变化 |
| `UPDATE_MATURITY` | `{cell_id, old_maturity, new_maturity}` | 成熟度变化 |
| `PROMOTE` | `{cell_id, from_ring, to_ring, reason}` | 升层；reason ∈ {normal, overflow_force, overflow_demote} |
| `DEMOTE` | `{cell_id, from_ring, to_ring, reason}` | 降层；reason ∈ {normal, overflow_force, overflow_demote} |
| `MERGE` | `{source_ids, target_id}` | 合并 |
| `SPLIT` | `{source_id, target_ids}` | 分裂 |
| `QUARANTINE` | `{cell_id, reason}` | 隔离（腐朽裁定） |
| `SUPERSEDE` | `{old_id, new_id}` | 版本替代 |
| `MARK_REVIEW` | `{cell_id, flag, reason}` | needs_review 标志位翻转（Verdict=uncertain 时由 Decay Sentinel 置 True） |
| `UPDATE_RAY_WEIGHT` | `{from_id, to_id, old_weight, new_weight}` | 射线权重变化 |
| `SEVER_RAY` | `{from_id, to_id, reason}` | 切断射线 |
| `REINFORCE` | `{cell_id, episode_id}` | 已有 cell 被重复引用 |
| `RAY_ACTIVATED` | `{from_id, to_id, episode_id}` | 已存在 ray 被本 episode 命中（control lag 测量用） |

## 接口定义

```python
class OpLogProtocol(Protocol):
    def append(self, op: str, payload: dict, episode_id: Optional[str] = None) -> int:
        """追加一条日志，返回 seq。

        约束：
        - 调用方只传 op（底层 op_type）/ payload / episode_id 三项。
        - `operator` 字段由 OP_TO_OPERATOR[op] 自动派生写入，调用方**不允许**显式传入，
          防止"调用方写错 operator 与 op 不匹配"。
        - 若 op 不在 OP_TO_OPERATOR 键集合中（即未注册到 OpEnum），抛 KeyError——
          这是 I-Op2 的运行时兜底（import-time assert 是首道防线）。
        """
        ...
    
    def get_entries(self, since_seq: int = 0, op_filter: Optional[str] = None) -> list[OpLogEntry]:
        """查询日志条目"""
        ...
    
    def get_latest_seq(self) -> int:
        """获取当前最新 seq"""
        ...
    
    def get_cell_history(self, cell_id: str) -> list[OpLogEntry]:
        """获取某 cell 的完整操作历史"""
        ...
    
    def get_episode_ops(self, episode_id: str) -> list[OpLogEntry]:
        """获取某 episode 产生的所有操作"""
        ...
    
    def replay(self, tree_store, from_seq: int = 0, to_seq: Optional[int] = None) -> None:
        """从 op log 重建 tree 状态"""
        ...

    # --- 聚合统计（供 metrics & EpisodeReport 消费） ---
    def count_by_op_type(self, episode_id: Optional[str] = None) -> dict[str, int]:
        """按 5 算符聚合（不是底层 op）。返回 5 个 key 的 dict：
        {"CRYSTALLIZE", "CONNECT", "PROMOTE", "QUARANTINE", "DECAY"}.

        实现：SQL `GROUP BY operator`（operator 是落盘的派生字段）。
        episode_id 为 None 时统计全局；不为 None 时仅统计该 episode 切片。
        缺失的算符 key 用 0 填充，保证下游 metrics H5 永远拿到 5 个 key。
        """
        ...

    def count_promotes_by_reason(self, episode_id: Optional[str] = None) -> dict[str, int]:
        """统计 PROMOTE / DEMOTE 中各 reason 的次数。

        返回 {"normal", "overflow_force", "overflow_demote"} → int。
        overflow 比例 > 10% 是 L3/L4 容量调参信号（见 metrics H5 / ring_promotion 容量策略）。
        """
        ...
```

## 存储方式

SQLite 同库，独立表：

```sql
CREATE TABLE oplog (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    op TEXT NOT NULL,                -- 底层 op_type（OpEnum 之一）
    operator TEXT NOT NULL,          -- 派生算符：CRYSTALLIZE/CONNECT/PROMOTE/QUARANTINE/DECAY
    payload_json TEXT NOT NULL,
    episode_id TEXT
);

CREATE INDEX idx_oplog_op       ON oplog(op);
CREATE INDEX idx_oplog_operator ON oplog(operator);   -- count_by_op_type 走这条索引 GROUP BY
CREATE INDEX idx_oplog_episode  ON oplog(episode_id);
```

`operator` 列由 `append()` 在写入时按 `OP_TO_OPERATOR[op]` 自动填充，与 `op` 列绑定一致；不允许由调用方直接传入或事后改写（append-only 行为契约第 1 条同时覆盖该列）。

## 行为契约

1. OpLog 只追加，永不修改或删除已有条目
2. seq 严格递增，无间隔
3. 每次 TreeStore 的写操作都必须先写 OpLog 再执行实际写入
4. replay 能从空数据库完整重建 tree 当前状态
5. 单个写操作可能产生多条 op log（如 MERGE = QUARANTINE * N + INSERT_CELL + INSERT_RAY * M）

## 测试用例

1. append 3 条 → get_entries 返回 3 条，seq 为 1,2,3
2. get_cell_history 只返回包含该 cell_id 的条目
3. get_episode_ops 只返回该 episode 关联的条目
4. op_filter 过滤正确（只返回 INSERT_CELL 类型）
5. replay 从空库重建 → 与直接写入结果一致
6. append(op="INSERT_RAY", ...) → 返回的 OpLogEntry.operator == "CONNECT"，且 SQLite 行 operator 列 == "CONNECT"（调用方未传该字段，由 OP_TO_OPERATOR 自动派生）
7. append(op="NOT_A_REAL_OP", ...) → 抛 KeyError（I-Op2 运行时兜底）
8. 混合写入 INSERT_CELL × 2 + INSERT_RAY × 3 + UPDATE_ENERGY × 1 → count_by_op_type() 返回 `{CRYSTALLIZE:2, CONNECT:3, PROMOTE:0, QUARANTINE:0, DECAY:1}`（缺失算符 0 填充）
