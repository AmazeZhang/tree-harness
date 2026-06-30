# Metrics Spec

## 概述

Metrics 模块定义 Tree Harness 实验的全部度量指标。指标分为四类：

1. **Harness-Level Metrics**（核心，对齐 Harness Card arXiv:2605.23950 三属性）——stability / context drift / control lag 量化与 self-evolution 算符调用统计
2. **效果指标**（task-level）——resolve rate、学习曲线
3. **结构健康度指标**（harness-state-level）——ring 分布、ray 连通、活/死比，描述 Tree 内部状态
4. **效率指标**（cost-level）——token 消耗、Pareto 效率

Harness-Level Metrics 是论文实验表格的主菜——它们是其他 outer harness（static / freeform）也能算出的可比较指标，确保对照公平。

## Harness-Level Metrics（核心）

### H1. Ring Oscillation Rate —— Stability

```python
def ring_oscillation_rate(oplog: OpLog, window: int = 100) -> float:
    """
    Stability 指标：window 内一个 cell 被 promote 后又被 demote 的比例。
    对齐 Harness Card 论文 "Stability"。

    osc = |{cell: cell 在 window 内既出现 promote 又出现 demote}| / |{cell: cell 在 window 内被 promote}|

    健康范围：< 0.10。> 0.30 表示 hysteresis 失效，harness 在震荡。
    """
    promote_ops = oplog.get_entries(op_filter="PROMOTE", window=window)
    demote_ops = oplog.get_entries(op_filter="DEMOTE", window=window)
    promoted = {op.cell_id for op in promote_ops}
    demoted_after = {op.cell_id for op in demote_ops if op.cell_id in promoted}
    return len(demoted_after) / len(promoted) if promoted else 0.0
```

四档对照预期：bare_inner / static_outer 无算符调用，此指标为 N/A；freeform_outer 因自由改写 scaffold，等价 oscillation 较高；tree_outer 受 hysteresis 约束应最低。

### H2. Context Retention Score —— Context Drift

```python
def context_retention_score(injection_log: list[dict], horizon: int = 10) -> float:
    """
    Context Drift 指标：被标记为关键的 cell（L3/L4 或 user_directive 来源）
    在 horizon 步内仍出现在 prompt 中的概率。
    对齐 Harness Card 论文 "Context Drift"。

    retention = mean over t of |{key_cells_in_prompt_at_t and prompt_at_t+horizon}|
                              / |key_cells_in_prompt_at_t|

    健康范围：> 0.90。Tree 借 pinned 段应接近 1.0。
    """
    ...
```

四档对照预期：bare_inner 无 retention 概念（无注入）；static_outer ≈ 1.0（注入文本不变）；freeform_outer 随 LLM 改写漂移；tree_outer 应稳定 ≈ 1.0。

### H3. Control Lag —— Control Lag

```python
def control_lag(oplog: OpLog) -> float:
    """
    Control Lag 指标：从一个 cell 被 quarantine 到对应 warning 注入下一步 prompt
    的平均 step 距离。
    对齐 Harness Card 论文 "Control Lag"。

    lag = mean over q in QUARANTINE ops of (next_warning_inject_step - q.step)

    Tree 理论值 = 1.0（after_step 内 quarantine，下一次 before_step 即注入）
    freeform_outer 取决于 LLM 是否在下次 rewrite 中提及，期望值 > rewrite_budget
    static_outer / bare_inner：无 quarantine，N/A
    """
    ...
```

### H4. Entropy Release Per Episode

```python
def entropy_release_per_episode(report: EpisodeReport, weights: dict) -> float:
    """
    本 episode 通过算符调用释放的"熵量"，量化 harness 自演化的强度。

    entropy = w_c * |crystallize 合并的源数| + w_q * |quarantine 数| + w_d * |decay below threshold 数|

    用于：(a) tree_outer 内部健康度趋势；(b) 与 freeform_outer 对比时观察"演化是否实际发生"。
    """
    return (weights["compressed"] * report.compressed_count
            + weights["quarantined"] * report.quarantined_count
            + weights["decayed"] * report.decayed_count)
```

### H5. Operator Call Distribution

```python
def op_count_distribution(oplog: OpLog) -> dict[str, int]:
    """
    每种 Self-Evolution Operator 的调用次数：
    {"crystallize": ..., "connect": ..., "promote": ..., "quarantine": ..., "decay": ...}

    用于：(a) 验证算符封闭性（无其他类型出现）；(b) 比较不同 harness 的演化模式。
    """
    return oplog.count_by_op_type()


def promote_reason_distribution(oplog: OpLog) -> dict[str, int]:
    """
    Promote 算符的 reason 维度分解（见 ring_promotion.md Ring Capacity Policy）：
    {"normal": ..., "overflow_force": ..., "overflow_demote": ...}

    诊断信号：overflow 比例 = (overflow_force + overflow_demote) / total_promote
    若 > 0.10，提示 L3/L4 ring_capacity 配置过低或 crystallize 阈值过松，
    应进入 experiment 复盘检查；正常长 horizon 项目应稳定在 < 0.05。
    """
    return oplog.count_promotes_by_reason()
```

### H6. HV/MV Ratio（Portability 子实验专用）

```python
def hv_mv_ratio(results_grid: dict) -> float:
    """
    Harness Variance / Model Variance 比，对齐 Harness Card 论文实验。
    输入：3 inner × 4 outer 的网格 results。
    输出：HV/MV 数值（论文该指标 = 7.80×，预期 Tree 能将该比值降至 < 3）。
    """
    ...
```

## 效果指标

### E1. Resolve Rate（核心指标）

```python
def resolve_rate(results: list[TaskResult], window: int = None) -> float:
    if window:
        results = results[-window:]
    return sum(1 for r in results if r.resolved) / len(results)
```

报告方式：
- 全局 resolve rate（单个数字，用于表格）
- 滑动窗口 resolve rate vs episode index（曲线图，展示 self-evolution 累积效果）

### E2. Cumulative Resolve Curve

```python
def cumulative_resolve_curve(results: list[TaskResult]) -> list[float]:
    curve = []
    resolved = 0
    for i, r in enumerate(results):
        resolved += int(r.resolved)
        curve.append(resolved / (i + 1))
    return curve
```

### E3. Improvement Over Baseline

```python
def relative_improvement(target_rate: float, baseline_rate: float) -> float:
    if baseline_rate == 0:
        return float('inf')
    return (target_rate - baseline_rate) / baseline_rate * 100
```

主比较：(tree_outer vs bare_inner)、(tree_outer vs static_outer)、**(tree_outer vs freeform_outer)** 为最关键比较。

## 结构健康度指标

### S1. Ray Connectivity Rate

```python
def ray_connectivity_rate(tree_store: TreeStore) -> float:
    """连通率 = (total_cells - orphan_count) / total_cells。健康 > 0.8"""
    total = tree_store.count_cells(status="active")
    orphans = len(tree_store.find_orphans())
    return (total - orphans) / total if total > 0 else 1.0
```

### S2. Active/Dead Ratio

```python
def active_dead_ratio(tree_store: TreeStore) -> float:
    """活/死比，健康 > 2.0"""
    active = tree_store.count_cells(status="active")
    dead = (tree_store.count_cells(status="quarantined") +
            tree_store.count_cells(status="superseded") +
            tree_store.count_cells(status="archived"))
    return active / dead if dead > 0 else float('inf')
```

### S3. Lignification Compression Ratio

```python
def lignification_compression(oplog: OpLog) -> float:
    """合并压缩比 = merged_source_count / merged_target_count"""
    merges = oplog.get_entries(op_filter="MERGE")
    total_sources = sum(len(m.payload["source_ids"]) for m in merges)
    total_targets = len(merges)
    return total_sources / total_targets if total_targets > 0 else 0.0
```

### S4. Ring Distribution

```python
def ring_distribution(tree_store: TreeStore) -> dict[str, int]:
    """健康树呈金字塔形（L0 > L1 > L2 > L3 > L4）"""
    return {ring: tree_store.count_cells(ring=ring, status="active")
            for ring in ["L0", "L1", "L2", "L3", "L4"]}
```

### S5. Centrality Distribution (Gini)

```python
def centrality_gini(tree_store: TreeStore) -> float:
    """入度中心性 Gini。健康 0.3~0.6"""
    ...
```

## 效率指标

### C1. Token Usage Per Episode

```python
def token_per_episode(results: list[TaskResult]) -> list[int]:
    return [r.token_usage for r in results]
```

### C2. Outer-Harness Overhead

```python
def outer_overhead(results: list[TaskResult]) -> float:
    """
    outer harness 维护开销占比 = outer_tokens / total_tokens
    包括：crystallize/funnel verification 的 LLM 调用 + freeform_outer 的 scaffold 重写。
    用于公平比较 freeform_outer（rewrite 成本高）vs tree_outer（结构化算符成本低）。
    """
    ...
```

### C3. Pareto Efficiency

```python
def pareto_front(conditions: dict[str, tuple[float, float]]) -> list[str]:
    """{condition: (resolve_rate, avg_token_cost)} → Pareto 前沿"""
    ...
```

## 时间序列采集

每 episode 采集一次快照：

```python
@dataclass
class EpisodeSnapshot:
    episode_index: int
    timestamp: str

    # 效果
    resolved: bool
    cumulative_resolve_rate: float

    # Harness-level
    ring_oscillation_count: int
    context_retention_score: float
    control_lag_steps: float
    entropy_released: float
    op_counts: dict          # {"crystallize":..,"promote":..,...}

    # 结构健康
    total_active_cells: int
    ring_distribution: dict[str, int]
    total_active_rays: int
    orphan_count: int

    # 成本
    token_usage: int
    outer_overhead_ratio: float
    duration_seconds: float
```

## 可视化需求

| 图 | X 轴 | Y 轴 | 系列 |
|----|------|------|------|
| 学习曲线 | episode index | cumulative resolve rate | bare/static/freeform/tree |
| 滑动解决率 | episode index | window resolve rate (w=10) | bare/static/freeform/tree |
| Stability | episode index | ring_oscillation_rate（滑窗）| freeform vs tree |
| Context retention | episode index | retention_score | static / freeform / tree |
| Control lag 分布 | lag (steps) | density | freeform vs tree |
| 算符调用分布 | operator | count | freeform vs tree（freeform 全 0）|
| 树生长 | episode index | active cell count | tree only |
| Ring 演化 | episode index | stacked ring count | L0~L4 |
| Token 效率 | episode index | token usage | bare/static/freeform/tree |
| Pareto | avg token/episode | resolve rate | 四组 |
| HV/MV | inner × outer | resolve rate matrix | portability 实验 |

## Ablation 指标

对 tree_outer 做组件消融，验证每个机制对 Harness Card 三属性的贡献：

```python
@dataclass
class AblationResult:
    full_system: dict            # 三属性指标 + resolve_rate
    no_hysteresis: dict          # 关闭 ring hysteresis → 预期 stability 下降
    no_pinned: dict              # 关闭 L3/L4 pinned → 预期 context retention 下降
    no_warning_injection: dict   # 关闭 quarantine → warning 注入 → 预期 control lag 变大
    no_lignification: dict       # 关闭 promote 算符
    no_decay: dict               # 关闭 decay 算符
    no_rays: dict                # 关闭 connect 算符（flat 结构），破坏 ray-mediated retrieval 与 quarantine 信号传播
```

每个 ablation 报告：(a) 三属性指标变化幅度；(b) resolve_rate 变化。验证"每个机制承担的属性是它声称的属性"。

## 测试用例

1. 10 个 TaskResult (5 resolved) → resolve_rate = 0.5
2. cumulative_resolve_curve 长度 = 输入长度
3. 全部 cell 有 ray → connectivity_rate = 1.0
4. 3 个 cell merge 为 1 个 → compression = 3.0
5. ring_distribution 总和 = active cell 总数
6. pareto_front 正确识别非支配解
7. 空 harness → 所有结构健康指标返回默认值（不崩溃）
8. EpisodeSnapshot 序列化为 JSONL 格式正确
9. ring_oscillation_rate：模拟 promote→demote 序列 → 期望比例
10. control_lag：人造 quarantine→warning 序列 → 期望 lag = 1
11. op_count_distribution 五个 key 之和 = oplog 中状态变更 op 总数（算符封闭性回归测试）
