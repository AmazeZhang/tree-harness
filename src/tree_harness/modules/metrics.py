"""Metrics — Tree Harness 实验度量指标。

四类指标:
1. Harness-Level (H1-H6): stability / context drift / control lag / 算符调用统计
2. 效果指标 (E1-E3): resolve rate / 学习曲线
3. 结构健康度 (S1-S5): ring 分布 / ray 连通 / 活死比
4. 效率指标 (C1-C3): token / Pareto

对应 spec: docs/specs/metrics.md
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple

from tree_harness.core.oplog import OpLog, OpEnum
from tree_harness.store.tree_store import TreeStore


# ===========================================================================
# 数据结构
# ===========================================================================
@dataclass
class TaskResult:
    """单 task 结果 (Runner 摊平后的单行记录)。"""
    task_id: str
    repo: str
    condition: str
    trial: int = 0
    episode_index: int = 0
    inner_kind: str = ""
    resolved: bool = False
    outcome: str = "fail"
    duration_seconds: float = 0.0
    token_usage: int = 0
    n_steps: int = 0

    # Harness-Level
    ring_oscillation_count: int = 0
    context_retention_score: float = 0.0
    control_lag_steps: float = 0.0
    entropy_released: float = 0.0
    op_counts: dict = field(default_factory=dict)

    # 算符明细 (from EpisodeReport)
    new_cells_count: int = 0
    compressed_count: int = 0
    quarantined_count: int = 0
    decayed_count: int = 0
    promoted: list = field(default_factory=list)
    demoted: list = field(default_factory=list)

    # 结构健康
    total_active_cells: int = 0
    ring_distribution: dict = field(default_factory=dict)

    # Inner 能力
    inner_supports_pin_marker: bool = False
    inner_supports_warning_marker: bool = False

    # Freeform outer 专用
    rewritten_prompt: Optional[str] = None

    # 成本
    outer_overhead_ratio: float = 0.0


@dataclass
class EpisodeSnapshot:
    """每 episode 采集一次的快照 (JSONL 序列化)。"""
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
    op_counts: dict

    # 结构健康
    total_active_cells: int
    ring_distribution: Dict[str, int]
    total_active_rays: int
    orphan_count: int

    # 成本
    token_usage: int
    outer_overhead_ratio: float
    duration_seconds: float

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeSnapshot":
        return cls(**d)


@dataclass
class AblationResult:
    """组件消融结果。"""
    full_system: dict
    no_hysteresis: dict = field(default_factory=dict)
    no_pinned: dict = field(default_factory=dict)
    no_warning_injection: dict = field(default_factory=dict)
    no_lignification: dict = field(default_factory=dict)
    no_decay: dict = field(default_factory=dict)
    no_rays: dict = field(default_factory=dict)


# ===========================================================================
# H1. Ring Oscillation Rate — Stability
# ===========================================================================
def ring_oscillation_rate(oplog: OpLog, window: int = 100) -> float:
    """window 内一个 cell 被 promote 后又被 demote 的比例。

    osc = |{cell: 既 promote 又 demote}| / |{cell: 被 promote}|
    健康范围: < 0.10。> 0.30 表示 hysteresis 失效。
    """
    all_entries = oplog.get_entries()
    # 取最近 window 条
    recent = all_entries[-window:] if window > 0 else all_entries

    promoted: set[str] = set()
    demoted_after_promote: set[str] = set()

    for entry in recent:
        if entry.op == OpEnum.PROMOTE.value:
            cell_id = entry.payload.get("cell_id", "")
            if cell_id:
                promoted.add(cell_id)
        elif entry.op == OpEnum.DEMOTE.value:
            cell_id = entry.payload.get("cell_id", "")
            if cell_id and cell_id in promoted:
                demoted_after_promote.add(cell_id)

    if not promoted:
        return 0.0
    return len(demoted_after_promote) / len(promoted)


# ===========================================================================
# H2. Context Retention Score — Context Drift
# ===========================================================================
def context_retention_score(
    injection_log: List[dict],
    horizon: int = 10,
) -> float:
    """被标记为关键的 cell (L3/L4 或 user_directive) 在 horizon 步内
    仍出现在 prompt 中的概率。

    injection_log: [{"step": t, "cell_ids": [...], "key_cell_ids": [...]}]
    """
    if not injection_log:
        return 1.0  # 无数据 → 默认满分 (static_outer 场景)

    retention_scores: List[float] = []
    for i, entry in enumerate(injection_log):
        key_cells = set(entry.get("key_cell_ids", []))
        if not key_cells:
            continue
        # 查看 horizon 步后是否仍存在
        future_steps = injection_log[i + 1 : i + 1 + horizon]
        if not future_steps:
            # 没有未来步可检查 → 不计入
            continue
        # 检查 key_cells 是否出现在未来任一步
        retained = 0
        for key_cell in key_cells:
            for future in future_steps:
                if key_cell in set(future.get("cell_ids", [])):
                    retained += 1
                    break
        retention_scores.append(retained / len(key_cells))

    if not retention_scores:
        return 1.0
    return sum(retention_scores) / len(retention_scores)


# ===========================================================================
# H3. Control Lag
# ===========================================================================
def control_lag(
    quarantine_ops: List[dict],
    warning_injections: List[dict],
) -> float:
    """从一个 cell 被 quarantine 到对应 warning 注入下一步 prompt
    的平均 step 距离。

    quarantine_ops: [{"cell_id": ..., "step": ...}]
    warning_injections: [{"cell_id": ..., "step": ...}]

    Tree 理论值 = 1.0 (after_step quarantine, 下一次 before_step 即注入)
    """
    if not quarantine_ops:
        return 0.0

    lags: List[int] = []
    for q in quarantine_ops:
        cell_id = q.get("cell_id", "")
        q_step = q.get("step", 0)
        # 找到对应 cell 的第一个 warning injection (step > q_step)
        for w in warning_injections:
            if w.get("cell_id") == cell_id and w.get("step", 0) > q_step:
                lags.append(w["step"] - q_step)
                break
        # 如果没找到,说明 warning 未被注入 → lag = infinity (用大值表示)
        if not any(w.get("cell_id") == cell_id for w in warning_injections):
            lags.append(999)

    return sum(lags) / len(lags) if lags else 0.0


# ===========================================================================
# H4. Entropy Release Per Episode
# ===========================================================================
def entropy_release_per_episode(
    compressed_count: int,
    quarantined_count: int,
    decayed_count: int,
    weights: Optional[dict] = None,
) -> float:
    """本 episode 通过算符调用释放的"熵量"。"""
    w = weights or {"compressed": 1.0, "quarantined": 2.0, "decayed": 0.5}
    return (
        w.get("compressed", 1.0) * compressed_count
        + w.get("quarantined", 2.0) * quarantined_count
        + w.get("decayed", 0.5) * decayed_count
    )


# ===========================================================================
# H5. Operator Call Distribution
# ===========================================================================
def op_count_distribution(oplog: OpLog, episode_id: Optional[str] = None) -> dict[str, int]:
    """每种 Self-Evolution Operator 的调用次数 (5 算符)。"""
    return oplog.count_by_op_type(episode_id=episode_id)


def promote_reason_distribution(oplog: OpLog, episode_id: Optional[str] = None) -> dict[str, int]:
    """Promote 算符的 reason 维度分解。"""
    return oplog.count_promotes_by_reason(episode_id=episode_id)


# ===========================================================================
# H6. HV/MV Ratio
# ===========================================================================
def hv_mv_ratio(results_grid: dict) -> float:
    """Harness Variance / Model Variance 比。

    输入: {"inner1": {"outer1": rate, "outer2": rate, ...}, ...}
    输出: HV/MV 数值。
    HV = 方差 over (不同 outer, 固定 inner 的均值)
    MV = 方差 over (不同 inner, 固定 outer 的均值)
    """
    if not results_grid:
        return 0.0

    inners = list(results_grid.keys())
    outers = set()
    for inner_results in results_grid.values():
        outers.update(inner_results.keys())
    outers = sorted(outers)

    if len(inners) < 2 or len(outers) < 2:
        return 0.0

    # HV: 固定 inner, 变化 outer → 取每个 inner 的 outer 方差均值
    hv_values: List[float] = []
    for inner in inners:
        rates = [results_grid[inner].get(o, 0.0) for o in outers]
        if len(rates) > 1:
            mean = sum(rates) / len(rates)
            var = sum((r - mean) ** 2 for r in rates) / len(rates)
            hv_values.append(var)
    hv = sum(hv_values) / len(hv_values) if hv_values else 0.0

    # MV: 固定 outer, 变化 inner → 取每个 outer 的 inner 方差均值
    mv_values: List[float] = []
    for outer in outers:
        rates = [results_grid[i].get(outer, 0.0) for i in inners]
        if len(rates) > 1:
            mean = sum(rates) / len(rates)
            var = sum((r - mean) ** 2 for r in rates) / len(rates)
            mv_values.append(var)
    mv = sum(mv_values) / len(mv_values) if mv_values else 0.0

    if mv == 0:
        return float('inf') if hv > 0 else 0.0
    return hv / mv


# ===========================================================================
# E1. Resolve Rate
# ===========================================================================
def resolve_rate(results: List[TaskResult], window: Optional[int] = None) -> float:
    """resolve rate = resolved / total。"""
    if window:
        results = results[-window:]
    if not results:
        return 0.0
    return sum(1 for r in results if r.resolved) / len(results)


# ===========================================================================
# E2. Cumulative Resolve Curve
# ===========================================================================
def cumulative_resolve_curve(results: List[TaskResult]) -> List[float]:
    """累积 resolve rate 曲线。"""
    curve: List[float] = []
    resolved = 0
    for i, r in enumerate(results):
        resolved += int(r.resolved)
        curve.append(resolved / (i + 1))
    return curve


# ===========================================================================
# E3. Relative Improvement
# ===========================================================================
def relative_improvement(target_rate: float, baseline_rate: float) -> float:
    """相对提升百分比。"""
    if baseline_rate == 0:
        return float('inf')
    return (target_rate - baseline_rate) / baseline_rate * 100


# ===========================================================================
# S1. Ray Connectivity Rate
# ===========================================================================
def ray_connectivity_rate(tree_store: TreeStore) -> float:
    """连通率 = (total - orphan) / total。健康 > 0.8。"""
    total = tree_store.sqlite.count_cells(status="active")
    if total == 0:
        return 1.0
    orphans = len(tree_store.find_orphans())
    return (total - orphans) / total


# ===========================================================================
# S2. Active/Dead Ratio
# ===========================================================================
def active_dead_ratio(tree_store: TreeStore) -> float:
    """活/死比。健康 > 2.0。"""
    active = tree_store.sqlite.count_cells(status="active")
    dead = (
        tree_store.sqlite.count_cells(status="quarantined")
        + tree_store.sqlite.count_cells(status="superseded")
        + tree_store.sqlite.count_cells(status="archived")
    )
    if dead == 0:
        return float('inf')
    return active / dead


# ===========================================================================
# S3. Lignification Compression Ratio
# ===========================================================================
def lignification_compression(oplog: OpLog) -> float:
    """合并压缩比 = merged_source_count / merged_target_count。"""
    merges = oplog.get_entries(op_filter=OpEnum.MERGE.value)
    total_sources = sum(len(m.payload.get("source_ids", [])) for m in merges)
    total_targets = len(merges)
    return total_sources / total_targets if total_targets > 0 else 0.0


# ===========================================================================
# S4. Ring Distribution
# ===========================================================================
def ring_distribution(tree_store: TreeStore) -> dict[str, int]:
    """各 ring 的 active cell 数。健康呈金字塔形。"""
    return {
        ring: tree_store.sqlite.count_cells(ring=ring, status="active")
        for ring in ["L0", "L1", "L2", "L3", "L4"]
    }


# ===========================================================================
# S5. Centrality Distribution (Gini)
# ===========================================================================
def centrality_gini(tree_store: TreeStore) -> float:
    """入度中心性 Gini 系数。健康 0.3~0.6。

    Gini = (sum of |xi - xj|) / (2 * n * mean)
    """
    cells = tree_store.sqlite.list_active()
    if not cells:
        return 0.0

    in_degrees = [tree_store.get_in_degree(c.id) for c in cells]
    n = len(in_degrees)
    total = sum(in_degrees)

    if total == 0:
        return 0.0  # 所有入度都为 0

    mean = total / n
    # Gini = sum(|xi - xj|) / (2 * n^2 * mean)
    sum_diffs = sum(abs(in_degrees[i] - in_degrees[j])
                    for i in range(n) for j in range(n))
    return sum_diffs / (2 * n * n * mean) if mean > 0 else 0.0


# ===========================================================================
# C1. Token Usage Per Episode
# ===========================================================================
def token_per_episode(results: List[TaskResult]) -> List[int]:
    """每 episode 的 token 消耗列表。"""
    return [r.token_usage for r in results]


# ===========================================================================
# C2. Outer-Harness Overhead
# ===========================================================================
def outer_overhead(
    total_tokens: int,
    outer_tokens: int,
) -> float:
    """outer harness 维护开销占比 = outer_tokens / total_tokens。"""
    if total_tokens == 0:
        return 0.0
    return outer_tokens / total_tokens


# ===========================================================================
# C3. Pareto Efficiency
# ===========================================================================
def pareto_front(
    conditions: Dict[str, Tuple[float, float]],
) -> List[str]:
    """{condition: (resolve_rate, avg_token_cost)} → Pareto 前沿。

    非支配解: 不存在另一个解 resolve_rate 更高且 token_cost 更低。
    """
    if not conditions:
        return []

    names = list(conditions.keys())
    frontier: List[str] = []

    for name in names:
        rate_i, cost_i = conditions[name]
        dominated = False
        for other in names:
            if other == name:
                continue
            rate_j, cost_j = conditions[other]
            # other 支配 name: rate 更高且 cost 更低 (至少一个严格)
            if rate_j >= rate_i and cost_j <= cost_i:
                if rate_j > rate_i or cost_j < cost_i:
                    dominated = True
                    break
        if not dominated:
            frontier.append(name)

    return frontier


# ===========================================================================
# 快照采集
# ===========================================================================
def take_snapshot(
    tree_store: TreeStore,
    oplog: OpLog,
    episode_index: int,
    timestamp: str,
    resolved: bool,
    cumulative_rate: float,
    token_usage: int = 0,
    duration_seconds: float = 0.0,
    entropy_released: float = 0.0,
    injection_log: Optional[List[dict]] = None,
    quarantine_ops: Optional[List[dict]] = None,
    warning_injections: Optional[List[dict]] = None,
    outer_tokens: int = 0,
) -> EpisodeSnapshot:
    """采集一次 episode 快照。"""
    dist = ring_distribution(tree_store)
    total_active = sum(dist.values())
    stats = tree_store.stats()
    orphans = len(tree_store.find_orphans())

    # H1
    osc_rate = ring_oscillation_rate(oplog)
    # H2
    if injection_log:
        retention = context_retention_score(injection_log)
    else:
        retention = 1.0
    # H3
    if quarantine_ops and warning_injections:
        lag = control_lag(quarantine_ops, warning_injections)
    else:
        lag = 0.0
    # H5
    op_counts = op_count_distribution(oplog)

    return EpisodeSnapshot(
        episode_index=episode_index,
        timestamp=timestamp,
        resolved=resolved,
        cumulative_resolve_rate=cumulative_rate,
        ring_oscillation_count=int(osc_rate * total_active) if total_active > 0 else 0,
        context_retention_score=retention,
        control_lag_steps=lag,
        entropy_released=entropy_released,
        op_counts=op_counts,
        total_active_cells=total_active,
        ring_distribution=dist,
        total_active_rays=stats.get("active_rays", 0),
        orphan_count=orphans,
        token_usage=token_usage,
        outer_overhead_ratio=outer_overhead(token_usage, outer_tokens) if token_usage > 0 else 0.0,
        duration_seconds=duration_seconds,
    )
