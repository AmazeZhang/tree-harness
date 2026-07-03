"""OuterHarness — Tree Harness 系统对外的唯一入口。

把常规 coding agent harness 包成结构化自演化的外层 harness,
通过三个 hook 完成 harness 级别的自演化:
1. before_step: 动态组装 inner harness 的输入 context (只读)
2. after_step: 蒸馏新 cell + 抽样验证 + 生成纠正信号
3. after_episode: 批量演化 (promotion / decay / consolidation)

对应 spec: docs/specs/outer_harness.md
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Protocol

from tree_harness.core.cell_model import Cell
from tree_harness.core.oplog import OpLog
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem
from tree_harness.modules.cambium_engine import CambiumEngine
from tree_harness.modules.context_injector import (
    ContextInjector, InjectorConfig, RetrievedContext, WarningEntry,
)
from tree_harness.adapters.trajectory_adapter import TrajectoryAdapter


# ---------------------------------------------------------------------------
# 数据类型 (outer_harness.md 接口定义)
# ---------------------------------------------------------------------------
@dataclass
class Task:
    task_id: str
    description: str
    repo_path: str
    metadata: dict = field(default_factory=dict)


@dataclass
class StepObservation:
    action: dict
    result: dict
    is_terminal: bool = False
    outcome: Optional[str] = None
    raw_output: Any = None


@dataclass
class InnerCapabilities:
    supports_pin_marker: bool = False
    supports_warning_marker: bool = False
    history_window_tokens: int = 4000
    has_internal_compaction: bool = False


class InnerHarnessProtocol(Protocol):
    def step(self, state: Any) -> StepObservation: ...
    def is_terminal(self, state: Any) -> bool: ...
    def reset(self, task: Task) -> Any: ...
    def capabilities(self) -> InnerCapabilities: ...


@dataclass
class ContextBlock:
    pinned_text: str
    relevant_text: str
    warnings: List[str]
    injected_cell_ids: List[str]
    token_count: int
    budget_used: dict
    pin_open_tag: str = "<|PINNED_DO_NOT_COMPACT|>"
    pin_close_tag: str = "<|/PINNED|>"
    warning_open_tag: str = "<|WARNING_DO_NOT_COMPACT|>"
    warning_close_tag: str = "<|/WARNING|>"


@dataclass
class StepRecord:
    episode_id: str
    step_index: int
    state_before: dict
    action: dict
    observation: dict
    cells_referenced: List[str]


@dataclass
class StepReport:
    new_cells: List[str]
    quarantined_cells: List[str]
    warnings_for_next_step: List[str]


@dataclass
class EpisodeRecord:
    episode_id: str
    task: Task
    outcome: str
    steps: List[StepRecord]
    duration_seconds: float
    token_usage: int


@dataclass
class EpisodeReport:
    promoted: List[tuple] = field(default_factory=list)    # (cell_id, from, to)
    demoted: List[tuple] = field(default_factory=list)
    decayed_below_threshold: List[str] = field(default_factory=list)
    new_cells_count: int = 0
    compressed_count: int = 0
    quarantined_count: int = 0
    decayed_count: int = 0
    op_counts: dict = field(default_factory=dict)           # 5 算符聚合
    raw_op_counts: dict = field(default_factory=dict)       # 底层 op_type 明细 (P-fix: 避免聚合混淆)
    entropy_released: float = 0.0


@dataclass
class OuterHarnessConfig:
    total_context_tokens: int = 4000
    pinned_ratio: float = 0.30
    relevant_ratio: float = 0.50
    warnings_ratio: float = 0.20
    enable_inline_warnings: bool = True
    max_warnings_per_step: int = 3
    decay_per_episode: bool = True
    lignification_per_episode: bool = True
    maintenance_funnel_per_episode: bool = True
    funnel_sample_size: int = 10
    high_ring_sample_size: int = 5       # P1-1: 每 episode 抽检 L3/L4 cell 数量
    dynamic_pinned_budget: bool = True  # P1-2: pinned budget 随 L3/L4 数量自适应
    pinned_tokens_per_cell: int = 120   # P1-2: 每 cell 估算 token 数
    pinned_budget_floor_ratio: float = 0.10  # pinned 下限
    pinned_budget_cap_ratio: float = 0.50    # pinned 上限
    entropy_weight_compressed: float = 1.0
    entropy_weight_quarantined: float = 2.0
    entropy_weight_decayed: float = 0.5


def _generate_episode_id() -> str:
    return f"ep-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# OuterHarness
# ---------------------------------------------------------------------------
class OuterHarness:
    """结构化自演化外层 harness — 三 hook 编排所有 Tree 模块。"""

    def __init__(
        self,
        tree_store: TreeStore,
        context_injector: ContextInjector,
        trajectory_adapter: TrajectoryAdapter,
        cambium: CambiumEngine,
        energy_system: EnergySystem,
        oplog: OpLog,
        config: OuterHarnessConfig,
        decay_sentinel=None,    # Phase 3, 可选
        lignification=None,     # Phase 4, 可选
    ):
        self.tree_store = tree_store
        self.context_injector = context_injector
        self.trajectory_adapter = trajectory_adapter
        self.cambium = cambium
        self.energy_system = energy_system
        self.oplog = oplog
        self.config = config
        self.decay_sentinel = decay_sentinel
        self.lignification = lignification

        # episode-local 状态
        self._pending_warnings: Dict[str, List[str]] = {}
        self._neighbor_warning_queue: Dict[str, List[str]] = {}
        self._injected_cell_ids: Dict[str, List[str]] = {}
        self._episode_new_cells_count: Dict[str, int] = {}
        self._episode_quarantine_count: Dict[str, int] = {}
        self._episode_decayed_count: Dict[str, int] = {}
        self._episode_compressed_count: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Hook 1: before_step (只读)
    # ------------------------------------------------------------------
    def before_step(
        self, task: Task, step_index: int, episode_id: str,
    ) -> ContextBlock:
        # 设置 adapter 上下文
        self.trajectory_adapter.set_context(task.task_id, task.repo_path)

        # 1. 拉取 warnings
        warnings_entries = self._pending_warnings.pop(episode_id, [])
        neighbor_entries = self._neighbor_warning_queue.pop(episode_id, [])
        all_warning_texts = warnings_entries + neighbor_entries

        # 2. Pinned: L3/L4 无条件注入 (提前查,用于动态 budget)
        pinned_cells = self.tree_store.list_by_ring(["L3", "L4"], status="active")

        # 2b. P1-2: 动态 Budget 分配
        total = self.config.total_context_tokens
        warnings_budget = int(total * self.config.warnings_ratio)
        if self.config.dynamic_pinned_budget:
            estimated = int(
                len(pinned_cells) * self.config.pinned_tokens_per_cell * 1.2
            )
            pinned_budget = min(
                max(estimated, int(total * self.config.pinned_budget_floor_ratio)),
                int(total * self.config.pinned_budget_cap_ratio),
            )
        else:
            pinned_budget = int(total * self.config.pinned_ratio)
        relevant_budget = max(total - pinned_budget - warnings_budget, 0)

        # 3. Pinned 注入
        pinned_text = self.context_injector.format_pinned(
            pinned_cells, budget=pinned_budget,
        )

        # 4. Relevant: L0/L1/L2 按相似度
        relevant_ctx = self.context_injector.retrieve(
            task.description, task.repo_path,
            ring_filter=["L0", "L1", "L2"],
            token_budget=relevant_budget,
        )

        # 5. Warnings
        warning_entries = [
            WarningEntry(
                cell_id="", text=t, is_direct=True, ray_weight=1.0, recency=0,
            )
            for t in all_warning_texts
        ]
        warnings_text = self.context_injector.format_warnings(
            warning_entries, budget=warnings_budget,
        )

        # 6. 组装
        injected_ids = [c.id for c in pinned_cells] + relevant_ctx.cells
        self._injected_cell_ids.setdefault(episode_id, []).extend(injected_ids)

        return ContextBlock(
            pinned_text=pinned_text,
            relevant_text=relevant_ctx.formatted_text,
            warnings=all_warning_texts,
            injected_cell_ids=injected_ids,
            token_count=(
                len(pinned_text.split()) + len(relevant_ctx.formatted_text.split())
                + len(warnings_text.split())
            ),
            budget_used={
                "pinned": len(pinned_text.split()),
                "relevant": relevant_ctx.token_count,
                "warnings": len(warnings_text.split()),
            },
        )

    # ------------------------------------------------------------------
    # Hook 2: after_step
    # ------------------------------------------------------------------
    def after_step(self, record: StepRecord) -> StepReport:
        new_cells: List[str] = []
        quarantined: List[str] = []
        next_warnings: List[str] = []

        # 1. TrajectoryAdapter 翻译
        step_obs = self.trajectory_adapter.convert_step(record)

        # 2. Cambium 蒸馏
        if self.cambium.should_crystallize(step_obs):
            crystals = self.cambium.crystallize_step(step_obs)
            new_cells.extend(c.id for c in crystals)

        # 记录 new_cells 计数
        ep_id = record.episode_id
        self._episode_new_cells_count[ep_id] = (
            self._episode_new_cells_count.get(ep_id, 0) + len(new_cells)
        )

        # 3. 抽样 funnel verification (Phase 3, 可选)
        if self.decay_sentinel is not None and self.config.maintenance_funnel_per_episode:
            candidates = self.energy_system.get_decay_candidates(
                limit=self.config.funnel_sample_size
            )
            verdicts = self.decay_sentinel.funnel_verify(candidates, episode_id=ep_id)
            for cell_id, v in verdicts.items():
                if v.result == "decayed":
                    cell = self.tree_store.get_cell(cell_id)
                    if cell is None:
                        continue
                    self.tree_store.quarantine(
                        cell_id, reason=v.reason, episode_id=ep_id,
                    )
                    quarantined.append(cell_id)
                    if self.config.enable_inline_warnings:
                        next_warnings.append(
                            self.trajectory_adapter.format_quarantine_warning(
                                cell, v.evidence or v.reason, v.verifier_name,
                            )
                        )
                elif v.result == "uncertain":
                    # Sentinel 已内部完成 decay_one + mark_for_review
                    # 只需跟踪计数
                    self._episode_decayed_count[ep_id] = (
                        self._episode_decayed_count.get(ep_id, 0) + 1
                    )

        # 4. 写入 pending warnings
        self._pending_warnings.setdefault(ep_id, []).extend(
            next_warnings[: self.config.max_warnings_per_step]
        )
        self._episode_quarantine_count[ep_id] = (
            self._episode_quarantine_count.get(ep_id, 0) + len(quarantined)
        )

        return StepReport(
            new_cells=new_cells,
            quarantined_cells=quarantined,
            warnings_for_next_step=next_warnings,
        )

    # ------------------------------------------------------------------
    # Hook 3: after_episode
    # ------------------------------------------------------------------
    def after_episode(self, record: EpisodeRecord) -> EpisodeReport:
        ep_id = record.episode_id

        # 1. 注入回流: pass 时给 injected cell 喂 reference
        if record.outcome == "pass":
            all_injected = self._injected_cell_ids.get(ep_id, [])
            for cell_id in set(all_injected):
                self.energy_system.reference(cell_id, ep_id)

        # 2. 全局 decay tick
        if self.config.decay_per_episode:
            self.energy_system.decay_all(ep_id)
            self.energy_system.update_all_maturity(ep_id)

        # 2.5 推进 episode 计数 (idle 检测用,在 decay 之后、lignification 之前)
        self.energy_system.advance_episode()

        # 2.6 P1-1: 高 ring 抽检 — 随机抽 L3/L4 cell 做 funnel_verify
        # (不依赖 energy threshold,防止 L3/L4 不死)
        if self.decay_sentinel is not None and self.config.high_ring_sample_size > 0:
            high_ring_ids = self.decay_sentinel.sample_high_ring_cells(
                self.config.high_ring_sample_size,
            )
            if high_ring_ids:
                verdicts = self.decay_sentinel.funnel_verify(
                    high_ring_ids, episode_id=ep_id,
                )
                for cell_id, v in verdicts.items():
                    if v.result == "decayed":
                        cell = self.tree_store.get_cell(cell_id)
                        if cell is None:
                            continue
                        self.tree_store.quarantine(
                            cell_id, reason=v.reason, episode_id=ep_id,
                        )
                        self._episode_quarantine_count[ep_id] = (
                            self._episode_quarantine_count.get(ep_id, 0) + 1
                        )
                    elif v.result == "uncertain":
                        self._episode_decayed_count[ep_id] = (
                            self._episode_decayed_count.get(ep_id, 0) + 1
                        )

        # 3. 木质化 (Phase 4, 可选)
        promoted, demoted = [], []
        if self.lignification is not None and self.config.lignification_per_episode:
            result = self.lignification.run_maintenance_cycle(ep_id)
            promoted = result.promoted
            demoted = result.demoted
            merged_n = len(result.merged) if hasattr(result, "merged") else 0
            self._episode_compressed_count[ep_id] = (
                self._episode_compressed_count.get(ep_id, 0) + merged_n
            )

        # 4. 量化熵释放
        compressed_n = self._episode_compressed_count.get(ep_id, 0)
        quarantined_n = self._episode_quarantine_count.get(ep_id, 0)
        decayed_n = self._episode_decayed_count.get(ep_id, 0)
        new_cells_n = self._episode_new_cells_count.get(ep_id, 0)
        entropy = self._compute_entropy_released(
            compressed_n, quarantined_n, decayed_n,
        )

        # 5. OpLog 聚合
        op_counts = self.oplog.count_by_op_type(episode_id=ep_id)
        raw_op_counts = self.oplog.count_by_raw_op(episode_id=ep_id)

        # 6. 清理 episode-local 状态
        self._cleanup_episode_state(ep_id)

        return EpisodeReport(
            promoted=promoted,
            demoted=demoted,
            new_cells_count=new_cells_n,
            compressed_count=compressed_n,
            quarantined_count=quarantined_n,
            decayed_count=decayed_n,
            op_counts=op_counts,
            raw_op_counts=raw_op_counts,
            entropy_released=entropy,
        )

    def _compute_entropy_released(
        self, compressed_n: int, quarantined_n: int, decayed_n: int,
    ) -> float:
        c = self.config
        return (
            c.entropy_weight_compressed * compressed_n
            + c.entropy_weight_quarantined * quarantined_n
            + c.entropy_weight_decayed * decayed_n
        )

    def _cleanup_episode_state(self, episode_id: str) -> None:
        self._pending_warnings.pop(episode_id, None)
        self._neighbor_warning_queue.pop(episode_id, None)
        self._injected_cell_ids.pop(episode_id, None)
        self._episode_new_cells_count.pop(episode_id, None)
        self._episode_quarantine_count.pop(episode_id, None)
        self._episode_decayed_count.pop(episode_id, None)
        self._episode_compressed_count.pop(episode_id, None)

    # ------------------------------------------------------------------
    # Runner 支撑: serialize / deserialize / snapshot
    # ------------------------------------------------------------------
    def serialize(self) -> dict:
        """导出可 checkpoint 的状态 (仅元数据,实际数据在 SQLite/Kuzu)。"""
        return {
            "type": "tree_outer",
            "total_cells": self.tree_store.sqlite.count_cells(),
            "active_cells": self.tree_store.sqlite.count_cells(status="active"),
        }

    def deserialize(self, state: dict) -> None:
        """从 checkpoint 恢复 (SQLite/Kuzu 已持久化,此处仅恢复 episode-local 状态)。"""
        self._pending_warnings.clear()
        self._neighbor_warning_queue.clear()
        self._injected_cell_ids.clear()
        self._episode_new_cells_count.clear()
        self._episode_quarantine_count.clear()
        self._episode_decayed_count.clear()
        self._episode_compressed_count.clear()

    def snapshot_ring_distribution(self) -> dict:
        """episode 末调用,返回当前 ring 分布。"""
        return {
            ring: self.tree_store.sqlite.count_cells(ring=ring, status="active")
            for ring in ["L0", "L1", "L2", "L3", "L4"]
        }

    def reset(self) -> None:
        """trial 起点: 清空所有 cell + ray + episode-local 状态。"""
        # 清空存储
        self.tree_store.clear()
        # 清空 episode-local 状态
        self._pending_warnings.clear()
        self._neighbor_warning_queue.clear()
        self._injected_cell_ids.clear()
        self._episode_new_cells_count.clear()
        self._episode_quarantine_count.clear()
        self._episode_decayed_count.clear()
        self._episode_compressed_count.clear()

    # ------------------------------------------------------------------
    # wrap: 包裹 inner harness
    # ------------------------------------------------------------------
    def wrap(self, inner: InnerHarnessProtocol):
        outer = self

        class _Wrapped:
            def run_episode(self, task: Task) -> tuple:
                state = inner.reset(task)
                steps: List[StepRecord] = []
                step_index = 0
                episode_id = _generate_episode_id()

                while not inner.is_terminal(state):
                    # Hook 1
                    ctx = outer.before_step(task, step_index, episode_id)

                    # 注入 context 到 inner state (mini-swe-agent 等 real inner 需要)
                    if hasattr(state, "augment"):
                        state = state.augment(ctx)

                    # Inner step
                    obs = inner.step(state)

                    # 构造 record
                    record = StepRecord(
                        episode_id=episode_id,
                        step_index=step_index,
                        state_before={},
                        action=obs.action,
                        observation=obs.result,
                        cells_referenced=ctx.injected_cell_ids,
                    )

                    # Hook 2
                    outer.after_step(record)
                    steps.append(record)

                    state = state.advance(obs) if hasattr(state, "advance") else state
                    if obs.is_terminal:
                        break
                    step_index += 1

                # Hook 3
                outcome = getattr(state, "outcome", None) or "pass"
                ep_record = EpisodeRecord(
                    episode_id=episode_id,
                    task=task,
                    outcome=outcome,
                    steps=steps,
                    duration_seconds=0.0,
                    token_usage=0,
                )
                ep_report = outer.after_episode(ep_record)
                return ep_record, ep_report

        return _Wrapped()
