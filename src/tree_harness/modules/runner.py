"""TreeHarnessRunner — 序贯实验的最外层装配器。

唯一职责: 组装 OuterHarness 与 InnerHarness、驱动 run_episode 循环、
负责日志/checkpoint。所有自演化逻辑都在 OuterHarness 的三 hook 内执行,
Runner 不直接调用 cambium / injector / decay 等内部模块。

对应 spec: docs/specs/runner.md
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal, Protocol

from tree_harness.core.oplog import OpLog
from tree_harness.store.tree_store import TreeStore
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.llm_client import LLMClient, DeterministicLLMClient
from tree_harness.modules.energy_system import EnergySystem, EnergyConfig
from tree_harness.modules.cambium_engine import CambiumEngine, CambiumConfig
from tree_harness.modules.context_injector import (
    ContextInjector, InjectorConfig, RetrievedContext, WarningEntry,
)
from tree_harness.modules.outer_harness import (
    OuterHarness, OuterHarnessConfig,
    Task, StepObservation, ContextBlock,
    EpisodeRecord, EpisodeReport,
    InnerHarnessProtocol, InnerCapabilities,
    StepRecord, StepReport,
)
from tree_harness.modules.verifiers import VerifierRegistry
from tree_harness.modules.decay_sentinel import DecaySentinel
from tree_harness.modules.lignification import LignificationScheduler, LignificationConfig
from tree_harness.modules.metrics import TaskResult
from tree_harness.adapters.trajectory_adapter import TrajectoryAdapter


# ===========================================================================
# 配置
# ===========================================================================
@dataclass
class RunnerConfig:
    """装配选择 + 运行参数。"""
    # 装配选择
    outer_kind: Literal["bare_inner", "static_outer", "freeform_outer", "tree_outer"] = "tree_outer"
    inner_kind: Literal["swe-agent", "openhands", "mini-swe-agent", "mock"] = "mock"

    # tree_outer 专用
    db_path: Optional[str] = None
    energy_config: Optional[EnergyConfig] = None
    cambium_config: Optional[CambiumConfig] = None
    injector_config: Optional[InjectorConfig] = None
    outer_harness_config: Optional[OuterHarnessConfig] = None
    lignification_config: Optional[LignificationConfig] = None

    # freeform_outer 专用
    freeform_rewriter_model: Optional[str] = None
    freeform_rewrite_budget: int = 5

    # 通用
    repo_path: str = "."
    agent_config: dict = field(default_factory=dict)
    llm_client: Optional[LLMClient] = None
    log_dir: str = "./logs"
    trial_id: int = 0

    # Embedder: "deterministic" (测试用) 或 "sentence-transformer" (真实语义)
    embedder_kind: str = "deterministic"
    embedder_model: str = "BAAI/bge-base-zh-v1.5"


# ===========================================================================
# Baseline Outer Harnesses
# ===========================================================================
class NoOpOuterHarness:
    """bare_inner: 不做任何 outer 包装,直接透传 inner。"""

    def wrap(self, inner: InnerHarnessProtocol):
        outer = self

        class _Wrapped:
            def run_episode(self, task: Task) -> tuple:
                state = inner.reset(task)
                steps: List[StepRecord] = []
                step_index = 0
                episode_id = f"ep-{step_index:04d}"

                while not inner.is_terminal(state):
                    obs = inner.step(state)
                    record = StepRecord(
                        episode_id=episode_id,
                        step_index=step_index,
                        state_before={},
                        action=obs.action,
                        observation=obs.result,
                        cells_referenced=[],
                    )
                    steps.append(record)
                    state = state.advance(obs) if hasattr(state, "advance") else state
                    if obs.is_terminal:
                        break
                    step_index += 1

                outcome = getattr(state, "outcome", None) or "pass"
                ep_record = EpisodeRecord(
                    episode_id=episode_id,
                    task=task,
                    outcome=outcome,
                    steps=steps,
                    duration_seconds=0.0,
                    token_usage=0,
                )
                ep_report = EpisodeReport()
                return ep_record, ep_report

        return _Wrapped()

    def serialize(self) -> dict:
        return {"type": "bare_inner"}

    def deserialize(self, state: dict) -> None:
        pass

    def snapshot_ring_distribution(self) -> dict:
        return {ring: 0 for ring in ["L0", "L1", "L2", "L3", "L4"]}

    def reset(self) -> None:
        pass


class StaticOuterHarness:
    """static_outer: 注入固定 system prompt,不做任何演化。"""

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self._total_context_tokens = 4000

    def wrap(self, inner: InnerHarnessProtocol):
        outer = self

        class _Wrapped:
            def run_episode(self, task: Task) -> tuple:
                state = inner.reset(task)
                steps: List[StepRecord] = []
                step_index = 0
                episode_id = f"ep-{step_index:04d}"

                while not inner.is_terminal(state):
                    # 注入固定 prompt
                    ctx = ContextBlock(
                        pinned_text=outer.system_prompt,
                        relevant_text="",
                        warnings=[],
                        injected_cell_ids=[],
                        token_count=len(outer.system_prompt.split()),
                        budget_used={"pinned": len(outer.system_prompt.split()),
                                     "relevant": 0, "warnings": 0},
                    )

                    obs = inner.step(state)
                    record = StepRecord(
                        episode_id=episode_id,
                        step_index=step_index,
                        state_before={},
                        action=obs.action,
                        observation=obs.result,
                        cells_referenced=[],
                    )
                    steps.append(record)
                    state = state.advance(obs) if hasattr(state, "advance") else state
                    if obs.is_terminal:
                        break
                    step_index += 1

                outcome = getattr(state, "outcome", None) or "pass"
                ep_record = EpisodeRecord(
                    episode_id=episode_id,
                    task=task,
                    outcome=outcome,
                    steps=steps,
                    duration_seconds=0.0,
                    token_usage=0,
                )
                ep_report = EpisodeReport()
                return ep_record, ep_report

        return _Wrapped()

    def serialize(self) -> dict:
        return {"type": "static_outer", "system_prompt": self.system_prompt}

    def deserialize(self, state: dict) -> None:
        self.system_prompt = state.get("system_prompt", "")

    def snapshot_ring_distribution(self) -> dict:
        return {ring: 0 for ring in ["L0", "L1", "L2", "L3", "L4"]}

    def reset(self) -> None:
        pass


class FreeformOuterHarness:
    """freeform_outer (SIA-style): LLM 在 episode 间改写 scaffold。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rewriter_model: Optional[str] = None,
        rewrite_budget: int = 5,
        initial_scaffold: str = "",
    ):
        self.llm_client = llm_client or DeterministicLLMClient()
        self.rewriter_model = rewriter_model
        self.rewrite_budget = rewrite_budget
        self.scaffold = initial_scaffold or (
            "You are a coding agent. Fix the issue described in the task. "
            "Write clean, tested code."
        )
        self._episode_count = 0
        self._recent_trajectories: List[dict] = []
        self._scaffold_changes: List[dict] = []
        self._current_rewritten_prompt: Optional[str] = None

    def wrap(self, inner: InnerHarnessProtocol):
        outer = self

        class _Wrapped:
            def run_episode(self, task: Task) -> tuple:
                state = inner.reset(task)
                steps: List[StepRecord] = []
                step_index = 0
                episode_id = f"ep-{outer._episode_count:04d}"

                while not inner.is_terminal(state):
                    obs = inner.step(state)
                    record = StepRecord(
                        episode_id=episode_id,
                        step_index=step_index,
                        state_before={},
                        action=obs.action,
                        observation=obs.result,
                        cells_referenced=[],
                    )
                    steps.append(record)
                    state = state.advance(obs) if hasattr(state, "advance") else state
                    if obs.is_terminal:
                        break
                    step_index += 1

                outcome = getattr(state, "outcome", None) or "pass"
                ep_record = EpisodeRecord(
                    episode_id=episode_id,
                    task=task,
                    outcome=outcome,
                    steps=steps,
                    duration_seconds=0.0,
                    token_usage=0,
                )

                # after_episode: 每 N episode 改写 scaffold
                outer._episode_count += 1
                outer._recent_trajectories.append({
                    "task_id": task.task_id,
                    "outcome": outcome,
                    "n_steps": len(steps),
                })
                if outer._episode_count % outer.rewrite_budget == 0:
                    old_scaffold = outer.scaffold
                    outer._rewrite_scaffold()
                    outer._scaffold_changes.append({
                        "episode": outer._episode_count,
                        "old": old_scaffold,
                        "new": outer.scaffold,
                    })

                ep_report = EpisodeReport()
                ep_report.entropy_released = 0.0
                return ep_record, ep_report

        return _Wrapped()

    def _rewrite_scaffold(self) -> None:
        """LLM 改写 scaffold (简化版: 追加最近 trajectory 摘要)。"""
        recent = self._recent_trajectories[-self.rewrite_budget:]
        summary = "; ".join(
            f"ep{r['task_id']}:{r['outcome']}" for r in recent
        )
        self.scaffold = (
            f"{self.scaffold}\n"
            f"[Feedback from recent episodes: {summary}]"
        )
        self._current_rewritten_prompt = self.scaffold

    def serialize(self) -> dict:
        return {
            "type": "freeform_outer",
            "scaffold": self.scaffold,
            "episode_count": self._episode_count,
            "scaffold_changes": self._scaffold_changes,
        }

    def deserialize(self, state: dict) -> None:
        self.scaffold = state.get("scaffold", self.scaffold)
        self._episode_count = state.get("episode_count", 0)
        self._scaffold_changes = state.get("scaffold_changes", [])

    def snapshot_ring_distribution(self) -> dict:
        return {ring: 0 for ring in ["L0", "L1", "L2", "L3", "L4"]}

    def reset(self) -> None:
        self._episode_count = 0
        self._recent_trajectories.clear()
        self._scaffold_changes.clear()
        self._current_rewritten_prompt = None
        self.scaffold = (
            "You are a coding agent. Fix the issue described in the task. "
            "Write clean, tested code."
        )

    @property
    def rewritten_prompt(self) -> Optional[str]:
        return self._current_rewritten_prompt


# ===========================================================================
# Runner
# ===========================================================================
class TreeHarnessRunner:
    """序贯实验最外层装配器。

    薄装配: 只持有 outer + inner + 日志器。
    所有自演化逻辑在 OuterHarness 三 hook 内执行。
    """

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.llm_client = config.llm_client or DeterministicLLMClient()

        # Embedder 选择
        if config.embedder_kind == "sentence-transformer":
            from tree_harness.core.embedding import SentenceTransformerEmbedder
            self.embedder = SentenceTransformerEmbedder(config.embedder_model)
        else:
            self.embedder = DeterministicEmbedder(dim=32)

        # 装配
        self.outer = self._build_outer(config)
        self.inner = self._build_inner(config)
        self.wrapped = self.outer.wrap(self.inner)

        # 状态
        self.episode_count = 0
        self._results: List[TaskResult] = []

        # 日志
        os.makedirs(config.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 装配逻辑
    # ------------------------------------------------------------------
    def _build_outer(self, config: RunnerConfig):
        if config.outer_kind == "bare_inner":
            return NoOpOuterHarness()

        if config.outer_kind == "static_outer":
            return StaticOuterHarness(
                system_prompt=config.agent_config.get("system_prompt", ""),
            )

        if config.outer_kind == "freeform_outer":
            return FreeformOuterHarness(
                llm_client=self.llm_client,
                rewriter_model=config.freeform_rewriter_model,
                rewrite_budget=config.freeform_rewrite_budget,
                initial_scaffold=config.agent_config.get("system_prompt", ""),
            )

        if config.outer_kind == "tree_outer":
            return self._build_tree_outer(config)

        raise ValueError(f"Unknown outer_kind: {config.outer_kind}")

    def _build_tree_outer(self, config: RunnerConfig) -> OuterHarness:
        """组装完整 Tree Harness (CambiumEngine + DecaySentinel + Lignification)。"""
        import tempfile
        import uuid as _uuid

        db_path = config.db_path or ":memory:"
        sqlite = SQLiteBackend(db_path, embedder=self.embedder)

        # KuzuDB 需要目录路径,每个实例使用独立临时目录避免冲突
        kuzu_path = config.db_path if config.db_path and config.db_path != ":memory:" else None
        if kuzu_path is None:
            kuzu_path = os.path.join(
                tempfile.gettempdir(),
                f"tree_kuzu_{_uuid.uuid4().hex[:8]}",
            )
        kuzu = KuzuBackend(kuzu_path)
        oplog = OpLog(":memory:")
        tree_store = TreeStore(sqlite, kuzu, oplog)

        energy = EnergySystem(
            config.energy_config or EnergyConfig(), tree_store,
        )
        cambium = CambiumEngine(
            tree_store, energy, self.llm_client,
            config.cambium_config or CambiumConfig(),
        )
        injector = ContextInjector(
            tree_store,
            config.injector_config or InjectorConfig(min_similarity=0.0),
        )
        adapter = TrajectoryAdapter()
        verifier_registry = VerifierRegistry(repo_path=config.repo_path)
        decay_sentinel = DecaySentinel(
            tree_store=tree_store, energy_system=energy,
            verifier_registry=verifier_registry, llm_client=self.llm_client,
            repo_path=config.repo_path,
        )
        lignification = LignificationScheduler(
            tree_store=tree_store, energy_system=energy,
            llm_client=self.llm_client, oplog=oplog,
            config=config.lignification_config or LignificationConfig(),
        )

        return OuterHarness(
            tree_store=tree_store,
            context_injector=injector,
            trajectory_adapter=adapter,
            cambium=cambium,
            energy_system=energy,
            oplog=oplog,
            config=config.outer_harness_config or OuterHarnessConfig(),
            decay_sentinel=decay_sentinel,
            lignification=lignification,
        )

    def _build_inner(self, config: RunnerConfig) -> InnerHarnessProtocol:
        if config.inner_kind == "mock":
            return config.agent_config.get("inner_factory", lambda: _MockInner())()
        # 真实 inner harness 尚未实现 — 返回 mock 并警告
        # SWE-agent / OpenHands / mini-swe-agent 需要外部依赖
        return _MockInner()

    # ------------------------------------------------------------------
    # 单 Episode
    # ------------------------------------------------------------------
    def run_episode(self, task: Task) -> TaskResult:
        """委托给 wrapped harness,自身只做日志与计数。"""
        ep_record, ep_report = self.wrapped.run_episode(task)

        result = self._flatten(ep_record, ep_report)
        self.episode_count += 1
        self._results.append(result)
        self._write_log(result)
        return result

    def _flatten(
        self, record: EpisodeRecord, report: EpisodeReport,
    ) -> TaskResult:
        """摊平 (EpisodeRecord, EpisodeReport) → TaskResult。"""
        resolved = record.outcome == "pass"
        ring_dist = self.outer.snapshot_ring_distribution()
        caps = self.inner.capabilities()

        result = TaskResult(
            task_id=record.task.task_id,
            repo=record.task.repo_path,
            condition=self.config.outer_kind,
            trial=self.config.trial_id,
            episode_index=self.episode_count,
            inner_kind=self.config.inner_kind,
            resolved=resolved,
            outcome=record.outcome,
            duration_seconds=record.duration_seconds,
            token_usage=record.token_usage,
            n_steps=len(record.steps),
            op_counts=report.op_counts,
            entropy_released=report.entropy_released,
            new_cells_count=report.new_cells_count,
            compressed_count=report.compressed_count,
            quarantined_count=report.quarantined_count,
            decayed_count=report.decayed_count,
            promoted=report.promoted,
            demoted=report.demoted,
            total_active_cells=sum(ring_dist.values()),
            ring_distribution=ring_dist,
            inner_supports_pin_marker=caps.supports_pin_marker,
            inner_supports_warning_marker=caps.supports_warning_marker,
        )

        # freeform_outer 专用
        if isinstance(self.outer, FreeformOuterHarness):
            result.rewritten_prompt = self.outer.rewritten_prompt

        return result

    # ------------------------------------------------------------------
    # 序贯实验
    # ------------------------------------------------------------------
    def run_sequential(self, tasks: List[Task]) -> List[TaskResult]:
        """按时间顺序依次执行 tasks,保留 outer 内部状态。"""
        records = []
        for task in tasks:
            record = self.run_episode(task)
            records.append(record)
            if self.episode_count % 10 == 0:
                self.checkpoint(self._auto_checkpoint_path())
        return records

    # ------------------------------------------------------------------
    # Trial 管理
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """trial 起点: 清空 outer 状态,inner 重新初始化。"""
        self.outer.reset()
        self.episode_count = 0
        self._results.clear()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def checkpoint(self, path: str) -> None:
        payload = {
            "episode_count": self.episode_count,
            "outer_state": self.outer.serialize(),
            "config": {
                "outer_kind": self.config.outer_kind,
                "inner_kind": self.config.inner_kind,
                "trial_id": self.config.trial_id,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def resume(self, path: str) -> None:
        with open(path) as f:
            payload = json.load(f)
        self.episode_count = payload["episode_count"]
        self.outer.deserialize(payload["outer_state"])

    def _auto_checkpoint_path(self) -> str:
        return os.path.join(
            self.config.log_dir,
            f"checkpoint_trial{self.config.trial_id}_ep{self.episode_count}.json",
        )

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _write_log(self, result: TaskResult) -> None:
        log_path = os.path.join(
            self.config.log_dir,
            f"episodes_trial{self.config.trial_id}.jsonl",
        )
        entry = asdict(result)
        # 附加 LLM token 累计统计
        llm = self.llm_client
        if hasattr(llm, "total_tokens"):
            entry["_llm_stats"] = {
                "total_calls": llm.call_count,
                "cache_hits": llm.cache_hit_count,
                "prompt_tokens": llm.prompt_tokens,
                "completion_tokens": llm.completion_tokens,
                "total_tokens": llm.total_tokens,
            }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @property
    def results(self) -> List[TaskResult]:
        return list(self._results)


# ===========================================================================
# Mock Inner (用于测试)
# ===========================================================================
class _MockState:
    def __init__(self, task, steps_plan):
        self.task = task
        self.steps_plan = steps_plan
        self.step_index = 0
        self.outcome = "pending"

    def advance(self, obs):
        self.step_index += 1
        if obs.is_terminal:
            self.outcome = obs.outcome or "pass"
        return self


class _MockInner:
    """Mock inner harness for testing — produces meaningful step sequences.

    默认 plan 产出有 patch_delta + test_results 的 step,触发 CambiumEngine crystallize。
    """

    def __init__(self, steps_plan=None):
        self._default_plan = steps_plan or [
            {
                "action": {"summary": "Modify model.py to add nulls_first=True"},
                "result": {
                    "summary": "Applied patch, test passed",
                    "patch": "diff --git a/model.py\n+nulls_first=True",
                    "tests": {"test_x": "pass"},
                    "outcome": "pending",
                    "duration": 1.0,
                    "tokens": 500,
                },
                "outcome": "pending",
            },
            {
                "action": {"summary": "Run test suite"},
                "result": {
                    "summary": "All tests passed",
                    "patch": "",
                    "tests": {"test_x": "pass", "test_y": "pass"},
                    "outcome": "pass",
                    "duration": 1.0,
                    "tokens": 300,
                },
                "outcome": "pass",
            },
        ]
        self._state = None

    def reset(self, task: Task):
        self._state = _MockState(task, self._default_plan)
        return self._state

    def step(self, state):
        idx = state.step_index
        plan = self._default_plan[idx] if idx < len(self._default_plan) else None
        if plan is None:
            return StepObservation(
                action={}, result={}, is_terminal=True, outcome="pass",
            )
        is_last = idx >= len(self._default_plan) - 1
        return StepObservation(
            action=plan.get("action", {}),
            result=plan.get("result", {}),
            is_terminal=is_last,
            outcome=plan.get("outcome", "pass" if is_last else "pending"),
        )

    def is_terminal(self, state):
        return state.step_index >= len(self._default_plan)

    def capabilities(self):
        return InnerCapabilities(
            supports_pin_marker=True,
            supports_warning_marker=True,
        )

    def serialize(self) -> dict:
        return {"type": "mock"}

    def deserialize(self, state: dict) -> None:
        pass
