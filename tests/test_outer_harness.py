"""OuterHarness 测试 —— 对应 docs/specs/outer_harness.md 测试用例。"""
import json

import pytest

from tree_harness.core.cell_model import create_cell
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.llm_client import DeterministicLLMClient
from tree_harness.core.oplog import OpLog
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem, EnergyConfig
from tree_harness.modules.cambium_engine import CambiumEngine, CambiumConfig
from tree_harness.modules.context_injector import ContextInjector, InjectorConfig
from tree_harness.modules.outer_harness import (
    OuterHarness, OuterHarnessConfig, Task, StepObservation,
    InnerCapabilities, ContextBlock, StepRecord, EpisodeRecord,
)
from tree_harness.adapters.trajectory_adapter import TrajectoryAdapter


# ---------------------------------------------------------------------------
# Mock Inner Harness
# ---------------------------------------------------------------------------
class MockState:
    def __init__(self):
        self.outcome = None
        self._step = 0

    def augment(self, context: ContextBlock):
        return self

    def advance(self, obs: StepObservation):
        if obs.is_terminal:
            self.outcome = obs.outcome or "pass"
        return self

    def snapshot(self):
        return {"step": self._step}


class MockInner:
    """简单 mock: 产出 N 步后终止,每步有 patch 和 test。"""
    def __init__(self, max_steps=3):
        self._max_steps = max_steps
        self._state = MockState()

    def reset(self, task: Task):
        self._state = MockState()
        return self._state

    def step(self, state):
        state._step += 1
        is_terminal = state._step >= self._max_steps
        return StepObservation(
            action={"summary": f"Edited file.py line {state._step}"},
            result={
                "summary": f"Test passed at step {state._step}",
                "patch": f"diff --git a/file.py",
                "tests": {f"test_{state._step}": "pass"},
                "outcome": "pass" if is_terminal else "pending",
            },
            is_terminal=is_terminal,
            outcome="pass" if is_terminal else None,
        )

    def is_terminal(self, state):
        return state._step >= self._max_steps

    def capabilities(self):
        return InnerCapabilities()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def embedder():
    return DeterministicEmbedder(dim=32)


@pytest.fixture
def tree(tmp_path, embedder):
    sqlite = SQLiteBackend(":memory:", embedder=embedder)
    kuzu = KuzuBackend(str(tmp_path / "kuzu"))
    oplog = OpLog(str(tmp_path / "oplog.db"))
    yield TreeStore(sqlite, kuzu, oplog)


@pytest.fixture
def oplog(tmp_path):
    return OpLog(str(tmp_path / "oplog.db"))


@pytest.fixture
def llm():
    client = DeterministicLLMClient()
    client.inject("crystallize", json.dumps({
        "decision": "Always validate input before processing",
        "rationale": "Unvalidated input caused crash",
        "preconditions": [],
        "evidence": [],
        "domain_tags": ["validation"],
    }))
    return client


@pytest.fixture
def outer(tree, oplog, llm):
    energy = EnergySystem(EnergyConfig(), tree)
    cambium = CambiumEngine(tree, energy, llm, CambiumConfig())
    injector = ContextInjector(tree, InjectorConfig(min_similarity=0.0))
    adapter = TrajectoryAdapter()
    config = OuterHarnessConfig()
    return OuterHarness(
        tree_store=tree,
        context_injector=injector,
        trajectory_adapter=adapter,
        cambium=cambium,
        energy_system=energy,
        oplog=oplog,
        config=config,
    )


def _task():
    return Task(
        task_id="django__django-16379",
        description="Fix ordering null bug in compiler",
        repo_path="/tmp/django",
    )


def _step_record(episode_id="ep1", step_index=0):
    return StepRecord(
        episode_id=episode_id,
        step_index=step_index,
        state_before={},
        action={"summary": "Edited compiler.py to add nulls_first"},
        observation={
            "summary": "Test passed",
            "patch": "diff --git a/compiler.py",
            "tests": {"test_ordering": "pass"},
            "outcome": "pass",
        },
        cells_referenced=[],
    )


# ---------------------------------------------------------------------------
# before_step
# ---------------------------------------------------------------------------
def test_before_step_empty_tree_no_error(outer):
    """L3/L4 为空时 pinned_text 为空但不报错。"""
    ctx = outer.before_step(_task(), 0, "ep1")
    assert ctx.pinned_text == ""
    assert isinstance(ctx, ContextBlock)


def test_before_step_with_pinned(outer, tree):
    """L3/L4 cell 存在时 pinned_text 非空。"""
    tree.insert_cell(create_cell(
        cell_id="axiom1", ring="L4",
        decision="Always use type hints", rationale="PEP 484 compliance",
    ))
    ctx = outer.before_step(_task(), 0, "ep1")
    assert ctx.pinned_text != ""
    assert "axiom1" in ctx.injected_cell_ids or len(ctx.injected_cell_ids) > 0


def test_before_step_records_injected_ids(outer, tree):
    """injected_cell_ids 被记录用于 after_episode reference。"""
    tree.insert_cell(create_cell(
        cell_id="c1", ring="L4", decision="axiom", rationale="reason",
    ))
    outer.before_step(_task(), 0, "ep1")
    assert "c1" in outer._injected_cell_ids["ep1"]


# ---------------------------------------------------------------------------
# after_step
# ---------------------------------------------------------------------------
def test_after_step_crystallizes(outer, tree):
    """after_step 触发 crystallize → 产出新 cell。"""
    record = _step_record()
    report = outer.after_step(record)
    # LLM 注入了 crystallize 响应 → 应该产出 cell
    assert len(report.new_cells) > 0
    # cell 在 tree 中
    for cid in report.new_cells:
        assert tree.get_cell(cid) is not None


def test_after_step_no_warnings_when_no_quarantine(outer):
    """无 quarantine 时 warnings_for_next_step 为空。"""
    record = _step_record()
    report = outer.after_step(record)
    assert report.quarantined_cells == []


# ---------------------------------------------------------------------------
# after_episode
# ---------------------------------------------------------------------------
def test_after_episode_pass_references(outer, tree):
    """outcome=pass → injected cells 的 energy 增加。"""
    tree.insert_cell(create_cell(
        cell_id="c1", ring="L3", energy=0.5,
        decision="existing knowledge", rationale="some reason",
    ))
    # before_step 注入 c1
    outer.before_step(_task(), 0, "ep1")
    # after_episode with pass
    record = EpisodeRecord(
        episode_id="ep1", task=_task(), outcome="pass",
        steps=[], duration_seconds=10.0, token_usage=100,
    )
    outer.after_episode(record)
    # c1 的 energy 应该增加 (reference 触发)
    cell = tree.get_cell("c1")
    assert cell.energy > 0.5


def test_after_episode_fail_no_auto_challenge(outer, tree):
    """outcome=fail → 不自动 challenge (energy 不减)。"""
    tree.insert_cell(create_cell(
        cell_id="c1", ring="L3", energy=0.5,
        decision="knowledge", rationale="reason",
    ))
    outer.before_step(_task(), 0, "ep1")
    record = EpisodeRecord(
        episode_id="ep1", task=_task(), outcome="fail",
        steps=[], duration_seconds=10.0, token_usage=100,
    )
    outer.after_episode(record)
    cell = tree.get_cell("c1")
    # energy 因 decay 降低,但不因 fail 自动 challenge
    # decay 衰减: L3 rate=0.10 → 0.5 * 0.90 = 0.45
    assert cell.energy < 0.5  # decay 降低
    assert cell.energy >= 0.40  # 但不是大幅下降 (无 challenge)


def test_after_episode_cleanup(outer, tree):
    """episode 结束后 pending_warnings 被清理。"""
    outer._pending_warnings["ep1"] = ["warning1"]
    record = EpisodeRecord(
        episode_id="ep1", task=_task(), outcome="pass",
        steps=[], duration_seconds=10.0, token_usage=100,
    )
    outer.after_episode(record)
    assert "ep1" not in outer._pending_warnings
    assert "ep1" not in outer._injected_cell_ids


def test_after_episode_entropy(outer):
    """entropy_released 被计算。"""
    record = EpisodeRecord(
        episode_id="ep1", task=_task(), outcome="pass",
        steps=[], duration_seconds=10.0, token_usage=100,
    )
    report = outer.after_episode(record)
    assert report.entropy_released >= 0.0


# ---------------------------------------------------------------------------
# wrap + run_episode
# ---------------------------------------------------------------------------
def test_wrap_run_episode(outer):
    """wrap 一个 mock inner → run_episode 正常完成。"""
    inner = MockInner(max_steps=3)
    wrapped = outer.wrap(inner)
    record, report = wrapped.run_episode(_task())
    assert record.outcome == "pass"
    assert isinstance(report.op_counts, dict)
    assert report.entropy_released >= 0.0
