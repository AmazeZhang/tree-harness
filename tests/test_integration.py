"""端到端集成测试 — 全模块串联,验证 5 算符完整生命周期。

模拟一个 inner harness 跑多个 episode,验证:
- before_step 注入 context (pinned + relevant + warnings)
- after_step crystallize 新 cell + connect ray + funnel verify
- after_episode promote / demote / decay / merge
- OpLog 记录完整
- TreeStore 双库一致性
"""
import os
import pytest

from tree_harness.core.cell_model import create_cell, Cell, Precondition, VerifyHint
from tree_harness.core.embedding import DeterministicEmbedder, embed_cell_text
from tree_harness.core.llm_client import DeterministicLLMClient
from tree_harness.core.oplog import OpLog, OpEnum
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem, EnergyConfig
from tree_harness.modules.cambium_engine import CambiumEngine, CambiumConfig
from tree_harness.modules.context_injector import ContextInjector, InjectorConfig
from tree_harness.modules.outer_harness import (
    OuterHarness, OuterHarnessConfig, Task, StepObservation,
    InnerHarnessProtocol, InnerCapabilities,
)
from tree_harness.modules.verifiers import VerifierRegistry
from tree_harness.modules.decay_sentinel import DecaySentinel
from tree_harness.modules.lignification import LignificationScheduler, LignificationConfig
from tree_harness.adapters.trajectory_adapter import TrajectoryAdapter


# ---------------------------------------------------------------------------
# Mock Inner Harness
# ---------------------------------------------------------------------------
class MockState:
    """模拟 inner harness 的 state 对象。"""
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


class MockInnerHarness:
    """模拟 coding agent harness — 产出有意义的 step 序列。"""

    def __init__(self, steps_plan):
        """steps_plan: List[dict] 每个 dict 描述一步的 action/observation。"""
        self.steps_plan = steps_plan
        self._state = None

    def reset(self, task):
        self._state = MockState(task, self.steps_plan)
        return self._state

    def step(self, state):
        idx = state.step_index
        plan = self.steps_plan[idx] if idx < len(self.steps_plan) else None
        if plan is None:
            return StepObservation(
                action={}, result={}, is_terminal=True, outcome="pass",
            )
        is_last = (idx >= len(self.steps_plan) - 1)
        return StepObservation(
            action=plan.get("action", {}),
            result=plan.get("result", {}),
            is_terminal=is_last,
            outcome=plan.get("outcome", "pass" if is_last else "pending"),
            raw_output=plan.get("raw_output"),
        )

    def is_terminal(self, state):
        return state.step_index >= len(self.steps_plan)

    def capabilities(self):
        return InnerCapabilities(
            supports_pin_marker=True,
            supports_warning_marker=True,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def embedder():
    return DeterministicEmbedder(dim=32)


@pytest.fixture
def llm():
    return DeterministicLLMClient()


@pytest.fixture
def full_setup(tmp_path, llm, embedder):
    """组装完整 Tree Harness 系统。"""
    # Storage
    sqlite = SQLiteBackend(":memory:", embedder=embedder)
    kuzu = KuzuBackend(str(tmp_path / "kuzu"))
    oplog = OpLog(str(tmp_path / "oplog.db"))
    tree = TreeStore(sqlite, kuzu, oplog)

    # Energy
    energy = EnergySystem(EnergyConfig(), tree)

    # Cambium
    cambium = CambiumEngine(tree, energy, llm, CambiumConfig())

    # Context Injector (降低 min_similarity 以适配 DeterministicEmbedder)
    injector = ContextInjector(tree, InjectorConfig(min_similarity=0.0))

    # Trajectory Adapter
    adapter = TrajectoryAdapter()

    # Verifiers + DecaySentinel
    registry = VerifierRegistry(repo_path=str(tmp_path))
    sentinel = DecaySentinel(
        tree_store=tree, energy_system=energy,
        verifier_registry=registry, llm_client=llm,
        repo_path=str(tmp_path),
    )

    # Lignification
    lignification = LignificationScheduler(
        tree_store=tree, energy_system=energy,
        llm_client=llm, oplog=oplog,
        config=LignificationConfig(),
    )

    # OuterHarness
    config = OuterHarnessConfig(
        total_context_tokens=2000,
        funnel_sample_size=5,
    )
    outer = OuterHarness(
        tree_store=tree,
        context_injector=injector,
        trajectory_adapter=adapter,
        cambium=cambium,
        energy_system=energy,
        oplog=oplog,
        config=config,
        decay_sentinel=sentinel,
        lignification=lignification,
    )

    return {
        "tree": tree, "energy": energy, "cambium": cambium,
        "injector": injector, "adapter": adapter, "sentinel": sentinel,
        "lignification": lignification, "outer": outer, "llm": llm,
        "oplog": oplog, "embedder": embedder,
    }


def _make_task(task_id="task-001", desc="Fix sorting bug in ORM", repo="django/django"):
    return Task(task_id=task_id, description=desc, repo_path=repo)


def _make_steps(n=3):
    """创建 n 步有意义的 step 序列 (触发 crystallize)。"""
    steps = []
    for i in range(n):
        is_last = (i == n - 1)
        steps.append({
            "action": {"summary": f"Modify model.py to add nulls_first=True to order_by call #{i}"},
            "result": {
                "summary": f"Applied patch to model.py, test_ordering_null passed",
                "patch": f"diff --git a/model.py b/model.py\n+order_by('field', nulls_first=True)",
                "tests": {"test_ordering_null": "pass"},
                "outcome": "pass" if is_last else "pending",
                "duration": 1.5,
                "tokens": 500,
            },
            "outcome": "pass" if is_last else "pending",
        })
    return steps


# ---------------------------------------------------------------------------
# Test 1: 基本端到端 — 单 episode 全 pass
# ---------------------------------------------------------------------------
class TestSingleEpisode:
    def test_full_episode_pass(self, full_setup):
        """单 episode: inner 跑 3 步全 pass → 应产生 cell + oplog 记录。"""
        s = full_setup
        outer = s["outer"]
        llm = s["llm"]

        # 注入 LLM 响应: crystallize 提取知识
        llm.inject("crystallize", '{"decision": "Always use nulls_first=True in ORM order_by", "rationale": "PG and MySQL differ on NULL sorting defaults", "preconditions": [{"kind": "code_invariant", "assertion": "ORM order_by is used", "verify_hint": {"type": "file_grep", "params": {"path": "model.py", "pattern": "order_by"}}}], "evidence": ["test_id:test_ordering_null"], "domain_tags": ["ORM", "sorting"]}')

        inner = MockInnerHarness(_make_steps(3))
        wrapped = outer.wrap(inner)
        task = _make_task()

        ep_record, ep_report = wrapped.run_episode(task)

        # 验证 EpisodeRecord
        assert ep_record.outcome == "pass"
        assert len(ep_record.steps) == 3

        # 验证 EpisodeReport
        assert ep_report.new_cells_count > 0, "Should have crystallized new cells"

        # 验证 OpLog
        op_counts = s["oplog"].count_by_op_type()
        assert op_counts["CRYSTALLIZE"] > 0, "Should have INSERT_CELL ops"
        assert op_counts["DECAY"] > 0, "Should have UPDATE_ENERGY/MATURITY ops"
        # CONNECT 可能是 0: 第一个 cell 在空树中无连接对象
        # 后续步骤的 REINFORCE 走 EnergySystem.reference → UPDATE_ENERGY (DECAY)

        # 验证 TreeStore 有新 cell
        all_cells = s["tree"].sqlite.list_active()
        assert len(all_cells) > 0, "Should have active cells in tree"

    def test_context_block_generated(self, full_setup):
        """before_step 应生成 ContextBlock。"""
        s = full_setup
        outer = s["outer"]

        # 先插入一个 cell,让 context 非空
        cell = create_cell(
            cell_id="seed-1", ring="L0", maturity=0.1, energy=0.5,
            source="distilled", decision="seed decision", rationale="seed rationale",
            domain_tags=["test"],
        )
        s["tree"].insert_cell(cell)

        task = _make_task()
        ctx = outer.before_step(task, step_index=0, episode_id="ep-test")

        assert ctx.token_count > 0
        assert isinstance(ctx.pinned_text, str)
        assert isinstance(ctx.relevant_text, str)
        assert isinstance(ctx.warnings, list)
        assert isinstance(ctx.injected_cell_ids, list)

    def test_context_block_warnings_type(self, full_setup):
        """ContextBlock.warnings 应为 List[str] 而非 List[List[str]]。"""
        s = full_setup
        outer = s["outer"]

        # 手动注入一个 warning
        outer._pending_warnings["ep-warn"] = ["test warning text"]

        task = _make_task()
        ctx = outer.before_step(task, step_index=0, episode_id="ep-warn")

        # warnings 应是 flat list of strings
        assert isinstance(ctx.warnings, list)
        if ctx.warnings:
            for w in ctx.warnings:
                assert isinstance(w, str), f"Warning should be str, got {type(w)}: {w}"


# ---------------------------------------------------------------------------
# Test 2: 多 episode 序贯 — cell 成长 + 升层
# ---------------------------------------------------------------------------
class TestSequentialEpisodes:
    def test_cell_grows_across_episodes(self, full_setup):
        """多 episode: 同一知识点被反复 reference → maturity 上升 → promote。"""
        s = full_setup
        outer = s["outer"]
        llm = s["llm"]

        # Episode 1: crystallize 一个 cell
        llm.inject("crystallize", '{"decision": "Use nulls_first in order_by", "rationale": "DB compatibility", "preconditions": [], "evidence": ["test_id:test_sort"], "domain_tags": ["ORM"]}')
        inner = MockInnerHarness(_make_steps(2))
        wrapped = outer.wrap(inner)
        wrapped.run_episode(_make_task("t1"))

        cells = s["tree"].sqlite.list_active()
        assert len(cells) > 0
        cell_id = cells[0].id
        initial_maturity = cells[0].maturity
        initial_ring = cells[0].ring

        # Episode 2-10: 同一 cell 被反复 reference (通过 before_step 注入)
        for ep in range(2, 12):
            llm.inject("crystallize", '{"decision": "Use nulls_first in order_by", "rationale": "DB compatibility", "preconditions": [], "evidence": ["test_id:test_sort"], "domain_tags": ["ORM"]}')
            inner = MockInnerHarness(_make_steps(2))
            wrapped = outer.wrap(inner)
            wrapped.run_episode(_make_task(f"t{ep}"))

        # 验证 cell 有成长
        cell = s["tree"].get_cell(cell_id)
        assert cell is not None
        assert cell.maturity > initial_maturity, \
            f"Maturity should have grown: {cell.maturity} vs {initial_maturity}"

    def test_decay_candidates_detected(self, full_setup):
        """低能量 cell 被 funnel verify 检测到。"""
        s = full_setup
        tree = s["tree"]
        energy = s["energy"]
        llm = s["llm"]
        outer = s["outer"]

        # 手动插入一个低能量 cell
        cell = create_cell(
            cell_id="decaying-1", ring="L0", maturity=0.05, energy=-0.3,
            source="distilled", decision="old decision", rationale="old rationale",
            domain_tags=["test"],
        )
        tree.insert_cell(cell)

        # 注入 LLM: DecaySentinel Step 3 → decayed
        llm.inject("decay sentinel", '{"result": "decayed", "reason": "outdated knowledge"}')

        # 跑一个 episode 触发 after_step funnel verify
        inner = MockInnerHarness([{
            "action": {"summary": "run tests"},
            "result": {"summary": "all pass", "outcome": "pass", "tests": {"test_x": "pass"}},
            "outcome": "pass",
        }])
        wrapped = outer.wrap(inner)
        ep_record, ep_report = wrapped.run_episode(_make_task())

        # 验证 cell 被 quarantine
        cell = tree.get_cell("decaying-1")
        assert cell.status == "quarantined", f"Cell should be quarantined, got {cell.status}"
        assert ep_report.quarantined_count > 0


# ---------------------------------------------------------------------------
# Test 3: Lignification — promote + merge
# ---------------------------------------------------------------------------
class TestLignificationIntegration:
    def test_promote_in_full_cycle(self, full_setup):
        """cell maturity 达到阈值 → after_episode 触发 promote。"""
        s = full_setup
        tree = s["tree"]
        outer = s["outer"]

        # 插入一个 maturity 足够高的 L1 cell
        cell = create_cell(
            cell_id="ready-promote", ring="L1", maturity=0.45, energy=0.8,
            source="distilled", decision="ready", rationale="tested",
            domain_tags=["test"],
        )
        tree.insert_cell(cell)

        # 跑一个 episode
        inner = MockInnerHarness([{
            "action": {"summary": "run tests"},
            "result": {"summary": "pass", "outcome": "pass"},
            "outcome": "pass",
        }])
        wrapped = outer.wrap(inner)
        llm = s["llm"]
        llm.inject("crystallize", '{"decision": "new", "rationale": "r", "preconditions": [], "evidence": [], "domain_tags": ["test"]}')
        _, ep_report = wrapped.run_episode(_make_task())

        # 验证 promote 发生
        cell = tree.get_cell("ready-promote")
        assert cell.ring == "L2", f"Cell should be promoted to L2, got {cell.ring}"
        assert len(ep_report.promoted) > 0

    def test_demote_in_full_cycle(self, full_setup):
        """cell maturity 低于 demote 阈值 → after_episode 触发 demote。"""
        s = full_setup
        tree = s["outer"].tree_store if hasattr(s["outer"], "tree_store") else s["tree"]
        outer = s["outer"]

        cell = create_cell(
            cell_id="fading-1", ring="L2", maturity=0.25, energy=0.2,
            source="distilled", decision="old", rationale="r",
            domain_tags=["test"],
        )
        s["tree"].insert_cell(cell)

        inner = MockInnerHarness([{
            "action": {"summary": "run tests"},
            "result": {"summary": "pass", "outcome": "pass"},
            "outcome": "pass",
        }])
        wrapped = outer.wrap(inner)
        s["llm"].inject("crystallize", '{"decision": "x", "rationale": "y", "preconditions": [], "evidence": [], "domain_tags": ["test"]}')
        _, ep_report = wrapped.run_episode(_make_task())

        cell = s["tree"].get_cell("fading-1")
        assert cell.ring == "L1", f"Cell should be demoted to L1, got {cell.ring}"


# ---------------------------------------------------------------------------
# Test 4: OpLog 完整性
# ---------------------------------------------------------------------------
class TestOpLogIntegrity:
    def test_all_ops_logged(self, full_setup):
        """跑完 episode 后, OpLog 应有完整操作记录。"""
        s = full_setup
        llm = s["llm"]
        llm.inject("crystallize", '{"decision": "test decision", "rationale": "test rationale", "preconditions": [], "evidence": ["test_id:test_x"], "domain_tags": ["test"]}')

        inner = MockInnerHarness(_make_steps(3))
        wrapped = s["outer"].wrap(inner)
        wrapped.run_episode(_make_task())

        entries = s["oplog"].get_entries()
        op_types = {e.op for e in entries}

        # 应至少有这些 op
        assert OpEnum.INSERT_CELL.value in op_types, "Should have INSERT_CELL"
        assert OpEnum.UPDATE_ENERGY.value in op_types, "Should have UPDATE_ENERGY"
        assert OpEnum.UPDATE_MATURITY.value in op_types, "Should have UPDATE_MATURITY"

    def test_oplog_count_by_operator(self, full_setup):
        """count_by_op_type 返回 5 个算符 key。"""
        s = full_setup
        s["llm"].inject("crystallize", '{"decision": "d", "rationale": "r", "preconditions": [], "evidence": [], "domain_tags": ["x"]}')

        inner = MockInnerHarness(_make_steps(2))
        wrapped = s["outer"].wrap(inner)
        wrapped.run_episode(_make_task())

        counts = s["oplog"].count_by_op_type()
        assert set(counts.keys()) == {"CRYSTALLIZE", "CONNECT", "PROMOTE", "QUARANTINE", "DECAY"}


# ---------------------------------------------------------------------------
# Test 5: 双库一致性
# ---------------------------------------------------------------------------
class TestDualStoreConsistency:
    def test_consistency_after_episode(self, full_setup):
        """跑完 episode 后, SQLite 与 KuzuDB 应一致。"""
        s = full_setup
        s["llm"].inject("crystallize", '{"decision": "d", "rationale": "r", "preconditions": [], "evidence": [], "domain_tags": ["x"]}')

        inner = MockInnerHarness(_make_steps(2))
        wrapped = s["outer"].wrap(inner)
        wrapped.run_episode(_make_task())

        inconsistent = s["tree"].consistency_check()
        assert inconsistent == [], f"Found inconsistent cells: {inconsistent}"

    def test_consistency_after_quarantine(self, full_setup):
        """quarantine 后双库仍一致。"""
        s = full_setup
        tree = s["tree"]
        llm = s["llm"]

        cell = create_cell(
            cell_id="q-1", ring="L0", maturity=0.05, energy=-0.3,
            source="distilled", decision="bad", rationale="wrong",
            domain_tags=["test"],
        )
        tree.insert_cell(cell)

        llm.inject("decay sentinel", '{"result": "decayed", "reason": "wrong"}')
        inner = MockInnerHarness([{
            "action": {"summary": "run tests"},
            "result": {"summary": "pass", "outcome": "pass"},
            "outcome": "pass",
        }])
        wrapped = s["outer"].wrap(inner)
        wrapped.run_episode(_make_task())

        assert s["tree"].consistency_check() == []


# ---------------------------------------------------------------------------
# Test 6: 空树 / 边界
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_tree_first_episode(self, full_setup):
        """空树跑第一个 episode: before_step 无 pinned/relevant cells。"""
        s = full_setup
        s["llm"].inject("crystallize", '{"decision": "d", "rationale": "r", "preconditions": [], "evidence": [], "domain_tags": ["x"]}')

        inner = MockInnerHarness(_make_steps(2))
        wrapped = s["outer"].wrap(inner)
        ep_record, ep_report = wrapped.run_episode(_make_task())

        # 应正常完成,不崩溃
        assert ep_record.outcome == "pass"

    def test_episode_with_no_crystallization(self, full_setup):
        """inner 只做机械操作 → 不 crystallize。"""
        s = full_setup

        steps = [{
            "action": {"summary": "ls -la"},
            "result": {"summary": "file list", "outcome": "pending"},
            "outcome": "pending",
        }, {
            "action": {"summary": "cat README"},
            "result": {"summary": "readme content", "outcome": "pass"},
            "outcome": "pass",
        }]
        inner = MockInnerHarness(steps)
        wrapped = s["outer"].wrap(inner)
        ep_record, ep_report = wrapped.run_episode(_make_task())

        assert ep_report.new_cells_count == 0

    def test_fail_outcome_still_works(self, full_setup):
        """inner 失败的 episode 也应正常走完三 hook。"""
        s = full_setup
        s["llm"].inject("crystallize", '{"decision": "learned from failure", "rationale": "avoid this approach", "preconditions": [], "evidence": [], "domain_tags": ["lesson"]}')

        steps = [{
            "action": {"summary": "apply wrong fix"},
            "result": {"summary": "tests failed", "patch": "bad patch", "tests": {"test_x": "fail"}, "outcome": "fail"},
            "outcome": "fail",
        }]
        inner = MockInnerHarness(steps)
        wrapped = s["outer"].wrap(inner)
        ep_record, ep_report = wrapped.run_episode(_make_task())

        assert ep_record.outcome == "fail"
        # 失败的 step 有 patch + test_results → should_crystallize = True
        assert ep_report.new_cells_count > 0


# ---------------------------------------------------------------------------
# Test 7: 多 episode 累积效果
# ---------------------------------------------------------------------------
class TestCumulativeEffect:
    def test_tree_grows_over_episodes(self, full_setup):
        """跑 5 个 episode 后,树中 cell 数应单调增长。"""
        s = full_setup
        llm = s["llm"]
        outer = s["outer"]

        cell_counts = []
        for ep in range(5):
            # 每次注入不同的 decision 以避免 dedup REINFORCE
            llm.inject("crystallize", f'{{"decision": "decision for episode {ep}", "rationale": "rationale {ep}", "preconditions": [], "evidence": ["test_id:test_{ep}"], "domain_tags": ["ep{ep}"]}}')
            inner = MockInnerHarness(_make_steps(2))
            wrapped = outer.wrap(inner)
            wrapped.run_episode(_make_task(f"task-{ep}"))
            cell_counts.append(len(s["tree"].sqlite.list_active()))

        # cell 数应增长 (至少前几个 episode)
        assert cell_counts[-1] > cell_counts[0], \
            f"Tree should grow: {cell_counts}"

    def test_stats_after_multiple_episodes(self, full_setup):
        """跑完后 tree.stats() 应返回有效数据。"""
        s = full_setup
        llm = s["llm"]
        outer = s["outer"]

        for ep in range(3):
            llm.inject("crystallize", f'{{"decision": "d{ep}", "rationale": "r{ep}", "preconditions": [], "evidence": [], "domain_tags": ["t{ep}"]}}')
            inner = MockInnerHarness(_make_steps(2))
            wrapped = outer.wrap(inner)
            wrapped.run_episode(_make_task(f"t{ep}"))

        stats = s["tree"].stats()
        assert stats["total_cells"] > 0
        assert "by_ring" in stats
        assert "by_status" in stats
        assert stats["oplog_seq"] > 0
