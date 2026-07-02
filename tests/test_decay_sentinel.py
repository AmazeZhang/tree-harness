"""DecaySentinel 测试 —— 对应 docs/specs/decay_sentinel.md 测试用例。"""
import pytest

from tree_harness.core.cell_model import (
    create_cell, Cell, Precondition, VerifyHint,
)
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.llm_client import DeterministicLLMClient
from tree_harness.core.oplog import OpLog
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem, EnergyConfig
from tree_harness.modules.verifiers import VerifierRegistry
from tree_harness.modules.decay_sentinel import DecaySentinel, Verdict, FunnelStats


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
    return TreeStore(sqlite, kuzu, oplog)


@pytest.fixture
def energy(tree):
    return EnergySystem(EnergyConfig(), tree)


@pytest.fixture
def llm():
    return DeterministicLLMClient()


@pytest.fixture
def registry(tmp_path):
    return VerifierRegistry(repo_path=str(tmp_path))


@pytest.fixture
def sentinel(tree, energy, registry, llm, tmp_path):
    return DecaySentinel(
        tree_store=tree,
        energy_system=energy,
        verifier_registry=registry,
        llm_client=llm,
        repo_path=str(tmp_path),
    )


def _cell(
    cid, ring="L0", maturity=0.0, energy_val=0.5, source="distilled",
    decision="d", rationale="r", evidence=None, preconditions=None,
):
    return create_cell(
        cell_id=cid, ring=ring, maturity=maturity, energy=energy_val,
        source=source, decision=decision, rationale=rationale,
        evidence=evidence or [], preconditions=preconditions or [],
    )


# ---------------------------------------------------------------------------
# Step 2a: 测试验证
# ---------------------------------------------------------------------------
class TestStep2aTestVerify:
    def test_test_passes_verdict_weak_valid(self, sentinel, tree, energy):
        """测试用例 1: cell 有 test_id, 测试 pass → weak_valid。"""
        class FakeRunner:
            def run_test(self, test_id, repo_path):
                return "pass"

        sentinel.test_runner = FakeRunner()
        cell = _cell("c1", energy_val=-0.3,
                     evidence=["test_id:test_foo"])
        tree.insert_cell(cell)
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].result == "weak_valid"
        assert verdicts["c1"].step_reached == 2

    def test_test_fails_verdict_decayed(self, sentinel, tree):
        """测试用例 2: cell 有 test_id, 测试 fail → decayed。"""
        class FakeRunner:
            def run_test(self, test_id, repo_path):
                return "fail"

        sentinel.test_runner = FakeRunner()
        cell = _cell("c1", energy_val=-0.3,
                     evidence=["test_id:test_foo"])
        tree.insert_cell(cell)
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].result == "decayed"
        assert verdicts["c1"].step_reached == 2
        assert "test_foo" in verdicts["c1"].evidence

    def test_no_test_runner_falls_through(self, sentinel, tree, llm):
        """无 test runner → 进 Step 2b/3。"""
        cell = _cell("c1", energy_val=-0.3, evidence=["test_id:test_foo"])
        tree.insert_cell(cell)
        # 无 test_runner (None) 且无 precondition hint → 进 Step 3 LLM
        llm.inject("decay sentinel", '{"result": "uncertain", "reason": "no data"}')
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].step_reached == 3


# ---------------------------------------------------------------------------
# Step 2b: Precondition 核查
# ---------------------------------------------------------------------------
class TestStep2bPreconditionVerify:
    def test_precondition_fail_verdict_decayed(self, sentinel, tree, tmp_path):
        """测试用例 3: precondition verify_hint=file_grep, 文件中不存在 pattern → decayed。"""
        # 创建文件
        import os
        fpath = os.path.join(str(tmp_path), "settings.py")
        with open(fpath, "w") as f:
            f.write("DEBUG = True\n")
        precond = Precondition(
            kind="config",
            assertion="DB engine is postgres",
            verify_hint=VerifyHint(
                type="file_grep",
                params={"path": "settings.py", "pattern": "ENGINE.*postgres"},
            ),
        )
        cell = _cell("c1", energy_val=-0.3, preconditions=[precond])
        tree.insert_cell(cell)
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].result == "decayed"
        assert verdicts["c1"].step_reached == 2
        assert verdicts["c1"].verifier_name == "precondition_verify"

    def test_precondition_pass_verdict_valid(self, sentinel, tree, tmp_path):
        """precondition 验证通过 → valid。"""
        import os
        fpath = os.path.join(str(tmp_path), "settings.py")
        with open(fpath, "w") as f:
            f.write("DATABASE_ENGINE = 'postgres'\n")
        precond = Precondition(
            kind="config",
            assertion="DB engine is postgres",
            verify_hint=VerifyHint(
                type="file_grep",
                params={"path": "settings.py", "pattern": "ENGINE.*postgres"},
            ),
        )
        cell = _cell("c1", energy_val=-0.3, preconditions=[precond])
        tree.insert_cell(cell)
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].result == "valid"
        assert verdicts["c1"].step_reached == 2

    def test_inconclusive_precondition_falls_through(self, sentinel, tree, llm):
        """precondition inconclusive → 进 Step 3。"""
        precond = Precondition(
            kind="code_invariant",
            assertion="some AST check",
            verify_hint=VerifyHint(type="ast_query", params={"query": "find something"}),
        )
        cell = _cell("c1", energy_val=-0.3, preconditions=[precond])
        tree.insert_cell(cell)
        llm.inject("decay sentinel", '{"result": "valid", "reason": "looks ok"}')
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].step_reached == 3


# ---------------------------------------------------------------------------
# Step 3: LLM 仲裁
# ---------------------------------------------------------------------------
class TestStep3LLMArbitrate:
    def test_no_test_no_hint_goes_to_llm(self, sentinel, tree, llm):
        """测试用例 4: 无 test_id 无 verify_hint → 进 Step 3 LLM。"""
        cell = _cell("c1", energy_val=-0.3)
        tree.insert_cell(cell)
        llm.inject("decay sentinel", '{"result": "decayed", "reason": "outdated"}')
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].result == "decayed"
        assert verdicts["c1"].step_reached == 3
        assert verdicts["c1"].verifier_name == "llm_arbitrate"

    def test_llm_uncertain(self, sentinel, tree, llm):
        """LLM 返回 uncertain → uncertain。"""
        cell = _cell("c1", energy_val=-0.3)
        tree.insert_cell(cell)
        llm.inject("decay sentinel", '{"result": "uncertain", "reason": "ambiguous"}')
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].result == "uncertain"

    def test_llm_unparseable_defaults_uncertain(self, sentinel, tree, llm):
        """LLM 无法解析 → 降级 uncertain。"""
        cell = _cell("c1", energy_val=-0.3)
        tree.insert_cell(cell)
        llm.inject("decay sentinel", "this is not json at all")
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].result == "uncertain"


# ---------------------------------------------------------------------------
# Signal-level 副作用
# ---------------------------------------------------------------------------
class TestSignalSideEffects:
    def test_valid_triggers_reference(self, sentinel, tree, energy):
        """测试用例 6: verdict=valid → cell.energy 增加 δ_reference。"""
        import os
        fpath = os.path.join(str(sentinel.repo_path), "settings.py")
        with open(fpath, "w") as f:
            f.write("DATABASE_ENGINE = 'postgres'\n")
        precond = Precondition(
            kind="config", assertion="DB engine is postgres",
            verify_hint=VerifyHint(type="file_grep",
                                   params={"path": "settings.py", "pattern": "ENGINE.*postgres"}),
        )
        cell = _cell("c1", energy_val=-0.3, preconditions=[precond])
        tree.insert_cell(cell)
        old_energy = tree.get_cell("c1").energy
        sentinel.funnel_verify(["c1"], episode_id="ep1")
        new_energy = tree.get_cell("c1").energy
        assert new_energy == pytest.approx(old_energy + 0.10)

    def test_uncertain_triggers_decay_one(self, sentinel, tree, energy, llm):
        """verdict=uncertain → energy -= 0.05 + mark_for_review oplog。"""
        cell = _cell("c1", energy_val=-0.3)
        tree.insert_cell(cell)
        llm.inject("decay sentinel", '{"result": "uncertain", "reason": "not sure"}')
        old_energy = tree.get_cell("c1").energy
        sentinel.funnel_verify(["c1"], episode_id="ep1")
        new_energy = tree.get_cell("c1").energy
        assert new_energy == pytest.approx(old_energy - 0.05)

        # Check MARK_REVIEW oplog entry exists
        history = tree.oplog.get_cell_history("c1")
        op_types = [e.op for e in history]
        assert "MARK_REVIEW" in op_types

    def test_decayed_no_signal_side_effect(self, sentinel, tree, energy, llm):
        """verdict=decayed → Sentinel 不写 TreeStore (quarantine 由 OuterHarness)。"""
        cell = _cell("c1", energy_val=-0.3)
        tree.insert_cell(cell)
        llm.inject("decay sentinel", '{"result": "decayed", "reason": "outdated"}')
        sentinel.funnel_verify(["c1"], episode_id="ep1")
        # cell should still be active (not quarantined)
        assert tree.get_cell("c1").status == "active"


# ---------------------------------------------------------------------------
# Step 1: 被动信号
# ---------------------------------------------------------------------------
class TestStep1PassiveSignals:
    def test_positive_references_verdict_valid(self, sentinel, tree, energy):
        """有引用且结果为正 → valid (Step 1)。"""
        cell = _cell("c1", energy_val=-0.3)
        tree.insert_cell(cell)
        # 模拟两次正面引用
        energy.reference("c1", "ep1")
        energy.reference("c1", "ep2")
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].result == "valid"
        assert verdicts["c1"].step_reached == 1

    def test_no_references_falls_through(self, sentinel, tree, llm):
        """无引用 → 进 Step 2/3。"""
        cell = _cell("c1", energy_val=-0.3)
        tree.insert_cell(cell)
        llm.inject("decay sentinel", '{"result": "valid", "reason": "ok"}')
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].step_reached > 1


# ---------------------------------------------------------------------------
# 漏斗统计
# ---------------------------------------------------------------------------
class TestFunnelStats:
    def test_stats_track_resolutions(self, sentinel, tree, llm, tmp_path):
        """测试用例 7: 漏斗统计正确。"""
        import os
        # Cell A: Step 2b valid (precondition passes)
        fpath = os.path.join(str(tmp_path), "settings.py")
        with open(fpath, "w") as f:
            f.write("DATABASE_ENGINE = 'postgres'\n")
        precond = Precondition(
            kind="config", assertion="DB engine is postgres",
            verify_hint=VerifyHint(type="file_grep",
                                   params={"path": "settings.py", "pattern": "ENGINE.*postgres"}),
        )
        cell_a = _cell("ca", energy_val=-0.3, preconditions=[precond])
        tree.insert_cell(cell_a)

        # Cell B: Step 3 LLM
        cell_b = _cell("cb", energy_val=-0.3)
        tree.insert_cell(cell_b)
        llm.inject("decay sentinel", '{"result": "uncertain", "reason": "no data"}')

        sentinel.funnel_verify(["ca", "cb"], episode_id="ep1")

        stats = sentinel.stats
        assert stats.step2b_resolved >= 1  # cell_a resolved at Step 2b
        assert stats.step3_resolved >= 1   # cell_b resolved at Step 3
        assert stats.verdicts["valid"] >= 1
        assert stats.verdicts["uncertain"] >= 1


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_nonexistent_cell(self, sentinel):
        """不存在的 cell → uncertain verdict。"""
        verdicts = sentinel.funnel_verify(["nonexistent"])
        assert verdicts["nonexistent"].result == "uncertain"

    def test_quarantined_cell(self, sentinel, tree):
        """已 quarantined 的 cell → uncertain (skip)。"""
        cell = _cell("c1", energy_val=-0.3)
        tree.insert_cell(cell)
        tree.quarantine("c1", reason="test")
        verdicts = sentinel.funnel_verify(["c1"])
        assert verdicts["c1"].result == "uncertain"

    def test_empty_candidate_list(self, sentinel):
        """空候选列表 → 空结果。"""
        verdicts = sentinel.funnel_verify([])
        assert verdicts == {}


# ---------------------------------------------------------------------------
# P1-1: 高 ring 抽检
# ---------------------------------------------------------------------------
class TestHighRingSampling:
    """L3/L4 cell 定期抽检,不依赖 energy threshold。"""

    def test_sample_returns_l3_l4_cells(self, sentinel, tree):
        """sample_high_ring_cells 只返回 L3/L4 active cell。"""
        tree.insert_cell(_cell("l0", ring="L0", energy_val=0.5))
        tree.insert_cell(_cell("l1", ring="L1", energy_val=0.5))
        tree.insert_cell(_cell("l3a", ring="L3", energy_val=5.0))
        tree.insert_cell(_cell("l3b", ring="L3", energy_val=3.0))
        tree.insert_cell(_cell("l4", ring="L4", energy_val=10.0))

        sampled = sentinel.sample_high_ring_cells(sample_size=10)
        # 只包含 L3/L4 cell
        assert all(cid in ("l3a", "l3b", "l4") for cid in sampled)
        assert len(sampled) == 3  # 全部 3 个

    def test_sample_respects_size_limit(self, sentinel, tree):
        """sample_size 限制返回数量。"""
        for i in range(10):
            tree.insert_cell(_cell(f"l3_{i}", ring="L3", energy_val=1.0))

        sampled = sentinel.sample_high_ring_cells(sample_size=3)
        assert len(sampled) == 3

    def test_sample_empty_rings(self, sentinel, tree):
        """无 L3/L4 cell → 空列表。"""
        tree.insert_cell(_cell("l0", ring="L0"))
        assert sentinel.sample_high_ring_cells(sample_size=5) == []

    def test_sample_excludes_non_active(self, sentinel, tree):
        """quarantined/archived 的 L3/L4 cell 不被抽取。"""
        tree.insert_cell(_cell("active", ring="L3", energy_val=1.0))
        tree.insert_cell(_cell("quarantined", ring="L3", energy_val=1.0))
        tree.quarantine("quarantined", reason="test")

        sampled = sentinel.sample_high_ring_cells(sample_size=10)
        assert "quarantined" not in sampled
        assert "active" in sampled
