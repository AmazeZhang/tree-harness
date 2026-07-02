"""Metrics 测试 —— 对应 docs/specs/metrics.md 测试用例。"""
import json
import pytest

from tree_harness.core.cell_model import create_cell
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.oplog import OpLog, OpEnum
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.metrics import (
    TaskResult,
    EpisodeSnapshot,
    AblationResult,
    ring_oscillation_rate,
    context_retention_score,
    control_lag,
    entropy_release_per_episode,
    op_count_distribution,
    promote_reason_distribution,
    hv_mv_ratio,
    resolve_rate,
    cumulative_resolve_curve,
    relative_improvement,
    ray_connectivity_rate,
    active_dead_ratio,
    lignification_compression,
    ring_distribution,
    centrality_gini,
    token_per_episode,
    outer_overhead,
    pareto_front,
    take_snapshot,
)


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
def oplog(tmp_path):
    return OpLog(str(tmp_path / "oplog_test.db"))


def _task_result(tid, resolved=False, tokens=100):
    return TaskResult(
        task_id=tid, repo="r", condition="tree_outer",
        resolved=resolved, token_usage=tokens,
    )


def _cell(cid, ring="L0", energy=0.5, maturity=0.1):
    return create_cell(
        cell_id=cid, ring=ring, maturity=maturity, energy=energy,
        decision=f"d-{cid}", rationale=f"r-{cid}",
        domain_tags=["test"],
    )


# ===========================================================================
# E1. Resolve Rate
# ===========================================================================
class TestResolveRate:
    def test_10_results_5_resolved(self):
        """测试用例 1: 10 TaskResult (5 resolved) → rate = 0.5。"""
        results = [_task_result(f"t{i}", resolved=(i < 5)) for i in range(10)]
        assert resolve_rate(results) == pytest.approx(0.5)

    def test_empty_results(self):
        assert resolve_rate([]) == 0.0

    def test_window_filter(self):
        results = [_task_result(f"t{i}", resolved=(i < 3)) for i in range(10)]
        # 最后 5 个: 0 resolved
        assert resolve_rate(results, window=5) == pytest.approx(0.0)


# ===========================================================================
# E2. Cumulative Resolve Curve
# ===========================================================================
class TestCumulativeCurve:
    def test_curve_length_matches_input(self):
        """测试用例 2: curve 长度 = 输入长度。"""
        results = [_task_result(f"t{i}", resolved=(i % 2 == 0)) for i in range(8)]
        curve = cumulative_resolve_curve(results)
        assert len(curve) == 8

    def test_curve_values(self):
        results = [
            _task_result("t0", resolved=True),
            _task_result("t1", resolved=False),
            _task_result("t2", resolved=True),
        ]
        curve = cumulative_resolve_curve(results)
        assert curve == pytest.approx([1.0, 0.5, 2/3])


# ===========================================================================
# E3. Relative Improvement
# ===========================================================================
class TestRelativeImprovement:
    def test_normal_case(self):
        assert relative_improvement(0.6, 0.4) == pytest.approx(50.0)

    def test_zero_baseline(self):
        assert relative_improvement(0.5, 0.0) == float('inf')

    def test_negative_improvement(self):
        assert relative_improvement(0.3, 0.5) == pytest.approx(-40.0)


# ===========================================================================
# H1. Ring Oscillation Rate
# ===========================================================================
class TestRingOscillation:
    def test_promote_then_demote(self, oplog):
        """测试用例 9: promote→demote 序列 → 期望比例。"""
        # c1: promote then demote (oscillated)
        oplog.append(OpEnum.PROMOTE, {"cell_id": "c1", "from_ring": "L0", "to_ring": "L1", "reason": "normal"})
        oplog.append(OpEnum.DEMOTE, {"cell_id": "c1", "from_ring": "L1", "to_ring": "L0", "reason": "normal"})
        # c2: promote only (no oscillation)
        oplog.append(OpEnum.PROMOTE, {"cell_id": "c2", "from_ring": "L0", "to_ring": "L1", "reason": "normal"})

        rate = ring_oscillation_rate(oplog, window=100)
        # 1 oscillated / 2 promoted = 0.5
        assert rate == pytest.approx(0.5)

    def test_no_promote(self, oplog):
        assert ring_oscillation_rate(oplog) == 0.0

    def test_promote_only_no_oscillation(self, oplog):
        oplog.append(OpEnum.PROMOTE, {"cell_id": "c1", "from_ring": "L0", "to_ring": "L1", "reason": "normal"})
        oplog.append(OpEnum.PROMOTE, {"cell_id": "c2", "from_ring": "L0", "to_ring": "L1", "reason": "normal"})
        assert ring_oscillation_rate(oplog) == pytest.approx(0.0)


# ===========================================================================
# H2. Context Retention Score
# ===========================================================================
class TestContextRetention:
    def test_all_retained(self):
        log = [
            {"step": 0, "cell_ids": ["c1", "c2"], "key_cell_ids": ["c1"]},
            {"step": 1, "cell_ids": ["c1", "c2"], "key_cell_ids": ["c1"]},
            {"step": 2, "cell_ids": ["c1", "c2"], "key_cell_ids": ["c1"]},
        ]
        score = context_retention_score(log, horizon=2)
        assert score == pytest.approx(1.0)

    def test_not_retained(self):
        log = [
            {"step": 0, "cell_ids": ["c1"], "key_cell_ids": ["c1"]},
            {"step": 1, "cell_ids": ["c2"], "key_cell_ids": []},
        ]
        score = context_retention_score(log, horizon=1)
        # c1 at step 0, not present at step 1 → retention = 0
        assert score == pytest.approx(0.0)

    def test_empty_log(self):
        assert context_retention_score([]) == 1.0


# ===========================================================================
# H3. Control Lag
# ===========================================================================
class TestControlLag:
    def test_lag_equals_one(self):
        """测试用例 10: quarantine→warning 序列 → lag = 1。"""
        q_ops = [{"cell_id": "c1", "step": 5}]
        w_injections = [{"cell_id": "c1", "step": 6}]
        lag = control_lag(q_ops, w_injections)
        assert lag == pytest.approx(1.0)

    def test_no_quarantine(self):
        assert control_lag([], [{"cell_id": "c1", "step": 1}]) == 0.0

    def test_warning_not_injected(self):
        q_ops = [{"cell_id": "c1", "step": 5}]
        w_injections = []
        lag = control_lag(q_ops, w_injections)
        assert lag == pytest.approx(999.0)


# ===========================================================================
# H4. Entropy Release
# ===========================================================================
class TestEntropyRelease:
    def test_weighted_sum(self):
        result = entropy_release_per_episode(
            compressed_count=2, quarantined_count=1, decayed_count=3,
        )
        # default weights: 1.0*2 + 2.0*1 + 0.5*3 = 2 + 2 + 1.5 = 5.5
        assert result == pytest.approx(5.5)

    def test_custom_weights(self):
        result = entropy_release_per_episode(
            1, 1, 1,
            weights={"compressed": 3.0, "quarantined": 5.0, "decayed": 1.0},
        )
        assert result == pytest.approx(9.0)


# ===========================================================================
# H5. Op Count Distribution
# ===========================================================================
class TestOpCountDistribution:
    def test_five_keys(self, oplog):
        """测试用例 11: 五个 key 之和 = 状态变更 op 总数。"""
        oplog.append(OpEnum.INSERT_CELL, {"cell_id": "c1", "ring": "L0", "decision_summary": "d"})
        oplog.append(OpEnum.UPDATE_ENERGY, {"cell_id": "c1", "old_energy": 0.5, "new_energy": 0.6, "reason": "ref"})
        oplog.append(OpEnum.PROMOTE, {"cell_id": "c1", "from_ring": "L0", "to_ring": "L1", "reason": "normal"})

        dist = op_count_distribution(oplog)
        assert set(dist.keys()) == {"CRYSTALLIZE", "CONNECT", "PROMOTE", "QUARANTINE", "DECAY"}
        # INSERT_CELL → CRYSTALLIZE (1), UPDATE_ENERGY → DECAY (1), PROMOTE → PROMOTE (1)
        assert dist["CRYSTALLIZE"] == 1
        assert dist["DECAY"] == 1
        assert dist["PROMOTE"] == 1

    def test_promote_reasons(self, oplog):
        oplog.append(OpEnum.PROMOTE, {"cell_id": "c1", "from_ring": "L0", "to_ring": "L1", "reason": "normal"})
        oplog.append(OpEnum.PROMOTE, {"cell_id": "c2", "from_ring": "L0", "to_ring": "L1", "reason": "overflow_force"})
        dist = promote_reason_distribution(oplog)
        assert dist["normal"] == 1
        assert dist["overflow_force"] == 1


# ===========================================================================
# H6. HV/MV Ratio
# ===========================================================================
class TestHvMvRatio:
    def test_basic_grid(self):
        grid = {
            "inner1": {"outer1": 0.5, "outer2": 0.7, "outer3": 0.6},
            "inner2": {"outer1": 0.4, "outer2": 0.6, "outer3": 0.5},
            "inner3": {"outer1": 0.3, "outer2": 0.5, "outer3": 0.4},
        }
        ratio = hv_mv_ratio(grid)
        assert ratio > 0

    def test_single_inner(self):
        grid = {"inner1": {"outer1": 0.5, "outer2": 0.7}}
        assert hv_mv_ratio(grid) == 0.0  # 需要 >= 2 inner


# ===========================================================================
# S1. Ray Connectivity Rate
# ===========================================================================
class TestRayConnectivity:
    def test_all_connected(self, tree):
        """测试用例 3: 全部 cell 有 ray → connectivity = 1.0。"""
        c1 = _cell("c1", ring="L1")
        c2 = _cell("c2", ring="L0")
        tree.insert_cell(c1)
        tree.insert_cell(c2)
        tree.add_ray("c2", "c1", 0.8)  # c2→c1 ray
        rate = ray_connectivity_rate(tree)
        # c1 有 incoming ray, c2 has outgoing ray → no orphans
        # But find_orphans checks if cell has ANY RAY (in or out)
        assert rate == pytest.approx(1.0)

    def test_empty_tree(self, tree):
        """测试用例 7: 空 harness → 不崩溃。"""
        assert ray_connectivity_rate(tree) == 1.0


# ===========================================================================
# S2. Active/Dead Ratio
# ===========================================================================
class TestActiveDeadRatio:
    def test_all_active(self, tree):
        tree.insert_cell(_cell("c1"))
        tree.insert_cell(_cell("c2"))
        ratio = active_dead_ratio(tree)
        assert ratio == float('inf')  # no dead cells

    def test_mixed(self, tree):
        tree.insert_cell(_cell("c1"))
        tree.insert_cell(_cell("c2"))
        tree.quarantine("c2", reason="test")
        ratio = active_dead_ratio(tree)
        # 1 active / 1 dead = 1.0
        assert ratio == pytest.approx(1.0)

    def test_empty_tree(self, tree):
        assert active_dead_ratio(tree) == float('inf')


# ===========================================================================
# S3. Lignification Compression
# ===========================================================================
class TestLignificationCompression:
    def test_3_to_1_compression(self, oplog):
        """测试用例 4: 3 cell merge → compression = 3.0。"""
        oplog.append(OpEnum.MERGE, {"source_ids": ["c1", "c2", "c3"], "target_id": "merged"})
        ratio = lignification_compression(oplog)
        assert ratio == pytest.approx(3.0)

    def test_no_merges(self, oplog):
        assert lignification_compression(oplog) == 0.0

    def test_multiple_merges(self, oplog):
        oplog.append(OpEnum.MERGE, {"source_ids": ["c1", "c2"], "target_id": "m1"})
        oplog.append(OpEnum.MERGE, {"source_ids": ["c3", "c4", "c5"], "target_id": "m2"})
        # total sources = 5, total targets = 2 → 2.5
        assert lignification_compression(oplog) == pytest.approx(2.5)


# ===========================================================================
# S4. Ring Distribution
# ===========================================================================
class TestRingDistribution:
    def test_sum_equals_active(self, tree):
        """测试用例 5: ring_distribution 总和 = active cell 总数。"""
        tree.insert_cell(_cell("c1", ring="L0"))
        tree.insert_cell(_cell("c2", ring="L0"))
        tree.insert_cell(_cell("c3", ring="L1"))
        tree.insert_cell(_cell("c4", ring="L2"))

        dist = ring_distribution(tree)
        assert sum(dist.values()) == 4
        assert dist["L0"] == 2
        assert dist["L1"] == 1
        assert dist["L2"] == 1

    def test_empty_tree(self, tree):
        dist = ring_distribution(tree)
        assert sum(dist.values()) == 0


# ===========================================================================
# S5. Centrality Gini
# ===========================================================================
class TestCentralityGini:
    def test_uniform_distribution(self, tree):
        """所有 cell 入度相同 → Gini ≈ 0。"""
        c1 = _cell("c1", ring="L2")
        c2 = _cell("c2", ring="L2")
        tree.insert_cell(c1)
        tree.insert_cell(c2)
        # 两者都没有 incoming ray → all zeros → Gini = 0
        gini = centrality_gini(tree)
        assert gini == pytest.approx(0.0)

    def test_concentrated_distribution(self, tree):
        """一个 cell 有所有入度 → Gini 接近 1。"""
        c_hub = _cell("hub", ring="L2")
        tree.insert_cell(c_hub)
        for i in range(5):
            c = _cell(f"c{i}", ring="L1")
            tree.insert_cell(c)
            tree.add_ray(f"c{i}", "hub", 0.8)
        gini = centrality_gini(tree)
        assert gini > 0.5  # should be high (concentrated)

    def test_empty_tree(self, tree):
        assert centrality_gini(tree) == 0.0


# ===========================================================================
# C1. Token Per Episode
# ===========================================================================
class TestTokenPerEpisode:
    def test_returns_list(self):
        results = [_task_result(f"t{i}", tokens=100 * (i + 1)) for i in range(3)]
        tokens = token_per_episode(results)
        assert tokens == [100, 200, 300]


# ===========================================================================
# C2. Outer Overhead
# ===========================================================================
class TestOuterOverhead:
    def test_ratio(self):
        assert outer_overhead(1000, 200) == pytest.approx(0.2)

    def test_zero_total(self):
        assert outer_overhead(0, 100) == 0.0


# ===========================================================================
# C3. Pareto Front
# ===========================================================================
class TestParetoFront:
    def test_identifies_non_dominated(self):
        """测试用例 6: pareto_front 正确识别非支配解。"""
        conditions = {
            "A": (0.5, 100),  # moderate rate, moderate cost
            "B": (0.7, 150),  # higher rate, higher cost → not dominated by A (cost higher)
            "C": (0.4, 200),  # lower rate, higher cost → dominated by A (rate & cost both worse)
            "D": (0.8, 300),  # highest rate, highest cost → not dominated by B (rate higher, cost higher)
        }
        front = pareto_front(conditions)
        # A not dominated by B (cost lower), not dominated by C (rate higher)
        # B not dominated by A (rate higher), not dominated by D (cost lower)
        # D not dominated by B (rate higher, cost higher → not dominated)
        # C dominated by A (A has higher rate AND lower cost)
        assert "A" in front
        assert "B" in front
        assert "D" in front
        assert "C" not in front  # C is dominated by A

    def test_single_condition(self):
        assert pareto_front({"A": (0.5, 100)}) == ["A"]

    def test_empty(self):
        assert pareto_front({}) == []


# ===========================================================================
# EpisodeSnapshot
# ===========================================================================
class TestEpisodeSnapshot:
    def test_jsonl_serialization(self, tree, oplog):
        """测试用例 8: EpisodeSnapshot 序列化为 JSONL。"""
        tree.insert_cell(_cell("c1", ring="L0"))
        snapshot = take_snapshot(
            tree_store=tree, oplog=oplog,
            episode_index=0, timestamp="2026-06-26T10:00:00Z",
            resolved=True, cumulative_rate=1.0,
            token_usage=500, duration_seconds=10.5,
            entropy_released=3.0,
        )
        jsonl = snapshot.to_jsonl()
        parsed = json.loads(jsonl)

        assert parsed["episode_index"] == 0
        assert parsed["resolved"] is True
        assert parsed["cumulative_resolve_rate"] == 1.0
        assert "op_counts" in parsed
        assert "ring_distribution" in parsed
        assert parsed["total_active_cells"] == 1

    def test_from_dict_roundtrip(self, tree, oplog):
        tree.insert_cell(_cell("c1", ring="L0"))
        snapshot = take_snapshot(
            tree_store=tree, oplog=oplog,
            episode_index=1, timestamp="2026-06-26T11:00:00Z",
            resolved=False, cumulative_rate=0.5,
        )
        jsonl = snapshot.to_jsonl()
        parsed = json.loads(jsonl)
        restored = EpisodeSnapshot.from_dict(parsed)
        assert restored.episode_index == 1
        assert restored.resolved is False
