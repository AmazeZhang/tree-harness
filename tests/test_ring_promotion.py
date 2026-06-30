"""RingPromotion 测试 —— 对应 docs/specs/ring_promotion.md 测试用例。"""
import pytest

from tree_harness.core.cell_model import create_cell
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.oplog import OpLog, OpEnum
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.ring_promotion import RingPromotion, PromotionConfig


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
def rp(tree):
    return RingPromotion(PromotionConfig(), tree)


def _cell(cid, ring="L0", maturity=0.0, energy=0.5, source="distilled",
          decision="d", rationale="r"):
    return create_cell(
        cell_id=cid, ring=ring, maturity=maturity, energy=energy,
        source=source, decision=decision, rationale=rationale,
    )


# 测试用例 1: maturity=0.39 的 L1 cell → should_promote 返回 None
def test_should_promote_below_threshold(rp):
    cell = _cell("c1", ring="L1", maturity=0.39)
    assert rp.should_promote(cell, 100) is None


# 测试用例 2: maturity=0.41 的 L1 cell, age=12 → should_promote 返回 "L2"
def test_should_promote_meets_criteria(rp):
    cell = _cell("c1", ring="L1", maturity=0.41)
    assert rp.should_promote(cell, 12) == "L2"


# 测试用例 3: maturity=0.41 的 L1 cell, age=5 → should_promote 返回 None (blocked: min_age)
def test_should_promote_blocked_by_min_age(rp):
    cell = _cell("c1", ring="L1", maturity=0.41)
    assert rp.should_promote(cell, 5) is None


# 测试用例 4: maturity=0.29 的 L2 cell → should_demote 返回 "L1"
def test_should_demote_below_threshold(rp):
    cell = _cell("c1", ring="L2", maturity=0.29)
    assert rp.should_demote(cell) == "L1"


# 测试用例 5: maturity=0.31 的 L2 cell → should_demote 返回 None (dead zone)
def test_should_demote_in_dead_zone(rp):
    cell = _cell("c1", ring="L2", maturity=0.31)
    assert rp.should_demote(cell) is None


# 测试用例 6: maturity=0.90 的 L2 cell → should_promote 返回 "L3" (不是 "L4", 禁止跳级)
def test_should_promote_no_skip(rp):
    cell = _cell("c1", ring="L2", maturity=0.90)
    assert rp.should_promote(cell, 100) == "L3"


# 测试用例 7: evaluate_all 在 10 个 cell 中正确识别 2 promote + 1 demote + 1 blocked
def test_evaluate_all_mixed(rp, tree):
    # c1: L0, maturity=0.20 → promote L0→L1 (age=12 >= min_age=3)
    # c2: L1, maturity=0.45 → promote L1→L2 (age=12 >= min_age=10)
    # c3: L2, maturity=0.25 → demote L2→L1 (maturity < 0.30)
    # c4: L1, maturity=0.45 → blocked (age=5 < min_age=10)
    # c5-c10: L1, maturity=0.35 → dead zone [0.05, 0.40)
    tree.insert_cell(_cell("c1", ring="L0", maturity=0.20))
    tree.insert_cell(_cell("c2", ring="L1", maturity=0.45))
    tree.insert_cell(_cell("c3", ring="L2", maturity=0.25))
    tree.insert_cell(_cell("c4", ring="L1", maturity=0.45))
    for i in range(5, 11):
        tree.insert_cell(_cell(f"c{i}", ring="L1", maturity=0.35))

    # 注册大部分 cell 在 episode 0
    for cid in ["c1", "c2", "c3", "c5", "c6", "c7", "c8", "c9", "c10"]:
        rp.register_cell(cid)

    # 推进到 episode 7, 注册 c4 (使其 age=5)
    for _ in range(7):
        rp.advance_episode()
    rp.register_cell("c4")

    # 推进到 episode 12
    for _ in range(5):
        rp.advance_episode()

    report = rp.evaluate_all("ep1")

    assert len(report.promoted) == 2
    assert ("c1", "L0", "L1") in report.promoted
    assert ("c2", "L1", "L2") in report.promoted

    assert len(report.demoted) == 1
    assert ("c3", "L2", "L1") in report.demoted

    assert len(report.blocked) == 1
    assert ("c4", "L2", "min_maturity_age_not_met") in report.blocked


# 测试用例 8: execute_promotion 后 cell.ring 正确、oplog 有 PROMOTE 记录
def test_execute_promotion(rp, tree):
    tree.insert_cell(_cell("c1", ring="L0", maturity=0.20))
    rp.execute_promotion("c1", "L1", "ep1")
    assert tree.get_cell("c1").ring == "L1"
    assert tree.kuzu.get_ring("c1") == "L1"
    ops = [e.op for e in tree.oplog.get_entries()]
    assert OpEnum.PROMOTE in ops


# 测试用例 9: 连续 5 episode maturity 在 [0.30, 0.40) 之间震荡 → 无升降层发生
def test_hysteresis_no_oscillation(rp, tree):
    tree.insert_cell(_cell("c1", ring="L2", maturity=0.35))
    rp.register_cell("c1")

    maturities = [0.35, 0.31, 0.38, 0.32, 0.36]
    for i, m in enumerate(maturities):
        rp.advance_episode()
        tree.update_maturity("c1", m, f"ep{i}")
        report = rp.evaluate_all(f"ep{i}")
        assert len(report.promoted) == 0
        assert len(report.demoted) == 0

    assert tree.get_cell("c1").ring == "L2"
