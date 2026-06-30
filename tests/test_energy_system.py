"""EnergySystem 测试 —— 对应 docs/specs/energy_system.md 测试用例。"""
import pytest

from tree_harness.core.cell_model import create_cell
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.oplog import OpLog
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem, EnergyConfig


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
def energy(tree):
    return EnergySystem(EnergyConfig(), tree)


def _cell(cid, ring="L0", maturity=0.0, energy=0.5, source="distilled",
          decision="d", rationale="r"):
    return create_cell(
        cell_id=cid, ring=ring, maturity=maturity, energy=energy,
        source=source, decision=decision, rationale=rationale,
    )


# 测试用例 1: 新 cell (energy=0.5) 被 reference 一次 → energy = 0.60
def test_reference_increases_energy(energy, tree):
    tree.insert_cell(_cell("c1", energy=0.5))
    energy.reference("c1", "ep1")
    assert tree.get_cell("c1").energy == pytest.approx(0.60)


# 测试用例 2: L1 cell (energy=0.6) 经过一个 episode 无事件 → energy = 0.54
def test_decay_l1_one_episode(energy, tree):
    tree.insert_cell(_cell("c1", ring="L1", energy=0.6))
    energy.decay_all("ep1")
    assert tree.get_cell("c1").energy == pytest.approx(0.54)


# 测试用例 3: cell 被 challenge 3 次 → energy 下降 0.45
def test_challenge_three_times(energy, tree):
    tree.insert_cell(_cell("c1", energy=0.5))
    for _ in range(3):
        energy.challenge("c1", "ep1")
    assert tree.get_cell("c1").energy == pytest.approx(0.05)


# 测试用例 4: user_directive cell 经过 100 episode → energy 不变 (decay_rate=0)
def test_user_directive_no_decay(energy, tree):
    tree.insert_cell(_cell("c1", source="user_directive", energy=0.8))
    for i in range(100):
        energy.decay_all(f"ep{i}")
    assert tree.get_cell("c1").energy == pytest.approx(0.8)


# 测试用例 5: maturity 从 0.39 经一次正能量 episode → 跨 0.40 → promote 候选
def test_maturity_promote_candidate(energy, tree):
    tree.insert_cell(_cell("c1", ring="L1", maturity=0.39, energy=1.0))
    energy.update_maturity("c1", "ep1")
    candidates = energy.get_promotion_candidates()
    assert ("c1", "L2") in candidates


# 测试用例 6: maturity 从 0.41 经多次负能量 episode → 跌到 0.29 → demote 候选
def test_maturity_demote_candidate(energy, tree):
    tree.insert_cell(_cell("c1", ring="L2", maturity=0.41, energy=-1.0))
    for _ in range(3):
        energy.update_maturity("c1", "ep1")
    candidates = energy.get_demotion_candidates()
    assert ("c1", "L1") in candidates


# 测试用例 7: maturity 在 0.38 时不触发 demote (滞回: demote 阈值是 0.30 而非 0.40)
def test_dead_zone_no_demote(energy, tree):
    tree.insert_cell(_cell("c1", ring="L2", maturity=0.38))
    candidates = energy.get_demotion_candidates()
    assert ("c1", "L1") not in candidates


# 测试用例 8: 模拟 20 episode 纯衰减 → L0 趋近 0, L4 几乎不变
def test_decay_comparison_l0_vs_l4(energy, tree):
    tree.insert_cell(_cell("c0", ring="L0", energy=0.5))
    tree.insert_cell(_cell("c4", ring="L4", energy=0.5))
    for i in range(20):
        energy.decay_all(f"ep{i}")
    e0 = tree.get_cell("c0").energy
    e4 = tree.get_cell("c4").energy
    assert e0 < 0.01        # L0 (decay_rate=0.30) 趋近 0
    assert e4 > 0.45        # L4 (decay_rate=0.002) 几乎不变


# 测试用例 9: energy < -0.20 的 cell 出现在 get_decay_candidates 中
def test_decay_candidates_below_threshold(energy, tree):
    tree.insert_cell(_cell("c1", energy=-0.30))
    tree.insert_cell(_cell("c2", energy=0.5))
    candidates = energy.get_decay_candidates()
    assert "c1" in candidates
    assert "c2" not in candidates


# reference 返回 True (成功强化) / False (目标不存在或非 active) (P1-Bug4)
def test_reference_returns_bool(energy, tree):
    tree.insert_cell(_cell("c1", energy=0.5))
    assert energy.reference("c1", "ep1") is True
    # 不存在的 cell
    assert energy.reference("nope", "ep1") is False
    # 非 active cell
    tree.quarantine("c1", "decayed", "ep1")
    assert energy.reference("c1", "ep1") is False


# challenge 返回 True / False (P1-Bug4)
def test_challenge_returns_bool(energy, tree):
    tree.insert_cell(_cell("c1", energy=0.5))
    assert energy.challenge("c1", "ep1") is True
    # 不存在的 cell
    assert energy.challenge("nope", "ep1") is False
    # 非 active cell
    tree.quarantine("c1", "decayed", "ep1")
    assert energy.challenge("c1", "ep1") is False


# get_decay_candidates 支持 limit (outer_harness after_step 抽样验证用)
def test_get_decay_candidates_with_limit(energy, tree):
    for i in range(5):
        tree.insert_cell(_cell(f"c{i}", energy=-0.5))
    all_candidates = energy.get_decay_candidates()
    assert len(all_candidates) == 5
    limited = energy.get_decay_candidates(limit=2)
    assert len(limited) == 2
