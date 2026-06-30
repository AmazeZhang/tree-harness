"""100 episode 整合模拟 —— 验证半衰期梯度 / 升降层 / 滞回无震荡。

这是 Phase 0 + Phase 1 的端到端验证:
  - 不同 ring 的衰减半衰期梯度 (L0 快→L4 慢)
  - 频繁引用的 cell 升层, 频繁挑战的 cell 降层
  - user_directive cell 免疫衰减
  - 所有 ring 变化方向单调 (滞回机制防止震荡)
"""
import pytest

from tree_harness.core.cell_model import create_cell
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.oplog import OpLog
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem, EnergyConfig
from tree_harness.modules.ring_promotion import RingPromotion, PromotionConfig


@pytest.fixture
def setup(tmp_path):
    embedder = DeterministicEmbedder(dim=32)
    sqlite = SQLiteBackend(":memory:", embedder=embedder)
    kuzu = KuzuBackend(str(tmp_path / "kuzu"))
    oplog = OpLog(str(tmp_path / "oplog.db"))
    tree = TreeStore(sqlite, kuzu, oplog)
    energy_sys = EnergySystem(EnergyConfig(), tree)
    rp = RingPromotion(PromotionConfig(), tree)
    yield tree, energy_sys, rp


def _cell(cid, ring, maturity, energy, source="distilled"):
    return create_cell(
        cell_id=cid, ring=ring, maturity=maturity, energy=energy,
        source=source, decision=f"decision-{cid}", rationale=f"rationale-{cid}",
    )


def test_simulation_100_episodes(setup):
    tree, energy_sys, rp = setup
    ring_order = ["L0", "L1", "L2", "L3", "L4"]

    # === 创建初始 cell ===
    # 半衰期梯度验证: 不同 ring 的 cell 纯衰减
    # maturity 设在各 ring 的 dead zone 内 (高于 demote 阈值, 低于 promote 阈值)
    # 避免因 maturity=0 触发立即降级
    tree.insert_cell(_cell("decay_l0", "L0", 0.01, 0.5))   # L0: 无 demote, promote<0.15
    tree.insert_cell(_cell("decay_l1", "L1", 0.16, 0.5))   # L1: demote<0.05, promote≥0.40
    tree.insert_cell(_cell("decay_l3", "L3", 0.66, 0.5))   # L3: demote<0.55, promote≥0.85
    tree.insert_cell(_cell("decay_l4", "L4", 0.86, 0.5))   # L4: demote<0.75, 无 promote

    # user_directive 不衰减
    tree.insert_cell(_cell("directive", "L4", 0.85, 1.0, source="user_directive"))

    # 成长型: 每回合 2 次 reference → 能量/成熟度上升 → 升层
    tree.insert_cell(_cell("growing", "L0", 0.0, 0.5))

    # 衰退型: 每回合 1 次 challenge → 能量/成熟度下降 → 降层
    tree.insert_cell(_cell("fading", "L2", 0.50, 0.5))

    # 注册所有 cell 到 RingPromotion (birth episode = 0)
    for c in tree.sqlite.list_active():
        rp.register_cell(c.id)

    # 记录 ring 变化历史 (用于检测震荡)
    ring_history: dict[str, list[str]] = {}
    for c in tree.sqlite.list_active():
        ring_history[c.id] = [c.ring]

    # === 100 episode 模拟 ===
    for ep in range(100):
        ep_id = f"ep{ep}"
        rp.advance_episode()

        # 成长型: 2 次 reference
        energy_sys.reference("growing", ep_id)
        energy_sys.reference("growing", ep_id)

        # 衰退型: 1 次 challenge
        energy_sys.challenge("fading", ep_id)

        # 全局自然衰减
        energy_sys.decay_all(ep_id)

        # 全局成熟度更新
        energy_sys.update_all_maturity(ep_id)

        # 升降层评估
        rp.evaluate_all(ep_id)

        # 记录 ring
        for cid in ring_history:
            cell = tree.get_cell(cid)
            if cell is not None:
                ring_history[cid].append(cell.ring)

    # ================================================================
    # 断言 1: 半衰期梯度 — L0 衰减远快于 L4
    # ================================================================
    e_l0 = tree.get_cell("decay_l0").energy   # 0.5 * 0.7^100  ≈ 0
    e_l1 = tree.get_cell("decay_l1").energy   # 0.5 * 0.9^100  ≈ 0
    e_l3 = tree.get_cell("decay_l3").energy   # 0.5 * 0.99^100 ≈ 0.183
    e_l4 = tree.get_cell("decay_l4").energy   # 0.5 * 0.998^100 ≈ 0.409

    assert e_l0 < 0.001,       f"L0 energy should be ~0, got {e_l0}"
    assert e_l1 < 0.01,        f"L1 energy should be ~0, got {e_l1}"
    assert e_l3 > 0.10,        f"L3 energy should retain >0.10, got {e_l3}"
    assert e_l4 > 0.30,        f"L4 energy should retain >0.30, got {e_l4}"
    assert e_l0 < e_l4 * 0.1,  f"Half-life gradient: L0 ({e_l0}) should be << L4 ({e_l4})"

    # ================================================================
    # 断言 2: user_directive cell 免疫衰减
    # ================================================================
    assert tree.get_cell("directive").energy == pytest.approx(1.0)

    # ================================================================
    # 断言 3: 成长型 cell 被升层 (至少 L0→L1)
    # ================================================================
    growing = tree.get_cell("growing")
    assert growing.ring != "L0", \
        f"Growing cell should have been promoted from L0, still at {growing.ring}"
    assert growing.maturity > 0.15, \
        f"Growing cell maturity should be >0.15, got {growing.maturity}"

    # ================================================================
    # 断言 4: 衰退型 cell 被降层 (L2 → L1 或 L0)
    # ================================================================
    fading = tree.get_cell("fading")
    assert ring_order.index(fading.ring) < ring_order.index("L2"), \
        f"Fading cell should have been demoted from L2, still at {fading.ring}"

    # ================================================================
    # 断言 5: 无震荡 — 所有 cell 的 ring 变化方向单调
    # ================================================================
    for cid, history in ring_history.items():
        directions = []
        for i in range(1, len(history)):
            if history[i] != history[i - 1]:
                d = ring_order.index(history[i]) - ring_order.index(history[i - 1])
                directions.append(d)
        if directions:
            all_up = all(d > 0 for d in directions)
            all_down = all(d < 0 for d in directions)
            assert all_up or all_down, \
                f"Cell {cid} oscillated (non-monotonic ring changes): {history}"
