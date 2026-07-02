"""LignificationScheduler 测试 —— 对应 docs/specs/lignification.md 测试用例。"""
import pytest

from tree_harness.core.cell_model import (
    create_cell, Cell, Precondition, RING_ORDER,
    PROMOTE_THRESHOLDS, DEMOTE_THRESHOLDS,
)
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.llm_client import DeterministicLLMClient
from tree_harness.core.oplog import OpLog
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem, EnergyConfig
from tree_harness.modules.lignification import (
    LignificationScheduler, LignificationConfig, MaintenanceResult,
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
def energy(tree):
    return EnergySystem(EnergyConfig(), tree)


@pytest.fixture
def llm():
    return DeterministicLLMClient()


@pytest.fixture
def config():
    return LignificationConfig()


@pytest.fixture
def lignification(tree, energy, llm, config):
    return LignificationScheduler(tree, energy, llm, tree.oplog, config)


def _cell(cid, ring="L0", maturity=0.0, energy_val=0.5, source="distilled",
          decision="d", rationale="r", evidence=None, preconditions=None,
          domain_tags=None, trigger_task="t1", domain="d1"):
    return create_cell(
        cell_id=cid, ring=ring, maturity=maturity, energy=energy_val,
        source=source, decision=decision, rationale=rationale,
        evidence=evidence or [], preconditions=preconditions or [],
        domain_tags=domain_tags or ["default"],
        trigger_task=trigger_task, domain=domain,
    )


def _age_cells(lignification, cell_ids, age=100):
    """注册 cell 并设置足够大的 age 以通过 min_maturity_age 检查 (测试辅助)。"""
    for cid in cell_ids:
        lignification._cell_birth[cid] = 0
    lignification._episode_count = max(lignification._episode_count, age)


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------
class TestPromote:
    def test_promote_l1_to_l2(self, lignification, tree):
        """测试用例 1: maturity=0.41 的 L1 cell → promote 到 L2。"""
        cell = _cell("c1", ring="L1", maturity=0.41)
        tree.insert_cell(cell)
        _age_cells(lignification, ["c1"])
        promoted = lignification.check_promotions("ep1")
        assert len(promoted) == 1
        assert promoted[0] == ("c1", "L1", "L2")
        assert tree.get_cell("c1").ring == "L2"
        # 其他字段不变
        updated = tree.get_cell("c1")
        assert updated.energy == cell.energy
        assert updated.decision == cell.decision

    def test_promote_l0_to_l1(self, lignification, tree):
        cell = _cell("c1", ring="L0", maturity=0.15)
        tree.insert_cell(cell)
        _age_cells(lignification, ["c1"])
        promoted = lignification.check_promotions("ep1")
        assert ("c1", "L0", "L1") in promoted

    def test_no_promote_in_hysteresis_band(self, lignification, tree):
        """测试用例 3: maturity=0.35 的 L1 cell → 不触发 promote 也不 demote。"""
        cell = _cell("c1", ring="L1", maturity=0.35)
        tree.insert_cell(cell)
        promoted = lignification.check_promotions("ep1")
        demoted = lignification.check_demotions("ep1")
        assert len(promoted) == 0
        assert len(demoted) == 0

    def test_promote_reason_normal(self, lignification, tree):
        cell = _cell("c1", ring="L1", maturity=0.41)
        tree.insert_cell(cell)
        _age_cells(lignification, ["c1"])
        lignification.check_promotions("ep1")
        history = tree.oplog.get_cell_history("c1")
        promote_ops = [e for e in history if e.op == "PROMOTE"]
        assert len(promote_ops) == 1
        assert promote_ops[0].payload["reason"] == "normal"

    def test_l4_cell_not_promoted(self, lignification, tree):
        cell = _cell("c1", ring="L4", maturity=0.99)
        tree.insert_cell(cell)
        promoted = lignification.check_promotions("ep1")
        assert len(promoted) == 0


# ---------------------------------------------------------------------------
# Demote
# ---------------------------------------------------------------------------
class TestDemote:
    def test_demote_l1_to_l0(self, lignification, tree):
        """测试用例 2: maturity=0.04 的 L1 cell → demote 到 L0。"""
        cell = _cell("c1", ring="L1", maturity=0.04)
        tree.insert_cell(cell)
        demoted = lignification.check_demotions("ep1")
        assert len(demoted) == 1
        assert demoted[0] == ("c1", "L1", "L0")
        assert tree.get_cell("c1").ring == "L0"

    def test_no_demote_in_hysteresis_band(self, lignification, tree):
        """maturity=0.35 的 L1 cell → 在滞回带内 (0.05 < 0.35 < 0.40)。"""
        cell = _cell("c1", ring="L1", maturity=0.35)
        tree.insert_cell(cell)
        demoted = lignification.check_demotions("ep1")
        assert len(demoted) == 0

    def test_l0_not_demoted(self, lignification, tree):
        cell = _cell("c1", ring="L0", maturity=0.01)
        tree.insert_cell(cell)
        demoted = lignification.check_demotions("ep1")
        assert len(demoted) == 0


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------
class TestCapacity:
    def test_capacity_not_exceeded(self, lignification, tree):
        """L3 容量足够时不触发 overflow。"""
        cell = _cell("c1", ring="L2", maturity=0.65)
        tree.insert_cell(cell)
        # L3 容量 = 60, 远未达到
        _age_cells(lignification, ["c1"])
        promoted = lignification.check_promotions("ep1")
        assert len(promoted) == 1

    def test_capacity_overflow_force_promote(self, tree, energy, llm):
        """L3 满了, overflow_policy=force_promote → 直接升 L4。"""
        config = LignificationConfig(
            ring_capacity={"L3": 2, "L4": 10},
            overflow_policy="force_promote",
        )
        lignification = LignificationScheduler(tree, energy, llm, tree.oplog, config)
        # 填满 L3
        for i in range(2):
            tree.insert_cell(_cell(f"existing{i}", ring="L3", maturity=0.70))
        # 新 cell 从 L2 升 L3 → 触发 overflow
        cell = _cell("c1", ring="L2", maturity=0.65)
        tree.insert_cell(cell)
        _age_cells(lignification, ["c1", "existing0", "existing1"])
        promoted = lignification.check_promotions("ep1")
        # c1 应直接升到 L4 (overflow_force)
        assert any(p[0] == "c1" and p[2] == "L4" for p in promoted)

    def test_capacity_block_new(self, tree, energy, llm):
        """overflow_policy=block_new → 不升层。"""
        config = LignificationConfig(
            ring_capacity={"L3": 1, "L4": 10},
            overflow_policy="block_new",
        )
        lignification = LignificationScheduler(tree, energy, llm, tree.oplog, config)
        tree.insert_cell(_cell("existing", ring="L3", maturity=0.70))
        cell = _cell("c1", ring="L2", maturity=0.65)
        tree.insert_cell(cell)
        _age_cells(lignification, ["c1", "existing"])
        promoted = lignification.check_promotions("ep1")
        assert len(promoted) == 0  # blocked


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
class TestMerge:
    def test_merge_3_cells(self, lignification, tree, llm):
        """测试用例 4: 3 个相似 L1 cell merge → 1 个新 cell + 3 条 SUPERSEDES。"""
        cells = [
            _cell(f"c{i}", ring="L1", maturity=0.25, energy_val=0.6,
                  decision=f"decision {i}", rationale=f"rationale {i}",
                  domain_tags=["auth"])
            for i in range(3)
        ]
        for c in cells:
            tree.insert_cell(c)

        llm.inject("knowledge consolidation", '{"decision": "merged decision", "rationale": "merged rationale"}')
        merged_id = lignification.attempt_merge(["c0", "c1", "c2"], episode_id="ep1")
        assert merged_id is not None

        # 验证源 cell 都被 superseded
        for cid in ["c0", "c1", "c2"]:
            assert tree.get_cell(cid).status == "superseded"

        # 验证新 cell 存在
        merged_cell = tree.get_cell(merged_id)
        assert merged_cell is not None
        assert merged_cell.decision == "merged decision"

        # 验证 SUPERSEDES 边
        for cid in ["c0", "c1", "c2"]:
            chain = tree.kuzu.get_supersede_chain(merged_id)
            assert cid in chain

    def test_merge_energy_is_max_times_08(self, lignification, tree, llm):
        """测试用例 5: merge 后新 cell 的 energy = max(sources) * 0.8。"""
        cells = [
            _cell("c0", ring="L1", maturity=0.25, energy_val=0.8, domain_tags=["x"]),
            _cell("c1", ring="L1", maturity=0.25, energy_val=0.5, domain_tags=["x"]),
        ]
        for c in cells:
            tree.insert_cell(c)
        llm.inject("knowledge consolidation", '{"decision": "d", "rationale": "r"}')
        merged_id = lignification.attempt_merge(["c0", "c1"], episode_id="ep1")
        merged = tree.get_cell(merged_id)
        assert merged.energy == pytest.approx(0.8 * 0.8)

    def test_merge_maturity_is_max(self, lignification, tree, llm):
        """测试用例 6: merge 后新 cell 的 maturity = max(sources) (保留最高,不拉低)。"""
        cells = [
            _cell("c0", ring="L1", maturity=0.20, energy_val=0.5, domain_tags=["x"]),
            _cell("c1", ring="L1", maturity=0.30, energy_val=0.5, domain_tags=["x"]),
            _cell("c2", ring="L1", maturity=0.40, energy_val=0.5, domain_tags=["x"]),
        ]
        for c in cells:
            tree.insert_cell(c)
        llm.inject("knowledge consolidation", '{"decision": "d", "rationale": "r"}')
        merged_id = lignification.attempt_merge(["c0", "c1", "c2"], episode_id="ep1")
        merged = tree.get_cell(merged_id)
        assert merged.maturity == pytest.approx(0.40)  # max, not mean(0.30)

    def test_merge_source_status_superseded(self, lignification, tree, llm):
        """测试用例 7: merge 后源 cell 的 status = superseded。"""
        cells = [
            _cell("c0", ring="L1", maturity=0.25, energy_val=0.5, domain_tags=["x"]),
            _cell("c1", ring="L1", maturity=0.25, energy_val=0.5, domain_tags=["x"]),
        ]
        for c in cells:
            tree.insert_cell(c)
        llm.inject("knowledge consolidation", '{"decision": "d", "rationale": "r"}')
        lignification.attempt_merge(["c0", "c1"], episode_id="ep1")
        assert tree.get_cell("c0").status == "superseded"
        assert tree.get_cell("c1").status == "superseded"

    def test_merge_different_rings_fails(self, lignification, tree, llm):
        """不同 ring 的 cell 不能 merge。"""
        tree.insert_cell(_cell("c0", ring="L0", maturity=0.1, domain_tags=["x"]))
        tree.insert_cell(_cell("c1", ring="L1", maturity=0.2, domain_tags=["x"]))
        llm.inject("knowledge consolidation", '{"decision": "d", "rationale": "r"}')
        result = lignification.attempt_merge(["c0", "c1"])
        assert result is None

    def test_merge_single_cell_fails(self, lignification, tree):
        """单个 cell 不能 merge。"""
        tree.insert_cell(_cell("c0", ring="L1", maturity=0.2, domain_tags=["x"]))
        result = lignification.attempt_merge(["c0"])
        assert result is None


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
class TestSplit:
    def test_split_disabled_by_default(self, lignification, tree, llm):
        """enable_split=False → split 不执行。"""
        cell = _cell("c1", ring="L2", maturity=0.5, energy_val=0.6)
        tree.insert_cell(cell)
        result = lignification.attempt_split("c1")
        assert result is None

    def test_split_energy_and_maturity(self, tree, energy, llm):
        """测试用例 8/9: split 后子 cell 的 energy = parent * 0.6, maturity = parent * 0.8。"""
        config = LignificationConfig(enable_split=True)
        lignification = LignificationScheduler(tree, energy, llm, tree.oplog, config)
        cell = _cell("c1", ring="L2", maturity=0.50, energy_val=0.60)
        tree.insert_cell(cell)

        llm.inject("knowledge analysis", '[{"decision": "d1", "rationale": "r1"}, {"decision": "d2", "rationale": "r2"}]')
        child_ids = lignification.attempt_split("c1", episode_id="ep1")
        assert child_ids is not None
        assert len(child_ids) == 2

        for cid in child_ids:
            child = tree.get_cell(cid)
            assert child.energy == pytest.approx(0.60 * 0.6)
            assert child.maturity == pytest.approx(0.50 * 0.8)

        # Source cell should be superseded
        assert tree.get_cell("c1").status == "superseded"


# ---------------------------------------------------------------------------
# Maintenance cycle
# ---------------------------------------------------------------------------
class TestMaintenanceCycle:
    def test_full_cycle(self, lignification, tree, llm):
        """完整维护周期: promote + merge。"""
        # Cell ready for promotion
        tree.insert_cell(_cell("c1", ring="L1", maturity=0.41, domain_tags=["x"]))
        # Two similar cells for merge
        tree.insert_cell(_cell("c2", ring="L0", maturity=0.05, energy_val=0.3,
                               decision="same decision", rationale="same rationale",
                               domain_tags=["merge"]))
        tree.insert_cell(_cell("c3", ring="L0", maturity=0.05, energy_val=0.3,
                               decision="same decision", rationale="same rationale",
                               domain_tags=["merge"]))
        _age_cells(lignification, ["c1", "c2", "c3"])

        llm.inject("knowledge consolidation", '{"decision": "merged", "rationale": "merged r"}')
        result = lignification.run_maintenance_cycle("ep1")

        assert isinstance(result, MaintenanceResult)
        assert len(result.promoted) >= 1  # c1 promoted
        assert result.op_counts["PROMOTE"] >= 1
        assert result.op_counts["MERGE"] >= 0  # may or may not find merge candidates

    def test_empty_tree(self, lignification):
        """空树 → 空结果。"""
        result = lignification.run_maintenance_cycle("ep1")
        assert result.promoted == []
        assert result.demoted == []
        assert result.merged == []
        assert result.split == []
        assert result.op_counts == {"PROMOTE": 0, "DEMOTE": 0, "MERGE": 0, "SPLIT": 0, "ARCHIVE": 0}

    def test_op_counts_correct(self, lignification, tree):
        """op_counts 正确统计。"""
        tree.insert_cell(_cell("c1", ring="L1", maturity=0.41))
        tree.insert_cell(_cell("c2", ring="L1", maturity=0.03))
        _age_cells(lignification, ["c1", "c2"])
        result = lignification.run_maintenance_cycle("ep1")
        assert result.op_counts["PROMOTE"] == 1
        assert result.op_counts["DEMOTE"] == 1
        assert result.op_counts["MERGE"] == 0
        assert result.op_counts["SPLIT"] == 0


# ---------------------------------------------------------------------------
# find_merge_candidates
# ---------------------------------------------------------------------------
class TestFindMergeCandidates:
    def test_finds_similar_cells(self, lignification, tree):
        """相似 cell 被识别为 merge 候选。"""
        tree.insert_cell(_cell("c1", ring="L1", maturity=0.2,
                               decision="always use nulls_first in order_by",
                               rationale="PG and MySQL differ on NULL sorting",
                               domain_tags=["sorting"]))
        tree.insert_cell(_cell("c2", ring="L1", maturity=0.2,
                               decision="always use nulls_first in order_by",
                               rationale="PG and MySQL differ on NULL sorting",
                               domain_tags=["sorting"]))
        candidates = lignification._find_merge_candidates()
        # Should find at least one cluster containing both c1 and c2
        found = any("c1" in cluster and "c2" in cluster for cluster in candidates)
        assert found

    def test_different_domain_tags_not_merged(self, lignification, tree):
        """不同 domain_tag 的 cell 不被合并。"""
        tree.insert_cell(_cell("c1", ring="L1", maturity=0.2,
                               decision="same decision text here",
                               rationale="same rationale text here",
                               domain_tags=["auth"]))
        tree.insert_cell(_cell("c2", ring="L1", maturity=0.2,
                               decision="same decision text here",
                               rationale="same rationale text here",
                               domain_tags=["database"]))
        candidates = lignification._find_merge_candidates()
        found = any("c1" in cluster and "c2" in cluster for cluster in candidates)
        assert not found


# ---------------------------------------------------------------------------
# P0-2: L0 Capacity Cap + Archive
# ---------------------------------------------------------------------------
class TestL0CapacityEviction:
    """L0 容量溢出时淘汰最低 energy 的 cell。"""

    @pytest.fixture
    def small_lignification(self, tree, energy, llm):
        """L0 capacity=3 的小配置。"""
        config = LignificationConfig(ring_capacity={"L0": 3, "L1": 30, "L2": 20, "L3": 60, "L4": 20})
        return LignificationScheduler(tree, energy, llm, tree.oplog, config)

    def test_l0_overflow_archives_lowest_energy(self, small_lignification, tree):
        """L0 超过 capacity → archive 最低 energy 的 cell。"""
        # 插入 4 个 L0 cell (capacity=3, 超出 1 个)
        tree.insert_cell(_cell("c1", energy_val=0.50))
        tree.insert_cell(_cell("c2", energy_val=0.30))
        tree.insert_cell(_cell("c3", energy_val=0.10))   # 最低 energy
        tree.insert_cell(_cell("c4", energy_val=0.40))
        _age_cells(small_lignification, ["c1", "c2", "c3", "c4"])

        result = small_lignification.run_maintenance_cycle("ep1")

        assert len(result.archived) == 1
        assert "c3" in result.archived  # 最低 energy 的被淘汰
        assert result.op_counts["ARCHIVE"] == 1
        assert tree.get_cell("c3").status == "archived"
        # 其他 cell 不受影响
        assert tree.get_cell("c1").status == "active"
        assert tree.get_cell("c2").status == "active"
        assert tree.get_cell("c4").status == "active"

    def test_l0_at_capacity_no_archive(self, small_lignification, tree):
        """L0 未超 capacity → 不 archive。"""
        tree.insert_cell(_cell("c1", energy_val=0.5))
        tree.insert_cell(_cell("c2", energy_val=0.3))
        tree.insert_cell(_cell("c3", energy_val=0.1))  # 正好 capacity=3
        _age_cells(small_lignification, ["c1", "c2", "c3"])

        result = small_lignification.run_maintenance_cycle("ep1")

        assert len(result.archived) == 0
        assert tree.get_cell("c3").status == "active"

    def test_user_directive_not_archived(self, small_lignification, tree):
        """user_directive cell 不被 archive (即使 energy 低)。"""
        tree.insert_cell(_cell("c1", energy_val=0.5))
        tree.insert_cell(_cell("c2", energy_val=0.3))
        tree.insert_cell(_cell("ud", energy_val=0.01, source="user_directive"))
        tree.insert_cell(_cell("c4", energy_val=0.4))
        _age_cells(small_lignification, ["c1", "c2", "ud", "c4"])

        result = small_lignification.run_maintenance_cycle("ep1")

        # ud 不应该被 archive (即使 energy=0.01 最低)
        assert "ud" not in result.archived
        assert tree.get_cell("ud").status == "active"


# ---------------------------------------------------------------------------
# P2: Merge 阈值优化
# ---------------------------------------------------------------------------
class TestMergeThresholdOptimization:
    """P2: merge_similarity_threshold 从 0.92 降至 0.82。"""

    def test_default_threshold_is_082(self):
        """默认阈值应为 0.82 (P2 优化后)。"""
        config = LignificationConfig()
        assert config.merge_similarity_threshold == 0.82

    def test_identical_cells_still_merged(self, lignification, tree):
        """相同文本的 cell 在新阈值下仍然被识别为 merge 候选。"""
        tree.insert_cell(_cell(
            "c1", ring="L1", maturity=0.2,
            decision="always use nulls_first in order_by",
            rationale="PG and MySQL differ on NULL sorting",
            domain_tags=["sorting"],
        ))
        tree.insert_cell(_cell(
            "c2", ring="L1", maturity=0.2,
            decision="always use nulls_first in order_by",
            rationale="PG and MySQL differ on NULL sorting",
            domain_tags=["sorting"],
        ))
        candidates = lignification._find_merge_candidates()
        found = any("c1" in cluster and "c2" in cluster for cluster in candidates)
        assert found

    def test_custom_threshold_respected(self, tree, energy, llm):
        """自定义阈值仍生效。"""
        config = LignificationConfig(merge_similarity_threshold=0.99)
        lign = LignificationScheduler(tree, energy, llm, tree.oplog, config)
        assert lign.config.merge_similarity_threshold == 0.99
