"""Cell 模型测试 —— 对应 docs/specs/cell_model.md 测试用例。"""
import re

import pytest

from tree_harness.core.cell_model import (
    Cell,
    Precondition,
    VerifyHint,
    create_cell,
    generate_cell_id,
    MATURITY_RING_RANGES,
    PROMOTE_THRESHOLDS,
    DEMOTE_THRESHOLDS,
)


# 测试用例 1: 创建标准 cell,验证所有字段有值
def test_create_standard_cell_all_fields_populated():
    precond = Precondition(
        kind="code_invariant",
        assertion="存在跨数据库的排序测试",
        verify_hint=VerifyHint(type="test_id_lookup", params={"test_id": "test_ordering_null"}),
    )
    cell = create_cell(
        trigger_task="django#1234",
        domain="ORM/sorting",
        decision="在所有涉及 ORM order_by 的代码中显式添加 nulls_first=True",
        rationale="PG 和 MySQL 对 NULL 排序的默认行为不同",
        preconditions=[precond],
        evidence=["test_id:test_ordering_null", "commit:abc123"],
        domain_tags=["ORM", "sorting", "cross-db"],
    )
    # id 格式
    assert re.match(r"^cell-\d{14}-[0-9a-f]{6}$", cell.id)
    # 蒸馏 cell 初始值
    assert cell.ring == "L0"
    assert cell.maturity == 0.0
    assert cell.energy == 0.5
    assert cell.status == "active"
    assert cell.source == "distilled"
    # 内容三元组
    assert cell.context_trigger_task == "django#1234"
    assert cell.context_domain == "ORM/sorting"
    assert len(cell.context_preconditions) == 1
    assert cell.context_preconditions[0].verify_hint is not None
    assert cell.decision.startswith("在所有涉及")
    assert cell.rationale.startswith("PG 和 MySQL")
    assert cell.evidence == ["test_id:test_ordering_null", "commit:abc123"]
    assert cell.domain_tags == ["ORM", "sorting", "cross-db"]
    # 元数据
    assert cell.superseded_by is None
    assert cell.created_at is not None


# 测试用例 2: 创建 user_directive cell,验证初始 ring=L4, energy=1.0
def test_create_user_directive_cell_initial_values():
    cell = create_cell(
        source="user_directive",
        decision="统一使用 black 做代码格式化",
        rationale="团队工程约定",
    )
    assert cell.ring == "L4"
    assert cell.maturity == 0.85
    assert cell.energy == 1.0
    assert cell.source == "user_directive"
    assert cell.status == "active"


# 测试用例 3: 验证 ID 格式正确且唯一 (1000 个无重复)
def test_cell_id_unique_and_format():
    ids = [generate_cell_id() for _ in range(1000)]
    # 无重复
    assert len(set(ids)) == 1000
    # 格式正确
    for cid in ids:
        assert re.match(r"^cell-\d{14}-[0-9a-f]{6}$", cid), f"bad id: {cid}"


# 测试用例 4: maturity 超过 1.0 时被截断为 1.0
def test_maturity_clipped_high():
    cell = create_cell(maturity=1.5)
    assert cell.maturity == 1.0
    # 直接构造也截断
    cell2 = Cell(
        id="c1", ring="L0", maturity=2.0, energy=0.5,
        context_trigger_task="", context_domain="",
        context_preconditions=[], decision="d", rationale="r",
        evidence=[], domain_tags=[], status="active", source="distilled",
    )
    assert cell2.maturity == 1.0


# 测试用例 5: maturity 低于 0.0 时被截断为 0.0
def test_maturity_clipped_low():
    cell = create_cell(maturity=-0.3)
    assert cell.maturity == 0.0


# 公理六: id / decision / rationale 创建后不可修改
def test_immutable_fields_cannot_be_modified():
    cell = create_cell(decision="X", rationale="Y")
    with pytest.raises(AttributeError):
        cell.decision = "Z"
    with pytest.raises(AttributeError):
        cell.rationale = "Z2"
    with pytest.raises(AttributeError):
        cell.id = "other"
    # 可变字段仍可更新
    cell.energy = 0.3
    cell.ring = "L1"
    cell.maturity = 0.5
    cell.status = "quarantined"
    assert cell.energy == 0.3
    assert cell.ring == "L1"
    assert cell.status == "quarantined"


def test_ring_thresholds_sanity():
    """阈值单调递增,且 demote 与 promote 间留有 0.10 滞回带。"""
    assert MATURITY_RING_RANGES["L0"][0] == 0.0
    assert MATURITY_RING_RANGES["L4"][1] == 1.0
    # L1 promote 阈值 0.15, demote 阈值 0.05, 差 0.10
    assert PROMOTE_THRESHOLDS["L1"] - DEMOTE_THRESHOLDS["L1"] == pytest.approx(0.10)
    assert PROMOTE_THRESHOLDS["L2"] - DEMOTE_THRESHOLDS["L2"] == pytest.approx(0.10)
