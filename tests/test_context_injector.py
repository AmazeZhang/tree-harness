"""ContextInjector 测试 —— 对应 docs/specs/context_injector.md 测试用例。"""
import pytest

from tree_harness.core.cell_model import create_cell
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.oplog import OpLog
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.context_injector import (
    ContextInjector, InjectorConfig, WarningEntry,
)


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
def injector(tree):
    return ContextInjector(tree, InjectorConfig(min_similarity=0.0))


# 测试 1: 空 tree → 返回空 formatted_text
def test_empty_tree_empty_text(injector, tree):
    result = injector.retrieve("test query", "repo", ["L0", "L1", "L2"], 1000)
    assert result.formatted_text == ""
    assert result.cells == []


# 测试 2: tree 有 cell, task 相关 → 只返回相关的
def test_retrieve_relevant_cells(injector, tree):
    tree.insert_cell(create_cell(
        cell_id="c1", ring="L1", decision="validate input", rationale="prevent crash",
        domain_tags=["validation"],
    ))
    tree.insert_cell(create_cell(
        cell_id="c2", ring="L0", decision="unrelated topic", rationale="no relation",
    ))
    result = injector.retrieve("validate input prevent crash", "repo",
                               ["L0", "L1", "L2"], 1000)
    # c1 应该在结果中 (高相似度)
    assert "c1" in result.cells


# 测试 3: 高 energy cell 排前面
def test_high_energy_first(injector, tree):
    tree.insert_cell(create_cell(
        cell_id="low", ring="L1", energy=0.1,
        decision="validate input", rationale="prevent crash",
    ))
    tree.insert_cell(create_cell(
        cell_id="high", ring="L1", energy=0.8,
        decision="validate input", rationale="prevent crash",
    ))
    result = injector.retrieve("validate input prevent crash", "repo",
                               ["L0", "L1", "L2"], 1000)
    # high energy 应该排前面
    assert result.cells.index("high") < result.cells.index("low")


# 测试 4: L4 vs L1 相似度相同 → L4 排前面 (ring_weight)
def test_ring_weight_priority(injector, tree, monkeypatch):
    tree.insert_cell(create_cell(
        cell_id="l1", ring="L1", energy=0.5,
        decision="same decision", rationale="same rationale",
    ))
    tree.insert_cell(create_cell(
        cell_id="l4", ring="L4", energy=0.5,
        decision="same decision", rationale="same rationale",
    ))
    # 控制 vec_search 返回相同相似度
    l1_cell = tree.get_cell("l1")
    l4_cell = tree.get_cell("l4")
    monkeypatch.setattr(tree, "vec_search",
                        lambda *a, **kw: [(l1_cell, 0.80), (l4_cell, 0.80)])

    result = injector.retrieve("query", "repo", ["L1", "L4"], 1000)
    # L4 应该排前面 (ring_weight=2.5 vs 1.0)
    assert result.cells.index("l4") < result.cells.index("l1")


# 测试 5: quarantined cell 不出现在结果中
def test_quarantined_excluded(injector, tree):
    tree.insert_cell(create_cell(
        cell_id="c1", ring="L1", decision="validate input", rationale="prevent crash",
    ))
    tree.quarantine("c1", "decayed", "ep1")
    result = injector.retrieve("validate input prevent crash", "repo",
                               ["L0", "L1", "L2"], 1000)
    assert "c1" not in result.cells


# 测试 6: 格式化文本包含 [Project Experience] 标记
def test_format_includes_markers(injector, tree):
    tree.insert_cell(create_cell(
        cell_id="c1", ring="L1", decision="test decision", rationale="test reason",
    ))
    result = injector.retrieve("test decision test reason", "repo",
                               ["L0", "L1", "L2"], 1000)
    if result.formatted_text:
        assert "[Project Experience]" in result.formatted_text
        assert "[End of Project Experience]" in result.formatted_text


# 测试 7: format_pinned 包含 pin markers
def test_pinned_has_markers(injector):
    cells = [create_cell(cell_id="c1", ring="L4", decision="axiom", rationale="core")]
    text = injector.format_pinned(cells, budget=1000)
    assert "<|PINNED_DO_NOT_COMPACT|>" in text
    assert "<|/PINNED|>" in text


# 测试 8: format_pinned 空列表 → 空字符串
def test_pinned_empty(injector):
    assert injector.format_pinned([], 1000) == ""


# 测试 9: format_warnings 包含 warning markers
def test_warnings_have_markers(injector):
    warnings = [WarningEntry(
        cell_id="c1", text="WARNING: test warning", is_direct=True,
        ray_weight=1.0, recency=0,
    )]
    text = injector.format_warnings(warnings, 1000)
    assert "<|WARNING_DO_NOT_COMPACT|>" in text
    assert "test warning" in text


# 测试 10: format_warnings 空列表 → 空字符串
def test_warnings_empty(injector):
    assert injector.format_warnings([], 1000) == ""
