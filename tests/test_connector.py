"""Connector 测试 —— 对应 docs/specs/connector.md 测试用例。"""
import pytest

from tree_harness.core.cell_model import create_cell, Cell
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.oplog import OpLog
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.connector import Connector, ConnectorConfig


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
def connector(tree):
    return Connector(tree, ConnectorConfig())


def _cell(cid, ring="L0", decision="d", rationale="r", tags=None):
    return create_cell(
        cell_id=cid, ring=ring, decision=decision, rationale=rationale,
        domain_tags=tags or [],
    )


# 测试 1: 新 L0 cell + tree 中有 L1-L3 cell → 建立 ray,方向正确
def test_basic_ray_establishment(connector, tree):
    tree.insert_cell(_cell("c1", "L1", decision="same", rationale="text"))
    tree.insert_cell(_cell("c2", "L2", decision="same", rationale="text"))
    tree.insert_cell(_cell("c3", "L3", decision="same", rationale="text"))

    new_cell = _cell("c0", "L0", decision="same", rationale="text")
    tree.insert_cell(new_cell)

    rays = connector.connect(new_cell)
    ray_targets = {r[0] for r in rays}
    assert ray_targets == {"c1", "c2", "c3"}

    # 验证 ray 方向: c0 → c1/c2/c3 (外→内)
    outgoing = tree.get_outgoing_rays("c0")
    assert len(outgoing) == 3
    for ray in outgoing:
        assert ray["from_id"] == "c0"
        assert ray["to_id"] in {"c1", "c2", "c3"}


# 测试 2: 新 L0 cell + tree 为空 → 返回空列表
def test_empty_tree_no_rays(connector, tree):
    new_cell = _cell("c0", "L0", decision="solo", rationale="cell")
    tree.insert_cell(new_cell)
    rays = connector.connect(new_cell)
    assert rays == []


# 测试 3: 新 L2 cell → 不连接到 L0/L1 (ring 过滤)
def test_ring_filter(connector, tree):
    tree.insert_cell(_cell("c0", "L0", decision="same", rationale="text"))
    tree.insert_cell(_cell("c1", "L1", decision="same", rationale="text"))
    tree.insert_cell(_cell("c3", "L3", decision="same", rationale="text"))
    tree.insert_cell(_cell("c4", "L4", decision="same", rationale="text"))

    new_cell = _cell("new", "L2", decision="same", rationale="text")
    tree.insert_cell(new_cell)

    rays = connector.connect(new_cell)
    ray_targets = {r[0] for r in rays}
    # L2 只能连接到 L2+ → c3, c4 (不包括 c0, c1)
    assert "c3" in ray_targets
    assert "c4" in ray_targets
    assert "c0" not in ray_targets
    assert "c1" not in ray_targets


# 测试 4: domain_tags 重叠 → weight 更高
def test_domain_overlap_weight(connector, tree, monkeypatch):
    from tree_harness.core.cell_model import Cell

    tree.insert_cell(_cell("c1", "L1", decision="d1", rationale="r1",
                           tags=["validation", "safety"]))
    new_cell = _cell("c0", "L0", decision="d2", rationale="r2",
                     tags=["validation", "safety"])

    # 控制 vec_search 返回固定 similarity
    matched = tree.get_cell("c1")
    monkeypatch.setattr(tree, "vec_search",
                        lambda *a, **kw: [(matched, 0.80)])

    # 无 domain overlap 时的 weight
    new_cell_no_overlap = _cell("c0b", "L0", decision="d2", rationale="r2",
                                tags=["other"])
    monkeypatch.setattr(tree, "vec_search",
                        lambda *a, **kw: [(matched, 0.80)])

    # 有 domain overlap
    monkeypatch.setattr(tree, "vec_search",
                        lambda *a, **kw: [(matched, 0.80)])
    rays = connector.connect(new_cell)
    assert len(rays) == 1
    weight_overlap = rays[0][1]

    # 计算预期: 0.80 * (1 + min(2*0.2, 0.4)) = 0.80 * 1.4 = 1.12 → clip 1.0
    assert weight_overlap == pytest.approx(1.0)


# 测试 5: 10 个候选 → 只取 top-5
def test_top_k_selection(connector, tree, monkeypatch):
    from tree_harness.core.cell_model import Cell

    # 插入 10 个 cell
    cells = []
    for i in range(10):
        c = _cell(f"c{i}", "L1", decision=f"d{i}", rationale=f"r{i}")
        tree.insert_cell(c)
        cells.append(tree.get_cell(f"c{i}"))

    new_cell = _cell("new", "L0", decision="new", rationale="new")
    tree.insert_cell(new_cell)

    # 让 vec_search 返回 10 个候选,各有不同 similarity
    def mock_vec_search(*a, **kw):
        return [(c, 0.50 + i * 0.05) for i, c in enumerate(cells)]

    monkeypatch.setattr(tree, "vec_search", mock_vec_search)

    rays = connector.connect(new_cell)
    assert len(rays) <= 5  # max_rays_per_cell = 5


# 测试 6: weight > 1.0 → clip 到 1.0
def test_weight_clip(connector, tree, monkeypatch):
    tree.insert_cell(_cell("c1", "L4", decision="d", rationale="r",
                           tags=["a", "b"]))
    new_cell = _cell("c0", "L0", decision="d2", rationale="r2",
                     tags=["a", "b"])
    tree.insert_cell(new_cell)

    matched = tree.get_cell("c1")
    # similarity=1.0, domain_bonus=0.4 → weight=1.4 → clip 1.0
    monkeypatch.setattr(tree, "vec_search",
                        lambda *a, **kw: [(matched, 1.0)])

    rays = connector.connect(new_cell)
    assert len(rays) == 1
    assert rays[0][1] <= 1.0


# 测试 7: 不连接到自身
def test_no_self_loop(connector, tree):
    new_cell = _cell("c0", "L0", decision="solo", rationale="cell")
    tree.insert_cell(new_cell)
    rays = connector.connect(new_cell)
    # tree 中只有自己,排除自身后无候选
    assert rays == []
