"""SQLiteBackend 测试 —— 对应 docs/specs/sqlite_backend.md 测试用例。"""
import sqlite3

import pytest

from tree_harness.core.cell_model import create_cell, Precondition, VerifyHint
from tree_harness.core.embedding import DeterministicEmbedder, embed_cell_text
from tree_harness.store.sqlite_backend import SQLiteBackend


@pytest.fixture
def embedder():
    return DeterministicEmbedder(dim=32)


@pytest.fixture
def backend(embedder):
    be = SQLiteBackend(":memory:", embedder=embedder)
    yield be
    be.close()


def _make_cell(idx=0, domain="ORM", decision="d", rationale="r", energy=0.5, status="active"):
    cell = create_cell(
        trigger_task=f"task#{idx}",
        domain=domain,
        decision=decision,
        rationale=rationale,
        ring="L0",
        energy=energy,
        cell_id=f"cell-{idx}",
    )
    if status != "active":
        cell.status = status
    return cell


# 测试用例 1: 创建数据库 → 验证表结构存在
def test_init_db_creates_tables(backend):
    tables = {
        r[0] for r in backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table')"
        ).fetchall()
    }
    assert "cells" in tables
    assert "vec_cells" in tables


# 测试用例 2: 插入 cell → get_cell 取回 → 字段一致
def test_insert_and_get_cell(backend):
    precond = Precondition(
        kind="code_invariant",
        assertion="存在跨数据库排序测试",
        verify_hint=VerifyHint(type="test_id_lookup", params={"test_id": "t1"}),
    )
    cell = _make_cell(1, decision="显式添加 nulls_first", rationale="PG MySQL NULL 排序不同")
    cell.context_preconditions = [precond]
    cell.evidence = ["test_id:t1", "commit:abc"]
    cell.domain_tags = ["ORM", "sorting"]
    backend.insert_cell(cell)

    got = backend.get_cell("cell-1")
    assert got is not None
    assert got.id == "cell-1"
    assert got.ring == "L0"
    assert got.decision == "显式添加 nulls_first"
    assert got.rationale == "PG MySQL NULL 排序不同"
    assert got.evidence == ["test_id:t1", "commit:abc"]
    assert got.domain_tags == ["ORM", "sorting"]
    assert len(got.context_preconditions) == 1
    assert got.context_preconditions[0].verify_hint is not None
    assert got.context_preconditions[0].verify_hint.params == {"test_id": "t1"}


# 测试用例 3: 插入重复 ID → 抛出异常
def test_insert_duplicate_id_raises(backend):
    cell = _make_cell(1)
    backend.insert_cell(cell)
    with pytest.raises(sqlite3.IntegrityError):
        backend.insert_cell(_make_cell(1))


# 测试用例 4: update_cell(energy=0.3) → 验证只有 energy 变了
def test_update_cell_only_changes_allowed_fields(backend):
    cell = _make_cell(1, energy=0.5)
    backend.insert_cell(cell)
    backend.update_cell("cell-1", energy=0.3, maturity=0.2, status="quarantined")
    got = backend.get_cell("cell-1")
    assert got.energy == 0.3
    assert got.maturity == 0.2
    assert got.status == "quarantined"
    # decision/rationale 未变
    assert got.decision == "d"


# 测试用例 5: update_cell(decision_text=...) → 抛出异常
def test_update_cell_rejects_immutable_fields(backend):
    backend.insert_cell(_make_cell(1))
    with pytest.raises(ValueError):
        backend.update_cell("cell-1", decision_text="changed")
    with pytest.raises(ValueError):
        backend.update_cell("cell-1", rationale_text="changed")
    with pytest.raises(ValueError):
        backend.update_cell("cell-1", context_domain="other")


# 测试用例 6: 插入 5 个不同 domain 的 cell → query_by_domain 只返回匹配的
def test_query_by_domain(backend):
    for i, dom in enumerate(["ORM", "Auth", "ORM", "Cache", "Auth"]):
        backend.insert_cell(_make_cell(i, domain=dom))
    orm = backend.query_by_domain("ORM")
    assert len(orm) == 2
    assert all(c.context_domain == "ORM" for c in orm)
    assert len(backend.query_by_domain("None")) == 0


# 测试用例 7: 插入 10 个 cell → vec_search 返回按相似度排序的 top-5
def test_vec_search_returns_topk_sorted(backend, embedder):
    # 第 0 个用独特文本,其余用不同文本
    backend.insert_cell(_make_cell(0, decision="unique alpha decision", rationale="reason alpha"))
    for i in range(1, 10):
        backend.insert_cell(_make_cell(i, decision=f"other {i}", rationale=f"r{i}"))
    q = embedder.embed(embed_cell_text("unique alpha decision", "reason alpha"))
    results = backend.vec_search(q, top_k=5, threshold=0.0)
    assert len(results) == 5
    # 最相似的应是 cell-0 (完全匹配, distance≈0)
    assert results[0][0].id == "cell-0"
    # similarity 降序
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)


# 测试用例 8: quarantined cell 不出现在 vec_search 结果中
def test_vec_search_excludes_quarantined(backend, embedder):
    backend.insert_cell(_make_cell(0, decision="target text", rationale="r"))
    backend.insert_cell(_make_cell(1, decision="target text", rationale="r"))
    # 把 cell-1 隔离
    backend.update_cell("cell-1", status="quarantined")
    q = embedder.embed(embed_cell_text("target text", "r"))
    results = backend.vec_search(q, top_k=10, threshold=0.0)
    ids = [c.id for c, _ in results]
    assert "cell-0" in ids
    assert "cell-1" not in ids


# 测试用例 9: query_decay_candidates 正确识别 energy < threshold 的 cell
def test_query_decay_candidates(backend):
    backend.insert_cell(_make_cell(0, energy=0.5))
    backend.insert_cell(_make_cell(1, energy=-0.3))
    backend.insert_cell(_make_cell(2, energy=-0.5))
    # quarantined 的不应算候选
    backend.insert_cell(_make_cell(3, energy=-0.9))
    backend.update_cell("cell-3", status="quarantined")

    candidates = backend.query_decay_candidates(energy_threshold=-0.20)
    ids = {c.id for c in candidates}
    assert ids == {"cell-1", "cell-2"}


def test_count_cells(backend):
    for i in range(5):
        backend.insert_cell(_make_cell(i))
    backend.update_cell("cell-0", status="quarantined")
    assert backend.count_cells() == 5
    assert backend.count_cells(status="active") == 4
    assert backend.count_cells(ring="L0", status="active") == 4
