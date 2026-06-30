"""Dedup 测试 —— 对应 docs/specs/dedup.md 测试用例。"""
import json

import pytest

from tree_harness.core.cell_model import create_cell, CandidateCell
from tree_harness.core.embedding import DeterministicEmbedder, embed_cell_text
from tree_harness.core.llm_client import DeterministicLLMClient
from tree_harness.core.oplog import OpLog
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.dedup import Dedup, DedupConfig


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
def llm():
    return DeterministicLLMClient()


@pytest.fixture
def dedup(tree, llm):
    return Dedup(tree, llm, DedupConfig())


def _candidate(decision="test decision", rationale="test rationale"):
    return CandidateCell(
        decision=decision,
        rationale=rationale,
        preconditions=[],
        evidence=[],
        domain_tags=["test"],
        embedding=[0.1] * 32,  # dummy; overridden in real use
    )


# 测试 1: tree 为空 → INSERT_NEW
def test_empty_tree_insert_new(dedup, tree):
    candidate = _candidate()
    candidate.embedding = tree.sqlite.embedder.embed(
        embed_cell_text(candidate.decision, candidate.rationale)
    )
    result = dedup.check(candidate)
    assert result.action == "INSERT_NEW"
    assert result.matched_cell_id is None


# 测试 2: 完全相同 → score > 0.95 → REINFORCE
def test_exact_match_reinforce(dedup, tree):
    tree.insert_cell(create_cell(
        cell_id="c1", decision="test decision", rationale="test rationale",
    ))
    candidate = _candidate()  # 相同文本
    candidate.embedding = tree.sqlite.embedder.embed(
        embed_cell_text(candidate.decision, candidate.rationale)
    )
    result = dedup.check(candidate)
    assert result.action == "REINFORCE"
    assert result.matched_cell_id == "c1"
    assert result.similarity_score > 0.95


# 测试 3: 灰区 LLM 判 same → REINFORCE
def test_gray_zone_llm_same(dedup, tree, llm, monkeypatch):
    tree.insert_cell(create_cell(
        cell_id="c1", decision="existing", rationale="existing rationale",
    ))
    candidate = _candidate(decision="new variant", rationale="new rationale")
    matched_cell = tree.get_cell("c1")
    # 控制 vec_search 返回灰区分值
    monkeypatch.setattr(tree, "vec_search", lambda *a, **kw: [(matched_cell, 0.90)])
    llm.inject("dedup", json.dumps({"verdict": "same", "reason": "same decision"}))

    result = dedup.check(candidate)
    assert result.action == "REINFORCE"
    assert result.matched_cell_id == "c1"
    assert result.reason == "llm_arbitrate_same"


# 测试 4: 灰区 LLM 判 different → INSERT_NEW
def test_gray_zone_llm_different(dedup, tree, llm, monkeypatch):
    tree.insert_cell(create_cell(
        cell_id="c1", decision="existing", rationale="existing rationale",
    ))
    candidate = _candidate(decision="new variant", rationale="new rationale")
    matched_cell = tree.get_cell("c1")
    monkeypatch.setattr(tree, "vec_search", lambda *a, **kw: [(matched_cell, 0.90)])
    llm.inject("dedup", json.dumps({"verdict": "different", "reason": "different context"}))

    result = dedup.check(candidate)
    assert result.action == "INSERT_NEW"
    assert result.reason == "llm_arbitrate_different"


# 测试 5: score < 0.85 → INSERT_NEW
def test_below_threshold_insert_new(dedup, tree, monkeypatch):
    tree.insert_cell(create_cell(
        cell_id="c1", decision="existing", rationale="existing rationale",
    ))
    candidate = _candidate(decision="completely different", rationale="totally unrelated")
    matched_cell = tree.get_cell("c1")
    monkeypatch.setattr(tree, "vec_search", lambda *a, **kw: [(matched_cell, 0.50)])

    result = dedup.check(candidate)
    assert result.action == "INSERT_NEW"
    assert result.reason == "below_threshold"


# 测试 6: quarantined cell 不参与去重
def test_quarantined_not_matched(dedup, tree):
    tree.insert_cell(create_cell(
        cell_id="c1", decision="test decision", rationale="test rationale",
    ))
    tree.quarantine("c1", "decayed", "ep1")
    candidate = _candidate()
    candidate.embedding = tree.sqlite.embedder.embed(
        embed_cell_text(candidate.decision, candidate.rationale)
    )
    result = dedup.check(candidate)
    assert result.action == "INSERT_NEW"


# 测试 7: DedupResult 携带字段
def test_result_carries_fields(dedup, tree, monkeypatch):
    tree.insert_cell(create_cell(
        cell_id="c1", decision="existing", rationale="existing",
    ))
    candidate = _candidate()
    matched_cell = tree.get_cell("c1")
    monkeypatch.setattr(tree, "vec_search", lambda *a, **kw: [(matched_cell, 0.92)])

    result = dedup.check(candidate)
    assert result.similarity_score == 0.92
    assert result.matched_cell_id == "c1"
