"""TreeStore 测试 —— 对应 docs/specs/tree_store.md 测试用例。"""
import pytest

from tree_harness.core.cell_model import create_cell
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.oplog import OpLog, OpEnum
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore


@pytest.fixture
def embedder():
    return DeterministicEmbedder(dim=32)


@pytest.fixture
def tree(tmp_path, embedder):
    sqlite = SQLiteBackend(":memory:", embedder=embedder)
    kuzu = KuzuBackend(str(tmp_path / "kuzu"))
    oplog = OpLog(str(tmp_path / "oplog.db"))
    yield TreeStore(sqlite, kuzu, oplog)


def _cell(cid, ring="L0", decision="d", rationale="r"):
    return create_cell(cell_id=cid, ring=ring, decision=decision, rationale=rationale)


# 测试用例 1: insert_cell → SQLite 和 KuzuDB 中都能查到
def test_insert_cell_both_backends(tree):
    tree.insert_cell(_cell("c1", "L0"), episode_id="ep1")
    assert tree.get_cell("c1") is not None
    assert tree.kuzu.get_ring("c1") == "L0"


# 测试用例 2: insert_cell 带 rays → ray 正确建立在 KuzuDB 中
def test_insert_cell_with_rays(tree):
    tree.insert_cell(_cell("t1", "L1"))
    tree.insert_cell(_cell("c1", "L0"), rays=[("t1", 0.5)], episode_id="ep1")
    outgoing = tree.get_outgoing_rays("c1")
    assert len(outgoing) == 1
    assert outgoing[0]["to_id"] == "t1"
    assert tree.get_in_degree("t1") == 1


# 测试用例 3: quarantine → cell status 变更 + 外向 ray 全部 severed
def test_quarantine_severs_outgoing_rays(tree):
    tree.insert_cell(_cell("t1", "L1"))
    tree.insert_cell(_cell("c1", "L0"), rays=[("t1", 0.5)])
    tree.quarantine("c1", "decayed", "ep2")
    assert tree.get_cell("c1").status == "quarantined"
    outgoing = tree.get_outgoing_rays("c1")
    assert all(r["status"] == "severed" for r in outgoing)


# 测试用例 4: merge_cells → source 变 superseded + merged_cell 存在
def test_merge_cells(tree):
    tree.insert_cell(_cell("s1", "L1"))
    tree.insert_cell(_cell("s2", "L1"))
    merged = _cell("m1", "L2", decision="merged")
    tree.merge_cells(["s1", "s2"], merged, episode_id="ep1")
    assert tree.get_cell("s1").status == "superseded"
    assert tree.get_cell("s1").superseded_by == "m1"
    assert tree.get_cell("s2").superseded_by == "m1"
    assert tree.get_cell("m1") is not None
    assert tree.kuzu.get_ring("m1") == "L2"


# merge 后 incoming ray 重定向到 merged_cell
def test_merge_redirects_incoming_rays(tree):
    tree.insert_cell(_cell("inner", "L3"))
    tree.insert_cell(_cell("s1", "L1"), rays=[("inner", 0.5)])
    tree.insert_cell(_cell("outer", "L0"), rays=[("s1", 0.6)])  # outer→s1
    merged = _cell("m1", "L2", decision="merged")
    tree.merge_cells(["s1"], merged, episode_id="ep1")
    # outer 的出射 ray 现在应指向 m1 (重定向)
    outgoing = tree.get_outgoing_rays("outer")
    targets = {r["to_id"] for r in outgoing}
    assert "m1" in targets


# merge 后 source 的 outgoing active ray 被 sever (P0-Bug1)
def test_merge_severs_source_outgoing_rays(tree):
    tree.insert_cell(_cell("inner", "L3"))
    tree.insert_cell(_cell("s1", "L1"), rays=[("inner", 0.5)])  # s1 → inner
    merged = _cell("m1", "L2", decision="merged")
    tree.merge_cells(["s1"], merged, episode_id="ep1")
    # s1 被 supersede 后, 其指向 inner 的 active ray 应被 severed
    outgoing = tree.get_outgoing_rays("s1")
    assert len(outgoing) == 1
    assert outgoing[0]["status"] == "severed"


# split 后 source 的 outgoing active ray 被 sever (P0-Bug1)
def test_split_severs_source_outgoing_rays(tree):
    tree.insert_cell(_cell("inner", "L3"))
    tree.insert_cell(_cell("s1", "L1"), rays=[("inner", 0.5)])  # s1 → inner
    child1 = _cell("c1", "L0", decision="child1")
    child2 = _cell("c2", "L0", decision="child2")
    tree.split_cell("s1", [child1, child2], episode_id="ep1")
    # s1 被 supersede 后, 其指向 inner 的 active ray 应被 severed
    outgoing = tree.get_outgoing_rays("s1")
    assert all(r["status"] == "severed" for r in outgoing)


# 测试用例 5: promote → SQLite ring 字段和 KuzuDB ring 字段同步变更
def test_promote_syncs_both(tree):
    tree.insert_cell(_cell("c1", "L0"))
    tree.promote("c1", "L0", "L1", "ep1")
    assert tree.get_cell("c1").ring == "L1"
    assert tree.kuzu.get_ring("c1") == "L1"


def test_demote_syncs_both(tree):
    tree.insert_cell(_cell("c1", "L2"))
    tree.demote("c1", "L2", "L1", "ep1")
    assert tree.get_cell("c1").ring == "L1"
    assert tree.kuzu.get_ring("c1") == "L1"


# 测试用例 6: oplog 记录完整
def test_oplog_complete(tree):
    tree.insert_cell(_cell("t1", "L1"))
    tree.insert_cell(_cell("c1", "L0"), rays=[("t1", 0.5)], episode_id="ep1")
    ops = [e.op for e in tree.oplog.get_entries()]
    assert ops.count(OpEnum.INSERT_CELL) == 2
    assert ops.count(OpEnum.INSERT_RAY) == 1


# 测试用例 7: consistency_check 在正常状态下返回空列表
def test_consistency_check_clean(tree):
    tree.insert_cell(_cell("c1", "L0"))
    assert tree.consistency_check() == []


# 测试用例 8: 手动删除 KuzuDB 中一个节点 → consistency_check 检出不一致
def test_consistency_check_detects_missing_kuzu(tree):
    tree.insert_cell(_cell("c1", "L0"))
    tree.kuzu.remove_cell_ref("c1")
    assert "c1" in tree.consistency_check()


def test_consistency_check_detects_ring_mismatch(tree):
    tree.insert_cell(_cell("c1", "L0"))
    tree.kuzu.update_cell_ring("c1", "L3")  # KuzuDB 侧改了,SQLite 没改
    assert "c1" in tree.consistency_check()


# 测试用例 9: stats 返回正确的统计数字
def test_stats(tree):
    tree.insert_cell(_cell("c1", "L0"))
    tree.insert_cell(_cell("c2", "L1"), rays=[("c1", 0.5)])
    s = tree.stats()
    assert s["total_cells"] == 2
    assert s["by_ring"]["L0"] == 1
    assert s["by_ring"]["L1"] == 1
    assert s["total_rays"] == 1
    assert s["active_rays"] == 1


# replay 从空库重建 → 与直接写入结果一致
def test_replay_rebuilds(tree, tmp_path, embedder):
    tree.insert_cell(_cell("t1", "L1"))
    tree.insert_cell(_cell("c1", "L0"), rays=[("t1", 0.5)], episode_id="ep1")
    tree.update_energy("c1", 0.8, "ref", "ep1")
    tree.promote("c1", "L0", "L1", "ep1")

    # 新空库,从同一 oplog 文件 replay
    sqlite2 = SQLiteBackend(":memory:", embedder=embedder)
    kuzu2 = KuzuBackend(str(tmp_path / "kuzu2"))
    oplog2 = OpLog(str(tmp_path / "oplog.db"))
    ts2 = TreeStore(sqlite2, kuzu2, oplog2)
    oplog2.replay(ts2)

    c = ts2.get_cell("c1")
    assert c is not None
    assert c.energy == 0.8
    assert c.ring == "L1"
    assert len(ts2.get_outgoing_rays("c1")) == 1
    assert ts2.get_cell("t1") is not None
    assert ts2.consistency_check() == []


# consistency_check 检测非 active cell 的不一致 (P1-Bug5)
def test_consistency_check_scans_quarantined(tree):
    tree.insert_cell(_cell("c1", "L0"))
    tree.quarantine("c1", "decayed", "ep1")
    assert tree.get_cell("c1").status == "quarantined"
    # 手动从 KuzuDB 删除 quarantined cell
    tree.kuzu.remove_cell_ref("c1")
    # 即使不是 active, consistency_check 也应检出
    assert "c1" in tree.consistency_check()


# repair() 修复 KuzuDB 缺失节点 (P1-Bug3)
def test_repair_fixes_missing_kuzu(tree):
    tree.insert_cell(_cell("c1", "L0"))
    tree.kuzu.remove_cell_ref("c1")
    assert "c1" in tree.consistency_check()
    repaired = tree.repair()
    assert "c1" in repaired
    assert tree.consistency_check() == []


# repair() 修复 ring 不一致 (P1-Bug3)
def test_repair_fixes_ring_mismatch(tree):
    tree.insert_cell(_cell("c1", "L0"))
    tree.kuzu.update_cell_ring("c1", "L3")  # KuzuDB 侧 ring 错误
    assert "c1" in tree.consistency_check()
    repaired = tree.repair()
    assert "c1" in repaired
    assert tree.kuzu.get_ring("c1") == "L0"
    assert tree.consistency_check() == []


# repair() 修复 KuzuDB 孤儿节点 (P1-Bug3)
def test_repair_removes_orphan_kuzu(tree):
    tree.insert_cell(_cell("c1", "L0"))
    tree.kuzu.add_cell_ref("orphan", "L2")  # KuzuDB 有但 SQLite 没有
    assert "orphan" in tree.consistency_check()
    repaired = tree.repair()
    assert "orphan" in repaired
    assert tree.kuzu.get_ring("orphan") is None
    assert tree.consistency_check() == []


# list_by_ring 批量查询多个 ring (outer_harness before_step pinned 段用)
def test_list_by_ring_batch(tree):
    tree.insert_cell(_cell("c0", "L0"))
    tree.insert_cell(_cell("c1", "L1"))
    tree.insert_cell(_cell("c3", "L3"))
    tree.insert_cell(_cell("c4", "L4"))
    pinned = tree.list_by_ring(["L3", "L4"], status="active")
    ids = {c.id for c in pinned}
    assert ids == {"c3", "c4"}


# list_by_ring 空列表 → 返回空
def test_list_by_ring_empty(tree):
    tree.insert_cell(_cell("c1", "L0"))
    assert tree.list_by_ring([]) == []


# list_by_ring 排除非 active cell
def test_list_by_ring_excludes_inactive(tree):
    tree.insert_cell(_cell("c3", "L3"))
    tree.quarantine("c3", "decayed", "ep1")
    result = tree.list_by_ring(["L3"], status="active")
    assert result == []
    # 但查 quarantined 能查到
    result = tree.list_by_ring(["L3"], status="quarantined")
    assert len(result) == 1
