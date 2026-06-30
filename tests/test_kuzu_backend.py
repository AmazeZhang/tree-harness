"""KuzuBackend 测试 —— 对应 docs/specs/kuzu_backend.md 测试用例。"""
import pytest

from tree_harness.store.kuzu_backend import KuzuBackend


@pytest.fixture
def kb(tmp_path):
    backend = KuzuBackend(str(tmp_path / "kuzu_db"))
    yield backend


# 测试用例 1: 初始化 DB → schema 正确创建
def test_init_schema(kb):
    kb.add_cell_ref("c1", "L0")
    assert kb.get_ring("c1") == "L0"


# 测试用例 2: add_cell_ref + add_ray → get_outgoing_rays 正确返回
def test_add_ray_and_outgoing(kb):
    kb.add_cell_ref("a", "L0")
    kb.add_cell_ref("b", "L1")
    kb.add_ray("a", "b", 0.5)
    outgoing = kb.get_outgoing_rays("a")
    assert len(outgoing) == 1
    assert outgoing[0]["to_id"] == "b"
    assert outgoing[0]["from_id"] == "a"
    assert outgoing[0]["weight"] == 0.5
    assert outgoing[0]["status"] == "active"


# 测试用例 3: add_ray → get_incoming_rays 从被引用方能查到
def test_incoming_rays(kb):
    kb.add_cell_ref("a", "L0")
    kb.add_cell_ref("b", "L1")
    kb.add_ray("a", "b", 0.5)
    incoming = kb.get_incoming_rays("b")
    assert len(incoming) == 1
    assert incoming[0]["from_id"] == "a"
    assert incoming[0]["to_id"] == "b"


# 测试用例 4: get_in_degree 正确计数 (排除 severed ray)
def test_in_degree_excludes_severed(kb):
    kb.add_cell_ref("a", "L0")
    kb.add_cell_ref("b", "L0")
    kb.add_cell_ref("c", "L2")
    kb.add_ray("a", "c", 0.5)
    kb.add_ray("b", "c", 0.5)
    assert kb.get_in_degree("c") == 2
    kb.update_ray_status("a", "c", "severed")
    assert kb.get_in_degree("c") == 1
    # 全状态计数
    assert kb.get_in_degree("c", status="severed") == 1


# 测试用例 5: update_ray_weight 超过 1.0 → clip 到 1.0
def test_ray_weight_clip(kb):
    kb.add_cell_ref("a", "L0")
    kb.add_cell_ref("b", "L1")
    kb.add_ray("a", "b", 1.5)
    rays = kb.get_outgoing_rays("a")
    assert rays[0]["weight"] == 1.0
    kb.update_ray_weight("a", "b", -0.5)
    rays = kb.get_outgoing_rays("a")
    assert rays[0]["weight"] == 0.0


# 测试用例 6: add_supersedes → get_supersede_chain 返回完整链
def test_supersede_chain(kb):
    kb.add_cell_ref("v1", "L0")
    kb.add_cell_ref("v2", "L1")
    kb.add_cell_ref("v3", "L2")
    kb.add_supersedes("v2", "v1")
    kb.add_supersedes("v3", "v2")
    chain = kb.get_supersede_chain("v3")
    assert chain == ["v3", "v2", "v1"]


# 测试用例 7: find_orphans 正确识别无连接节点
def test_find_orphans(kb):
    kb.add_cell_ref("a", "L0")
    kb.add_cell_ref("b", "L1")
    kb.add_cell_ref("c", "L0")  # 孤立
    kb.add_ray("a", "b", 0.5)
    orphans = set(kb.find_orphans())
    assert "c" in orphans
    assert "a" not in orphans
    assert "b" not in orphans


# 测试用例 8: 添加 3 个 cell 形成链 A→B→C → find_path(A,C) 返回 [A,B,C]
def test_find_path(kb):
    kb.add_cell_ref("a", "L0")
    kb.add_cell_ref("b", "L1")
    kb.add_cell_ref("c", "L2")
    kb.add_ray("a", "b", 0.5)
    kb.add_ray("b", "c", 0.5)
    path = kb.find_path("a", "c")
    assert path == ["a", "b", "c"]
    # 反向无出射路径
    assert kb.find_path("c", "a") is None
    # 自身
    assert kb.find_path("a", "a") == ["a"]


def test_connected_component(kb):
    kb.add_cell_ref("a", "L0")
    kb.add_cell_ref("b", "L1")
    kb.add_cell_ref("c", "L2")
    kb.add_cell_ref("d", "L0")  # 独立
    kb.add_ray("a", "b", 0.5)
    kb.add_ray("b", "c", 0.5)
    comp = set(kb.get_connected_component("a"))
    assert comp == {"a", "b", "c"}
    assert "d" not in comp


def test_add_ray_idempotent(kb):
    """add_ray 幂等: 已存在的 ray 不重建,保留原始 weight 和时间戳。"""
    kb.add_cell_ref("a", "L0")
    kb.add_cell_ref("b", "L1")
    kb.add_ray("a", "b", 0.5)
    # 记录原始时间戳
    rays_before = kb.get_outgoing_rays("a")
    ts_before = rays_before[0]["created_at"]
    # 重复添加不同 weight — 不应覆盖
    kb.add_ray("a", "b", 0.8)
    outgoing = kb.get_outgoing_rays("a")
    assert len(outgoing) == 1
    assert outgoing[0]["weight"] == 0.5  # 保留原始 weight
    assert outgoing[0]["created_at"] == ts_before  # 保留原始时间戳
    # 如需更新 weight, 应显式调用 update_ray_weight
    kb.update_ray_weight("a", "b", 0.8)
    outgoing2 = kb.get_outgoing_rays("a")
    assert outgoing2[0]["weight"] == 0.8
