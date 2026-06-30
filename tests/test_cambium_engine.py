"""CambiumEngine 测试 —— 对应 docs/specs/cambium_engine.md 测试用例。"""
import json

import pytest

from tree_harness.core.cell_model import (
    create_cell, StandardStep, StandardTrajectory,
)
from tree_harness.core.embedding import DeterministicEmbedder
from tree_harness.core.llm_client import DeterministicLLMClient
from tree_harness.core.oplog import OpLog
from tree_harness.store.sqlite_backend import SQLiteBackend
from tree_harness.store.kuzu_backend import KuzuBackend
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem, EnergyConfig
from tree_harness.modules.cambium_engine import CambiumEngine, CambiumConfig


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


@pytest.fixture
def llm():
    client = DeterministicLLMClient()
    client.inject("crystallize", json.dumps({
        "decision": "Always validate input before processing",
        "rationale": "Unvalidated input caused crash in step 3",
        "preconditions": [],
        "evidence": ["test_id:test_validation"],
        "domain_tags": ["validation", "safety"],
    }))
    return client


@pytest.fixture
def cambium(tree, energy, llm):
    return CambiumEngine(tree, energy, llm, CambiumConfig())


def _step(
    task_id="django__django-16379",
    episode_id="ep-001",
    step_index=0,
    repo="django/django",
    action="Edited compiler.py to add nulls_first parameter",
    observation="Test test_ordering_null passed",
    patch="diff --git a/compiler.py",
    tests=None,
    outcome="pass",
):
    return StandardStep(
        task_id=task_id,
        episode_id=episode_id,
        step_index=step_index,
        repo=repo,
        action_summary=action,
        observation_summary=observation,
        patch_delta=patch,
        test_results=tests if tests is not None else {"test_ordering_null": "pass"},
        outcome_so_far=outcome,
        duration_seconds=12.5,
        token_usage=500,
    )


# ---------------------------------------------------------------------------
# should_crystallize 准入判断
# ---------------------------------------------------------------------------
def test_should_not_crystallize_error(cambium):
    step = _step(outcome="error")
    assert cambium.should_crystallize(step) is False


def test_should_not_crystallize_mechanical(cambium):
    step = _step(action="ls -la", outcome="pass")
    assert cambium.should_crystallize(step) is False


def test_should_not_crystallize_pending_no_changes(cambium):
    step = _step(action="ran search", patch=None, tests={}, outcome="pending")
    assert cambium.should_crystallize(step) is False


def test_should_crystallize_valid(cambium):
    step = _step()
    assert cambium.should_crystallize(step) is True


# ---------------------------------------------------------------------------
# crystallize_step
# ---------------------------------------------------------------------------
def test_crystallize_step_produces_cell(cambium, tree):
    step = _step()
    new_cells = cambium.crystallize_step(step)
    assert len(new_cells) == 1
    cell = new_cells[0]
    assert cell.decision == "Always validate input before processing"
    assert cell.rationale == "Unvalidated input caused crash in step 3"
    assert cell.ring == "L0"
    assert cell.source == "distilled"
    assert cell.status == "active"
    # cell 在 tree 中
    assert tree.get_cell(cell.id) is not None


def test_crystallize_step_connect_rays(cambium, tree):
    # 先放一个已有 cell
    tree.insert_cell(create_cell(
        cell_id="existing", ring="L3",
        decision="Always validate input before processing",
        rationale="Unvalidated input caused crash in step 3",
    ))
    step = _step()
    new_cells = cambium.crystallize_step(step)
    # 由于 decision/rationale 相同 → dedup 会 REINFORCE,不新建
    assert len(new_cells) == 0


def test_crystallize_step_reinforce_increases_energy(cambium, tree, energy):
    # 先放一个已有 cell,初始 energy=0.5
    tree.insert_cell(create_cell(
        cell_id="c1", ring="L3", energy=0.5,
        decision="Always validate input before processing",
        rationale="Unvalidated input caused crash in step 3",
    ))
    step = _step()
    # 第一次: REINFORCE,不新建 cell
    new_cells = cambium.crystallize_step(step)
    assert len(new_cells) == 0
    # energy 应增加
    cell = tree.get_cell("c1")
    assert cell.energy > 0.5  # δ_reference = 0.10 → 0.60


def test_crystallize_step_twice_second_reinforces(cambium, tree):
    step = _step()
    # 第一次: tree 空 → INSERT_NEW
    first = cambium.crystallize_step(step)
    assert len(first) == 1

    # 第二次: 相同 LLM 响应 → 相同 embedding → REINFORCE
    second = cambium.crystallize_step(step)
    assert len(second) == 0  # 不新建

    # tree 中仍然只有 1 个 cell
    all_cells = tree.sqlite.list_active()
    assert len(all_cells) == 1


# ---------------------------------------------------------------------------
# crystallize (trajectory-level batch)
# ---------------------------------------------------------------------------
def test_crystallize_trajectory_error(cambium):
    traj = StandardTrajectory(
        task_id="t1", task_description="Fix bug", repo="r",
        base_commit="abc", outcome="error",
        patches=[], test_results={},
        key_actions=["something"], duration_seconds=10.0, token_usage=100,
        steps=[_step(outcome="error")],
    )
    result = cambium.crystallize(traj)
    assert result == []


def test_crystallize_trajectory_mechanical(cambium):
    traj = StandardTrajectory(
        task_id="t1", task_description="Fix bug", repo="r",
        base_commit="abc", outcome="pass",
        patches=["diff"], test_results={"t1": "pass"},
        key_actions=[],  # 空 key_actions → mechanical
        duration_seconds=10.0, token_usage=100,
        steps=[],
    )
    result = cambium.crystallize(traj)
    assert result == []


def test_crystallize_trajectory_valid(cambium, tree):
    step = _step()
    traj = StandardTrajectory(
        task_id="t1", task_description="Fix bug", repo="r",
        base_commit="abc", outcome="pass",
        patches=["diff"], test_results={"t1": "pass"},
        key_actions=["edited compiler.py"],
        duration_seconds=10.0, token_usage=100,
        steps=[step],
    )
    result = cambium.crystallize(traj)
    assert len(result) == 1
    assert tree.get_cell(result[0].id) is not None


# ---------------------------------------------------------------------------
# connect_new_cells
# ---------------------------------------------------------------------------
def test_connect_new_cells(cambium, tree):
    # 先放几个已有 cell (同文本 → 高相似度 → 会建 ray)
    tree.insert_cell(create_cell(
        cell_id="c1", ring="L3",
        decision="Always validate input before processing",
        rationale="Unvalidated input caused crash in step 3",
    ))
    step = _step()
    new_cells = cambium.crystallize_step(step)
    # 由于和 c1 相同 → REINFORCE,不新建
    # 改用不同文本来测试 connect
    # 直接测 connect_new_cells
    assert len(new_cells) == 0  # REINFORCE


def test_connect_new_cells_batch(cambium, tree):
    # 放已有 cell (不同文本 → INSERT_NEW → 新 cell → connect)
    tree.insert_cell(create_cell(
        cell_id="target", ring="L3",
        decision="some existing knowledge", rationale="some reason",
        domain_tags=["validation"],
    ))
    # 用不同 LLM 响应确保 INSERT_NEW
    llm = cambium.llm_client
    llm.inject("crystallize", json.dumps({
        "decision": "A different decision entirely",
        "rationale": "A different rationale",
        "preconditions": [],
        "evidence": [],
        "domain_tags": ["validation"],
    }))

    step = _step()
    new_cells = cambium.crystallize_step(step)
    assert len(new_cells) == 1

    # connect_new_cells 批量再连边
    cambium.connect_new_cells(new_cells)

    # 检查 ray 是否建立 (如果相似度够高的话)
    outgoing = tree.get_outgoing_rays(new_cells[0].id)
    # 可能 0 条 (如果相似度低于 threshold) 也可能 >0
    # 主要验证不报错
