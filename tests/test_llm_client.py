"""LLMClient 测试 —— DeterministicLLMClient + parse_llm_json。"""
import json

import pytest

from tree_harness.core.llm_client import DeterministicLLMClient, parse_llm_json
from tree_harness.core.cell_model import (
    StandardStep, StandardTrajectory, CandidateCell, Precondition,
)


# ---------------------------------------------------------------------------
# DeterministicLLMClient
# ---------------------------------------------------------------------------
def test_deterministic_caching():
    """相同输入 → 相同输出 (temperature=0 + 全量缓存约束)。"""
    client = DeterministicLLMClient(default_response='{"k":"v"}')
    r1 = client.complete("system", "user")
    r2 = client.complete("system", "user")
    assert r1 == r2
    assert client.call_count == 2
    assert client.cache_hit_count == 1


def test_deterministic_different_inputs():
    """不同输入 → 可能不同输出。"""
    client = DeterministicLLMClient(default_response='{"k":"v"}')
    r1 = client.complete("system A", "user")
    r2 = client.complete("system B", "user")
    # 都是 default,但各自缓存
    assert r1 == r2  # default 值相同
    assert client.call_count == 2
    assert client.cache_hit_count == 0


def test_inject_response():
    """注入特定 system prompt 的响应。"""
    client = DeterministicLLMClient(default_response='{"default":true}')
    client.inject("crystallize", json.dumps({"decision": "test"}))

    r = client.complete("You are a crystallize assistant", "step data")
    assert json.loads(r)["decision"] == "test"

    # 未匹配的 prompt → default
    r2 = client.complete("You are a dedup assistant", "cell data")
    assert json.loads(r2)["default"] is True


def test_inject_clears_cache():
    """注入新响应后清空缓存,确保后续调用使用新注入。"""
    client = DeterministicLLMClient(default_response='{"old":true}')
    r1 = client.complete("crystallize prompt", "data")
    assert json.loads(r1)["old"] is True

    client.inject("crystallize", json.dumps({"new": True}))
    r2 = client.complete("crystallize prompt", "data")
    assert json.loads(r2)["new"] is True


def test_inject_fifo_priority():
    """多个注入匹配时,先注入的优先 (FIFO)。"""
    client = DeterministicLLMClient()
    client.inject("crystallize", json.dumps({"first": True}))
    client.inject("assistant", json.dumps({"second": True}))

    # 两者都匹配,但 "crystallize" 先注入
    r = client.complete("You are a crystallize assistant", "data")
    assert json.loads(r)["first"] is True


# ---------------------------------------------------------------------------
# parse_llm_json
# ---------------------------------------------------------------------------
def test_parse_plain_json():
    result = parse_llm_json('{"decision": "test", "rationale": "because"}')
    assert result["decision"] == "test"


def test_parse_markdown_wrapped_json():
    response = '```json\n{"decision": "test", "rationale": "because"}\n```'
    result = parse_llm_json(response)
    assert result["decision"] == "test"


def test_parse_with_whitespace():
    result = parse_llm_json('  \n  {"decision": "test"}  \n  ')
    assert result["decision"] == "test"


# ---------------------------------------------------------------------------
# Phase 2 数据结构
# ---------------------------------------------------------------------------
def test_standard_step_creation():
    step = StandardStep(
        task_id="django__django-16379",
        episode_id="ep-001",
        step_index=0,
        repo="django/django",
        action_summary="Edited compiler.py to add nulls_first parameter",
        observation_summary="Test test_ordering_null passed",
        patch_delta="diff --git a/compiler.py",
        test_results={"test_ordering_null": "pass"},
        outcome_so_far="pass",
        duration_seconds=12.5,
        token_usage=500,
    )
    assert step.task_id == "django__django-16379"
    assert step.outcome_so_far == "pass"


def test_standard_trajectory_creation():
    step = StandardStep(
        task_id="t1", episode_id="ep1", step_index=0, repo="r",
        action_summary="a", observation_summary="o",
        patch_delta=None, test_results={},
        outcome_so_far="pending", duration_seconds=1.0, token_usage=10,
    )
    traj = StandardTrajectory(
        task_id="t1", task_description="Fix bug", repo="r",
        base_commit="abc123", outcome="pass",
        patches=["diff"], test_results={"test1": "pass"},
        key_actions=["edited file"], duration_seconds=10.0, token_usage=100,
        steps=[step],
    )
    assert traj.outcome == "pass"
    assert len(traj.steps) == 1


def test_candidate_cell_creation():
    candidate = CandidateCell(
        decision="Always validate input before processing",
        rationale="Unvalidated input caused crash",
        preconditions=[
            Precondition(kind="test_existence", assertion="test_validation exists"),
        ],
        evidence=["test_id:test_validation", "file:validator.py"],
        domain_tags=["validation", "safety"],
    )
    assert candidate.decision == "Always validate input before processing"
    assert len(candidate.preconditions) == 1
    assert candidate.preconditions[0].kind == "test_existence"
