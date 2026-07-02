"""MiniSWEAgentInner 单元测试。

测试不需要真实 LLM 调用, 用 mock agent 验证适配器逻辑。
"""
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from tree_harness.adapters.mini_swe_inner import (
    MiniSWEAgentInner, MiniSWEConfig, MiniSWEState,
    _extract_action_summary, _extract_output,
)
from tree_harness.modules.outer_harness import (
    ContextBlock, StepObservation, Task,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_context(
    pinned: str = "", relevant: str = "", warnings=None,
) -> ContextBlock:
    return ContextBlock(
        pinned_text=pinned,
        relevant_text=relevant,
        warnings=warnings or [],
        injected_cell_ids=[],
        token_count=0,
        budget_used={"pinned": 0, "relevant": 0, "warnings": 0},
    )


# ---------------------------------------------------------------------------
# MiniSWEState.augment
# ---------------------------------------------------------------------------
class TestMiniSWEStateAugment:
    """测试 context 注入逻辑。"""

    def test_augment_inserts_injection_message(self):
        """augment 后 messages 中应出现注入消息。"""
        agent = MagicMock()
        agent.messages = []

        state = MiniSWEState(agent)
        ctx = make_context(pinned="Pinned L3 cell content")
        state.augment(ctx)

        assert len(agent.messages) == 1
        msg = agent.messages[0]
        assert msg["role"] == "user"
        assert MiniSWEState.INJECTION_MARKER in msg["content"]
        assert "Pinned L3 cell content" in msg["content"]

    def test_augment_replaces_previous_injection(self):
        """第二次 augment 应移除上一步的注入, 只保留一条。"""
        agent = MagicMock()
        agent.messages = []

        state = MiniSWEState(agent)

        # 第一次注入
        ctx1 = make_context(pinned="First injection")
        state.augment(ctx1)
        assert len(agent.messages) == 1

        # 第二次注入 (替换)
        ctx2 = make_context(pinned="Second injection")
        state.augment(ctx2)
        assert len(agent.messages) == 1  # 只有 1 条注入
        assert "Second injection" in agent.messages[0]["content"]
        assert "First injection" not in agent.messages[0]["content"]

    def test_augment_preserves_non_injection_messages(self):
        """augment 不应移除 agent 自己的对话消息。"""
        agent = MagicMock()
        agent.messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Solve this issue"},
            {"role": "assistant", "content": "Let me check...", "extra": {"actions": [{"command": "ls"}]}},
            {"role": "user", "content": "output of ls"},
        ]

        state = MiniSWEState(agent)
        ctx = make_context(pinned="Pinned knowledge")
        state.augment(ctx)

        # 原有 4 条 + 1 条注入 = 5 条
        assert len(agent.messages) == 5
        # 注入在最后
        assert agent.messages[-1]["content"].startswith(MiniSWEState.INJECTION_MARKER)
        # 原有消息不变
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["content"] == "Solve this issue"

    def test_augment_includes_all_sections(self):
        """注入应包含 pinned + relevant + warnings。"""
        agent = MagicMock()
        agent.messages = []

        state = MiniSWEState(agent)
        ctx = make_context(
            pinned="Pinned: Django ORM tips",
            relevant="Relevant: sort fix pattern",
            warnings=["Warning: cell decayed"],
        )
        state.augment(ctx)

        content = agent.messages[0]["content"]
        assert "Pinned: Django ORM tips" in content
        assert "Relevant: sort fix pattern" in content
        assert "Warning: cell decayed" in content

    def test_augment_empty_context_is_noop(self):
        """空 ContextBlock 不应插入消息。"""
        agent = MagicMock()
        agent.messages = []

        state = MiniSWEState(agent)
        ctx = make_context()  # 全空
        state.augment(ctx)

        assert len(agent.messages) == 0


# ---------------------------------------------------------------------------
# MiniSWEState.advance
# ---------------------------------------------------------------------------
class TestMiniSWEStateAdvance:
    """测试 state 前进逻辑。"""

    def test_advance_updates_outcome_on_terminal(self):
        agent = MagicMock()
        state = MiniSWEState(agent)
        obs = StepObservation(
            action={}, result={},
            is_terminal=True, outcome="pass",
        )
        result = state.advance(obs)
        assert result is state
        assert state.outcome == "pass"

    def test_advance_no_outcome_on_non_terminal(self):
        agent = MagicMock()
        state = MiniSWEState(agent)
        obs = StepObservation(
            action={}, result={},
            is_terminal=False, outcome=None,
        )
        state.advance(obs)
        assert state.outcome is None


# ---------------------------------------------------------------------------
# 消息提取工具
# ---------------------------------------------------------------------------
class TestMessageExtraction:
    """测试从 mini-swe-agent 消息中提取 action/output。"""

    def test_extract_action_summary_from_assistant(self):
        msgs = [
            {"role": "assistant", "content": "Let me check", "extra": {"actions": [{"command": "grep -r 'def sort' ."}]}},
        ]
        summary = _extract_action_summary(msgs)
        assert "grep -r 'def sort' ." in summary

    def test_extract_action_summary_no_assistant(self):
        msgs = [
            {"role": "user", "content": "some output"},
        ]
        summary = _extract_action_summary(msgs)
        assert summary == ""

    def test_extract_output_basic(self):
        msgs = [
            {"role": "user", "content": "total 42\ndrwxr-xr-x  2 root root 4096"},
        ]
        output, patch, tests = _extract_output(msgs)
        assert "total 42" in output
        assert patch == ""
        assert tests == {}

    def test_extract_output_with_diff(self):
        msgs = [
            {"role": "user", "content": "diff --git a/file.py b/file.py\n- old\n+ new"},
        ]
        _, patch, _ = _extract_output(msgs)
        assert "diff --git" in patch

    def test_extract_output_with_tests(self):
        msgs = [
            {"role": "user", "content": "test_sort PASSED\ntest_filter FAILED"},
        ]
        _, _, tests = _extract_output(msgs)
        assert "test_sort" in tests
        assert tests["test_sort"] == "pass"

    def test_extract_exit_submission(self):
        msgs = [
            {"role": "exit", "content": "Submitted", "extra": {"exit_status": "Submitted", "submission": "patch content here"}},
        ]
        output, _, _ = _extract_output(msgs)
        assert "Submission" in output


# ---------------------------------------------------------------------------
# MiniSWEAgentInner (不依赖真实 LLM)
# ---------------------------------------------------------------------------
class TestMiniSWEAgentInnerCapabilities:
    """测试 capabilities 声明。"""

    def test_capabilities_returns_correct_defaults(self):
        inner = MiniSWEAgentInner.__new__(MiniSWEAgentInner)
        caps = inner.capabilities()
        assert caps.supports_pin_marker is False
        assert caps.supports_warning_marker is False
        assert caps.history_window_tokens == 128000
        assert caps.has_internal_compaction is False

    def test_capabilities_has_no_pin_or_warning_markers(self):
        """mini-swe-agent 不识别 pin/warning marker, 只能通过 user 消息注入。"""
        inner = MiniSWEAgentInner.__new__(MiniSWEAgentInner)
        caps = inner.capabilities()
        assert not caps.supports_pin_marker
        assert not caps.supports_warning_marker


class TestMiniSWEAgentInnerIsTerminal:
    """测试终止检测。"""

    def test_is_terminal_no_agent(self):
        inner = MiniSWEAgentInner.__new__(MiniSWEAgentInner)
        inner._agent = None
        assert inner.is_terminal(None) is True

    def test_is_terminal_empty_messages(self):
        agent = MagicMock()
        agent.messages = []
        inner = MiniSWEAgentInner.__new__(MiniSWEAgentInner)
        inner._agent = agent
        assert inner.is_terminal(None) is True

    def test_is_terminal_exit_role(self):
        agent = MagicMock()
        agent.messages = [
            {"role": "system", "content": "..."},
            {"role": "exit", "content": "Submitted"},
        ]
        inner = MiniSWEAgentInner.__new__(MiniSWEAgentInner)
        inner._agent = agent
        assert inner.is_terminal(None) is True

    def test_is_terminal_not_exit(self):
        agent = MagicMock()
        agent.messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "working"},
        ]
        inner = MiniSWEAgentInner.__new__(MiniSWEAgentInner)
        inner._agent = agent
        assert inner.is_terminal(None) is False


class TestMiniSWEConfig:
    """测试配置默认值。"""

    def test_default_config_values(self):
        cfg = MiniSWEConfig()
        assert cfg.model_name == "openai/qwen-2.5-coder-32b-instruct"
        assert cfg.step_limit == 30
        assert cfg.cost_limit == 3.0
        assert cfg.env_kind == "local"
        assert cfg.env_timeout == 120

    def test_custom_config(self):
        cfg = MiniSWEConfig(
            model_name="openai/gpt-4o",
            step_limit=10,
            env_cwd="/tmp/test",
        )
        assert cfg.model_name == "openai/gpt-4o"
        assert cfg.step_limit == 10
        assert cfg.env_cwd == "/tmp/test"
