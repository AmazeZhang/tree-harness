"""MiniSWEAgentInner — 将 mini-swe-agent 包装为 InnerHarnessProtocol。

适配器模式: 不修改 mini-swe-agent 源码, 通过包装实现 Protocol。
OuterHarness 不感知 mini-swe-agent 的存在, inner_kind 可替换。

对应 spec: docs/specs/mini_swe_inner.md
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

from tree_harness.modules.outer_harness import (
    ContextBlock, StepObservation, InnerHarnessProtocol,
    InnerCapabilities,
)

logger = logging.getLogger(__name__)

# 延迟导入 mini-swe-agent (仅在 inner_kind="mini-swe-agent" 时需要)
_DEFAULT_AGENT = None
_LOCAL_ENV = None
_GET_MODEL = None


def _lazy_imports():
    global _DEFAULT_AGENT, _LOCAL_ENV, _GET_MODEL
    if _DEFAULT_AGENT is None:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models import get_model
        _DEFAULT_AGENT = DefaultAgent
        _LOCAL_ENV = LocalEnvironment
        _GET_MODEL = get_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MiniSWEConfig:
    """mini-swe-agent inner harness 配置。"""
    # Model
    model_name: str = "openai/qwen-2.5-coder-32b-instruct"

    # Agent
    step_limit: int = 30               # 每 episode 最大步数
    cost_limit: float = 3.0            # 最大 LLM 费用 ($)
    wall_time_limit_seconds: int = 600  # 墙钟超时 (秒)
    system_template: str = ""          # 留空用 mini-swe 默认
    instance_template: str = ""        # 留空用 mini-swe 默认

    # Environment
    env_kind: str = "local"            # "local" (未来支持 "docker")
    env_cwd: str = ""                  # 工作目录 (空=临时目录)
    env_timeout: int = 120             # 命令超时 (秒)


# ---------------------------------------------------------------------------
# State — 持有 agent.messages 引用
# ---------------------------------------------------------------------------
class MiniSWEState:
    """mini-swe-agent 的 StepState 实现。

    持有 DefaultAgent 引用, augment 往 messages 插入 Tree context。
    采用替换式注入: 每步移除旧注入, 插入新注入, 避免 messages 膨胀。
    """

    INJECTION_MARKER = "<!--TREE_CONTEXT_INJECTION-->"

    def __init__(self, agent):
        self._agent = agent
        self.outcome: Optional[str] = None

    def augment(self, context: ContextBlock) -> "MiniSWEState":
        """把 Tree context 注入到 agent.messages。

        替换式: 先移除上一步的注入消息, 再插入新的。
        """
        # 1. 移除上一步的注入
        self._agent.messages = [
            m for m in self._agent.messages
            if not str(m.get("content", "")).startswith(self.INJECTION_MARKER)
        ]

        # 2. 拼装注入文本
        parts = []
        if context.pinned_text:
            parts.append(
                "<|PINNED_DO_NOT_COMPACT|>\n"
                f"{context.pinned_text}\n"
                "<|/PINNED|>"
            )
        if context.relevant_text:
            parts.append(
                "[Relevant Experience from Tree]\n"
                f"{context.relevant_text}"
            )
        if context.warnings:
            warning_text = "\n".join(f"- {w}" for w in context.warnings)
            parts.append(
                "<|WARNING_DO_NOT_COMPACT|>\n"
                f"{warning_text}\n"
                "<|/WARNING|>"
            )

        if not parts:
            return self

        injection = self.INJECTION_MARKER + "\n" + "\n\n".join(parts)

        # 3. 插入为 user 消息
        self._agent.messages.append({
            "role": "user",
            "content": injection,
        })

        return self

    def advance(self, observation: StepObservation) -> "MiniSWEState":
        """mini-swe-agent 的 step() 已经自动更新了 messages, 这里只需更新 outcome。"""
        if observation.is_terminal and observation.outcome:
            self.outcome = observation.outcome
        return self

    def snapshot(self) -> dict:
        """浅快照, 供 StepRecord 序列化。"""
        return {
            "n_messages": len(self._agent.messages),
            "n_calls": self._agent.n_calls,
            "cost": self._agent.cost,
            "outcome": self.outcome,
        }


# ---------------------------------------------------------------------------
# Inner — 包装 DefaultAgent
# ---------------------------------------------------------------------------
class MiniSWEAgentInner:
    """mini-swe-agent 的 InnerHarnessProtocol 实现。

    将 DefaultAgent 的 run() 循环拆成 step-by-step,
    让 OuterHarness 在每步之间注入 context 并观测 trajectory。
    """

    def __init__(self, config: Optional[MiniSWEConfig] = None):
        _lazy_imports()
        self.config = config or MiniSWEConfig()
        self._agent: Optional[_DEFAULT_AGENT] = None
        self._state: Optional[MiniSWEState] = None
        # 外部注入的环境/模型 (用于 SWE-bench Docker 等场景)
        self._external_env = None
        self._external_model = None

    # ------------------------------------------------------------------
    # 外部注入 (SWE-bench Docker 等)
    # ------------------------------------------------------------------
    def set_environment(self, env) -> None:
        """注入预配置的环境 (如 SWE-bench 的 DockerEnvironment)。

        调用后 reset() 将使用此环境而非创建 LocalEnvironment。
        """
        self._external_env = env

    def set_model(self, model) -> None:
        """注入预配置的 model (如带 SWE-bench 模板配置的 litellm model)。

        调用后 reset() 将使用此 model 而非自行创建。
        """
        self._external_model = model

    # ------------------------------------------------------------------
    # InnerHarnessProtocol
    # ------------------------------------------------------------------
    def reset(self, task) -> MiniSWEState:
        """开启新 episode, 初始化 agent。

        提取自 DefaultAgent.run() 的初始化逻辑:
        1. 构建 Model + Environment
        2. 用默认或自定义模板创建 AgentConfig
        3. 渲染 system + instance 消息
        """
        cfg = self.config

        # 构建 Model: 优先用外部注入 (SWE-bench), 否则自行创建
        if self._external_model is not None:
            model = self._external_model
        else:
            model = _GET_MODEL(cfg.model_name, config={
                "model_kwargs": {"drop_params": False},
            })

        # 构建 Environment: 优先用外部注入 (SWE-bench Docker), 否则按 env_kind 创建
        if self._external_env is not None:
            env = self._external_env
        elif cfg.env_kind == "local":
            cwd = cfg.env_cwd or tempfile.mkdtemp(prefix="tree_inner_")
            env = _LOCAL_ENV(cwd=cwd, timeout=cfg.env_timeout)
        else:
            raise ValueError(f"Unsupported env_kind: {cfg.env_kind}")

        # 加载模板 (用 swebench 配置, 它支持 toolcall 格式)
        from minisweagent.config import get_config_from_spec
        default_cfg = get_config_from_spec("swebench")
        default_agent_cfg = default_cfg.get("agent", {})

        system_template = cfg.system_template or default_agent_cfg.get("system_template", "")
        instance_template = cfg.instance_template or default_agent_cfg.get("instance_template", "")

        if not system_template or not instance_template:
            raise ValueError(
                "system_template and instance_template are required "
                "(set in config or rely on mini-swe default)"
            )

        # 创建 DefaultAgent
        self._agent = _DEFAULT_AGENT(
            model, env,
            system_template=system_template,
            instance_template=instance_template,
            step_limit=cfg.step_limit,
            cost_limit=cfg.cost_limit,
            wall_time_limit_seconds=cfg.wall_time_limit_seconds,
        )

        # 设置 task 变量, 渲染初始消息 (提取自 run() 前两行)
        self._agent.extra_template_vars = {"task": task.description}
        self._agent.messages = []
        self._agent.add_messages(
            self._agent.model.format_message(
                role="system",
                content=self._agent._render_template(
                    self._agent.config.system_template,
                ),
            ),
            self._agent.model.format_message(
                role="user",
                content=self._agent._render_template(
                    self._agent.config.instance_template,
                ),
            ),
        )

        # 创建 state
        self._state = MiniSWEState(self._agent)
        return self._state

    def step(self, state: MiniSWEState) -> StepObservation:
        """执行一步: query LLM + execute actions。

        处理 mini-swe-agent 的异常:
        - FormatError: LLM 格式错误, 加错误消息重试
        - Submitted: 任务完成, 加 exit 消息
        - LimitsExceeded/TimeExceeded: 超限, 加 exit 消息
        """
        from minisweagent.exceptions import (
            FormatError, InterruptAgentFlow,
        )
        prev_len = len(self._agent.messages)

        try:
            self._agent.step()
            self._agent.n_consecutive_format_errors = 0
        except FormatError as e:
            # 加错误消息, 让 LLM 下次修正格式 (同 run() 的处理)
            self._agent.add_messages(*e.messages)
            self._agent.n_consecutive_format_errors += 1
            if 0 < self._agent.config.max_consecutive_format_errors <= self._agent.n_consecutive_format_errors:
                self._agent.add_messages({
                    "role": "exit",
                    "content": "RepeatedFormatError",
                    "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                })
        except InterruptAgentFlow as e:
            self._agent.add_messages(*e.messages)
        except Exception as e:
            # Submitted, LimitsExceeded, TimeExceeded 等
            # handle_uncaught_exception 会加 exit 消息
            self._agent.handle_uncaught_exception(e)

        # 从新追加的消息中提取 observation
        new_msgs = self._agent.messages[prev_len:]
        last_msg = self._agent.messages[-1] if self._agent.messages else {}

        # 检测终止
        is_terminal = last_msg.get("role") == "exit"
        outcome = None
        if is_terminal:
            extra = last_msg.get("extra", {})
            exit_status = extra.get("exit_status", "")
            # 映射 exit_status → 我们的 outcome
            if exit_status == "Submitted":
                outcome = "pass"
            elif exit_status in ("LimitsExceeded", "TimeExceeded", "RepeatedFormatError"):
                outcome = "error"
            else:
                outcome = "fail"

        # 提取 action 和 result
        action_summary = _extract_action_summary(new_msgs)
        output_text, patch, tests = _extract_output(new_msgs)

        return StepObservation(
            action={"summary": action_summary},
            result={
                "summary": output_text,
                "patch": patch,
                "tests": tests,
                "outcome": outcome or "pending",
                "duration": 0.0,
                "tokens": 0,
            },
            is_terminal=is_terminal,
            outcome=outcome,
            raw_output=new_msgs,
        )

    def is_terminal(self, state: MiniSWEState) -> bool:
        """检查 episode 是否结束。"""
        if not self._agent or not self._agent.messages:
            return True
        return self._agent.messages[-1].get("role") == "exit"

    def capabilities(self) -> InnerCapabilities:
        """mini-swe-agent 的能力声明。"""
        return InnerCapabilities(
            supports_pin_marker=False,       # mini-swe 不识别 pin marker
            supports_warning_marker=False,   # mini-swe 不识别 warning marker
            history_window_tokens=128000,    # 取决于 model, 默认 128k
            has_internal_compaction=False,   # mini-swe 线性历史, 无 compaction
        )


# ---------------------------------------------------------------------------
# 消息提取工具
# ---------------------------------------------------------------------------
def _extract_action_summary(new_msgs: list[dict]) -> str:
    """从新消息中提取 action 摘要。"""
    for msg in new_msgs:
        if msg.get("role") == "assistant":
            actions = msg.get("extra", {}).get("actions", [])
            if actions:
                cmds = [a.get("command", "") for a in actions]
                return "; ".join(cmds)[:500]
            content = msg.get("content", "")
            if content:
                return content[:200]
    return ""


def _extract_output(new_msgs: list[dict]) -> tuple[str, str, dict]:
    """从新消息中提取输出文本, patch, 测试结果。

    mini-swe-agent v2 的 observation 消息 role="tool" (toolcall 格式),
    旧版或 human-issued 命令的 role="user"。
    """
    output_parts = []
    patch = ""
    tests = {}

    for msg in new_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("tool", "user"):
            # observation message (tool result or user observation)
            output_parts.append(content)
            # 尝试提取 patch (diff 格式)
            if "diff --git" in content:
                lines = content.split("\n")
                diff_start = next(
                    (i for i, l in enumerate(lines) if l.startswith("diff --git")),
                    None,
                )
                if diff_start is not None:
                    patch = "\n".join(lines[diff_start:])

            # 尝试提取测试结果
            if "PASSED" in content or "FAILED" in content or "passed" in content:
                for line in content.split("\n"):
                    if "PASSED" in line or "passed" in line:
                        test_name = line.strip().split()[0] if line.strip() else ""
                        if test_name:
                            tests[test_name] = "pass"
                    elif "FAILED" in line or "failed" in line:
                        test_name = line.strip().split()[0] if line.strip() else ""
                        if test_name:
                            tests[test_name] = "fail"

        elif role == "exit":
            submission = msg.get("extra", {}).get("submission", "")
            if submission:
                output_parts.append(f"[Submission]: {submission[:200]}")

    output_text = "\n".join(output_parts)[:2000]
    return output_text, patch, tests
