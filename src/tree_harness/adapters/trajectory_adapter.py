"""TrajectoryAdapter — StepRecord → StandardStep 翻译层。

OuterHarness after_step 的内部实现细节。不调用任何算符,只做格式翻译。

对应 spec: docs/specs/trajectory_adapter.md
"""
from __future__ import annotations

from typing import Optional, Protocol, Any

from tree_harness.core.cell_model import Cell, StandardStep


class TrajectoryAdapter:
    """轨迹翻译层 — 纯数据转换,不持有 Tree 引用。"""

    def __init__(self):
        self._current_task_id: str = ""
        self._current_repo: str = ""

    def set_context(self, task_id: str, repo: str) -> None:
        """OuterHarness 在 episode 开始时调用,设置当前 task 上下文。"""
        self._current_task_id = task_id
        self._current_repo = repo

    # ------------------------------------------------------------------
    # convert_step: StepRecord → StandardStep
    # ------------------------------------------------------------------
    def convert_step(self, record: "StepRecord") -> StandardStep:
        """单步翻译: StepRecord → StandardStep。"""
        action = record.action or {}
        observation = record.observation or {}

        return StandardStep(
            task_id=self._current_task_id,
            episode_id=record.episode_id,
            step_index=record.step_index,
            repo=self._current_repo,
            action_summary=action.get("summary", action.get("action", "")),
            observation_summary=observation.get("summary", observation.get("stdout", "")),
            patch_delta=observation.get("patch"),
            test_results=observation.get("tests", observation.get("test_results", {})),
            outcome_so_far=observation.get("outcome", "pending"),
            duration_seconds=observation.get("duration", 0.0),
            token_usage=observation.get("tokens", 0),
        )

    # ------------------------------------------------------------------
    # 告警格式化 (after_step quarantine 后调用)
    # ------------------------------------------------------------------
    def format_quarantine_warning(
        self, cell: Cell, evidence: str, verifier_name: str,
    ) -> str:
        """把 quarantine 算符的执行结果翻译为自然语言告警。

        约束 (trajectory_adapter.md):
        - 必须引用被 quarantine cell 的 Decision 原文
        - 必须给出 verification 失败原因 (verifier 名 + evidence 摘要)
        - 必须以祈使句结尾,告知 inner agent disregard
        """
        return (
            f'WARNING: A previously injected guideline ("{cell.decision}") '
            f"has been quarantined by verifier `{verifier_name}` — {evidence}. "
            f"Disregard it for the remainder of this episode."
        )

    def format_neighbor_warning(
        self, neighbor: Cell, quarantined: Cell, ray: dict,
    ) -> str:
        """沿 incoming ray 一跳传播时使用,生成邻居预警文本。

        约束 (trajectory_adapter.md):
        - 必须引用被 quarantine 的源 cell (不引用邻居自己的 Decision)
        - 必须说明 ray 关系
        - 必须以 "verify before relying" 语气结尾
        """
        weight = ray.get("weight", 0.0)
        source_type = ray.get("source_type", "semantic")
        return (
            f'ADJACENT WARNING: The cell "{neighbor.decision}" is connected '
            f"(ray weight={weight:.2f}, source={source_type}) to a guideline "
            f'that was just quarantined ("{quarantined.decision}"). '
            f"The adjacent cell itself has not been refuted, but verify its "
            f"preconditions against the current repo state before relying on it."
        )
