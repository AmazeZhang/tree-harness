"""Cambium Engine — 形成层,从 agent 轨迹蒸馏结构化 cell。

Self-Evolution Operator Set 中 crystallize 算符的实现策略
(同时承担 connect 算符在新建 cell 上的初始连边)。

三步管线:
    StandardStep
      → [Step A: Crystallize] LLM 提取 CandidateCell
      → [Step B: Dedup]       判定 INSERT_NEW / REINFORCE
      → [Step C: Connect]     新 cell 建立 RAY 连接
      → cells in tree

定位: 无状态算法服务,由 OuterHarness.after_step() 调用。
不维护 episode-local 状态,不感知三 hook 的存在。
对应 spec: docs/specs/cambium_engine.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal

from tree_harness.core.cell_model import (
    Cell, CandidateCell, Precondition, VerifyHint,
    StandardStep, StandardTrajectory,
    create_cell,
)
from tree_harness.core.embedding import embed_cell_text
from tree_harness.core.llm_client import LLMClient, parse_llm_json
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem
from tree_harness.modules.dedup import Dedup, DedupConfig
from tree_harness.modules.connector import Connector, ConnectorConfig


@dataclass
class CambiumConfig:
    """CambiumEngine 顶层配置; 持有 dedup 与 connector 的嵌套子配置。

    Cambium 是唯一 config 注入点,不会出现 config 在多个模块间漂移。
    """
    dedup: DedupConfig = field(default_factory=DedupConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    initial_energy: float = 0.5
    initial_maturity: float = 0.0


# ---------------------------------------------------------------------------
# 准入: 机械操作检测
# ---------------------------------------------------------------------------
_MECHANICAL_COMMANDS = frozenset({"ls", "cd", "cat", "pwd", "git status", "git diff"})


class CambiumEngine:
    """形成层 — 从 StandardStep 蒸馏 cell 写入 TreeStore。

    无状态算法服务: 是否调用、何时调用、调用频率全部由 OuterHarness 决定。
    """

    def __init__(
        self,
        tree_store: TreeStore,
        energy_system: EnergySystem,
        llm_client: LLMClient,
        config: CambiumConfig,
    ):
        self.tree_store = tree_store
        self.energy_system = energy_system
        self.llm_client = llm_client
        self.config = config
        self.dedup = Dedup(tree_store, llm_client, config.dedup)
        self.connector = Connector(tree_store, config.connector)

    # ------------------------------------------------------------------
    # step-level 入口 (OuterHarness.after_step 使用)
    # ------------------------------------------------------------------
    def should_crystallize(self, step: StandardStep) -> bool:
        """准入判断: 本步是否值得蒸馏。"""
        return self._worth_extracting_step(step)

    def crystallize_step(self, step: StandardStep) -> List[Cell]:
        """单步蒸馏。完整执行 Step A → B → C,返回本步新生 cell (可能为空)。

        REINFORCE 命中已有 cell 时不出现在返回值中。
        """
        candidates = self._llm_extract_step(step)
        if not candidates:
            return []

        new_cells: List[Cell] = []
        for candidate in candidates:
            # 补全 candidate 的上下文字段
            candidate.context_trigger_task = step.task_id
            # 预计算 embedding (dedup / connector 需要)
            candidate.embedding = self._compute_embedding(candidate)

            # Step B: Dedup
            result = self.dedup.check(candidate)
            if result.action == "REINFORCE":
                self.energy_system.reference(result.matched_cell_id, step.episode_id)
                continue

            # INSERT_NEW: 创建 cell
            cell = create_cell(
                source="distilled",
                trigger_task=step.task_id,
                domain=candidate.context_domain or self._infer_domain(candidate),
                decision=candidate.decision,
                rationale=candidate.rationale,
                preconditions=candidate.preconditions,
                evidence=candidate.evidence,
                domain_tags=candidate.domain_tags,
                energy=self.config.initial_energy,
                maturity=self.config.initial_maturity,
            )
            self.tree_store.insert_cell(cell)

            # Step C: Connect
            self._connect(cell, step.episode_id)
            new_cells.append(cell)

        return new_cells

    def connect_new_cells(self, new_cells: List[Cell]) -> None:
        """批量为 new_cells 建立外向 ray (跨 new cell 的相互 connect)。

        非必需——若 OuterHarness 不调用,则只依赖 Step C 的单 cell connect。
        """
        for cell in new_cells:
            self.connector.connect(cell)

    # ------------------------------------------------------------------
    # episode-level 入口 (离线批处理 / 烟测用, OuterHarness 不调用)
    # ------------------------------------------------------------------
    def crystallize(self, trajectory: StandardTrajectory) -> List[Cell]:
        """对完整 trajectory 批量蒸馏。"""
        if not self._worth_extracting(trajectory):
            return []

        all_new_cells: List[Cell] = []
        for step in trajectory.steps:
            if self.should_crystallize(step):
                new_cells = self.crystallize_step(step)
                all_new_cells.extend(new_cells)

        # 批量再连边 (跨 new cell 的相互 connect)
        if all_new_cells:
            self.connect_new_cells(all_new_cells)
        return all_new_cells

    # ------------------------------------------------------------------
    # 内部: 准入判断
    # ------------------------------------------------------------------
    def _worth_extracting_step(self, step: StandardStep) -> bool:
        if step.outcome_so_far == "error":
            return False
        if step.outcome_so_far == "pending":
            if not step.patch_delta and not step.test_results:
                return False
        if step.outcome_so_far == "fail" and not self._step_has_clear_lesson(step):
            return False
        if self._is_mechanical_step(step):
            return False
        return True

    def _worth_extracting(self, trajectory: StandardTrajectory) -> bool:
        if trajectory.outcome == "error":
            return False
        if trajectory.outcome == "fail" and not self._has_clear_lesson(trajectory):
            return False
        if self._is_mechanical(trajectory):
            return False
        return True

    def _step_has_clear_lesson(self, step: StandardStep) -> bool:
        return bool(step.patch_delta or step.test_results)

    def _has_clear_lesson(self, trajectory: StandardTrajectory) -> bool:
        return bool(trajectory.patches or trajectory.test_results)

    def _is_mechanical_step(self, step: StandardStep) -> bool:
        action = step.action_summary.strip().lower()
        for cmd in _MECHANICAL_COMMANDS:
            if action.startswith(cmd):
                return True
        return False

    def _is_mechanical(self, trajectory: StandardTrajectory) -> bool:
        # 如果 key_actions 为空或全是机械命令
        if not trajectory.key_actions:
            return True
        return all(
            any(a.strip().lower().startswith(cmd) for cmd in _MECHANICAL_COMMANDS)
            for a in trajectory.key_actions
        )

    # ------------------------------------------------------------------
    # 内部: Step A — LLM 提取
    # ------------------------------------------------------------------
    def _llm_extract_step(self, step: StandardStep) -> List[CandidateCell]:
        """Step A: LLM 提取候选 cell。"""
        system_prompt = (
            "You are a crystallize assistant. "
            "Extract reusable decision knowledge from agent execution records."
        )
        user_prompt = self._render_crystallize_prompt(step)
        response = self.llm_client.complete(system_prompt, user_prompt)

        try:
            data = parse_llm_json(response)
            # 支持 LLM 返回单个对象或数组
            items = data if isinstance(data, list) else [data]
        except Exception:
            return []

        candidates: List[CandidateCell] = []
        for item in items:
            if not item.get("decision"):
                continue
            preconditions = self._parse_preconditions(item.get("preconditions", []))
            candidate = CandidateCell(
                decision=item["decision"],
                rationale=item.get("rationale", ""),
                preconditions=preconditions,
                evidence=item.get("evidence", []),
                domain_tags=item.get("domain_tags", []),
            )
            candidates.append(candidate)
        return candidates

    def _parse_preconditions(self, raw: list) -> List[Precondition]:
        preconditions: List[Precondition] = []
        for pc in raw:
            verify_hint = None
            vh = pc.get("verify_hint")
            if vh:
                verify_hint = VerifyHint(
                    type=vh.get("type", "file_grep"),
                    params=vh.get("params", {}),
                )
            preconditions.append(Precondition(
                kind=pc.get("kind", "fact"),
                assertion=pc.get("assertion", ""),
                verify_hint=verify_hint,
            ))
        return preconditions

    def _render_crystallize_prompt(self, step: StandardStep) -> str:
        return (
            f"从以下 agent 执行记录中提取可复用的决策知识。\n\n"
            f"执行记录:\n"
            f"- Task: {step.task_id}\n"
            f"- Repo: {step.repo}\n"
            f"- Outcome: {step.outcome_so_far}\n"
            f"- Action: {step.action_summary}\n"
            f"- Observation: {step.observation_summary}\n"
            f"- Patch: {step.patch_delta or 'none'}\n"
            f"- Test results: {step.test_results or 'none'}\n\n"
            f"要求每条知识输出 JSON 格式:\n"
            f'{{"decision": "具体做了什么决策", '
            f'"rationale": "为什么这样决策", '
            f'"preconditions": [], "evidence": [], "domain_tags": []}}\n\n'
            f"规则:\n"
            f"- 只提取\"下次遇到类似情况能直接复用\"的知识\n"
            f"- 跳过纯机械操作 (如 git add, cd 到目录)\n"
            f"- 对 config/dependency/code_invariant 类 precondition 尽量给出 verify_hint\n"
            f"- 避免过于具体 (绑定特定文件行号) 或过于宽泛 (适用于任何项目)"
        )

    # ------------------------------------------------------------------
    # 内部: 辅助
    # ------------------------------------------------------------------
    def _compute_embedding(self, candidate: CandidateCell) -> List[float]:
        embedder = self.tree_store.sqlite.embedder
        return embedder.embed(
            embed_cell_text(candidate.decision, candidate.rationale)
        )

    def _infer_domain(self, candidate: CandidateCell) -> str:
        if candidate.domain_tags:
            return candidate.domain_tags[0]
        return "general"

    def _connect(self, new_cell: Cell, episode_id: Optional[str] = None) -> None:
        """Step C: 建立 ray 连接。"""
        self.connector.connect(new_cell, episode_id=episode_id)
