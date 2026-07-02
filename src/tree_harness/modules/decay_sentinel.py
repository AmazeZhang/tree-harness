"""DecaySentinel — 树的免疫系统 (CODIT)。

无状态算法服务,由 OuterHarness.after_step() 调用:
拿到候选 cell 列表后跑漏斗式验证,把裁决结果回传给 OuterHarness。

定位:Sentinel 只做"判定",不做"执行"(quarantine 等算符副作用由 OuterHarness 完成)。
Signal-level 副作用 (energy 微调 / mark_review) 由 Sentinel 直接触发。

对应 spec: docs/specs/decay_sentinel.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Literal, Protocol, runtime_checkable

from tree_harness.core.cell_model import Cell
from tree_harness.core.oplog import OpLog, OpEnum
from tree_harness.store.tree_store import TreeStore
from tree_harness.modules.energy_system import EnergySystem
from tree_harness.modules.verifiers import VerifierRegistry, CellVerifyResult


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    """漏斗验证裁决。"""
    result: Literal["valid", "weak_valid", "decayed", "uncertain"]
    reason: str
    step_reached: int          # 1 / 2 / 3
    evidence: Optional[str] = None
    verifier_name: str = ""   # 哪个 verifier/step 做出了裁决 (for warning text)


# ---------------------------------------------------------------------------
# 漏斗统计
# ---------------------------------------------------------------------------
@dataclass
class FunnelStats:
    """漏斗各层处理统计 (供 metrics 消费)。"""
    step1_resolved: int = 0
    step2a_resolved: int = 0
    step2b_resolved: int = 0
    step3_resolved: int = 0
    verdicts: Dict[str, int] = field(default_factory=lambda: {
        "valid": 0, "weak_valid": 0, "decayed": 0, "uncertain": 0,
    })


# ---------------------------------------------------------------------------
# TestRunner 协议 (可注入, 默认 None → 跳过 Step 2a)
# ---------------------------------------------------------------------------
@runtime_checkable
class TestRunnerProtocol(Protocol):
    def run_test(self, test_id: str, repo_path: str) -> Literal["pass", "fail", "unknown"]: ...


# ---------------------------------------------------------------------------
# DecaySentinel
# ---------------------------------------------------------------------------
class DecaySentinel:
    """漏斗式验证 — 多层证据收集,从便宜到贵。"""

    def __init__(
        self,
        tree_store: TreeStore,
        energy_system: EnergySystem,
        verifier_registry: VerifierRegistry,
        llm_client,
        test_runner: Optional[TestRunnerProtocol] = None,
        repo_path: str = ".",
    ):
        self.tree_store = tree_store
        self.energy_system = energy_system
        self.verifier_registry = verifier_registry
        self.llm_client = llm_client
        self.test_runner = test_runner
        self.repo_path = repo_path
        self._stats = FunnelStats()

    @property
    def stats(self) -> FunnelStats:
        return self._stats

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def sample_high_ring_cells(self, sample_size: int = 5) -> List[str]:
        """随机抽取 L3/L4 active cell (不依赖 energy threshold)。

        P1-1: 解决 L3/L4 不死性 — 即使 energy 为正,也定期抽检高 ring cell。
        """
        import random
        high_ring_cells = self.tree_store.list_by_ring(
            ["L3", "L4"], status="active",
        )
        if not high_ring_cells:
            return []
        k = min(sample_size, len(high_ring_cells))
        return [c.id for c in random.sample(high_ring_cells, k)]

    def funnel_verify(
        self,
        candidate_ids: List[str],
        episode_id: Optional[str] = None,
    ) -> Dict[str, Verdict]:
        """对一批候选 cell 执行漏斗验证,返回 {cell_id: verdict}。

        Sentinel 内部副作用:
        - valid → EnergySystem.reference()
        - uncertain → EnergySystem.decay_one(Δ=-0.05) + TreeStore.mark_for_review()
        - decayed → 无 (OuterHarness 执行 quarantine)
        """
        results: Dict[str, Verdict] = {}
        for cell_id in candidate_ids:
            cell = self.tree_store.get_cell(cell_id)
            if cell is None or cell.status != "active":
                results[cell_id] = Verdict(
                    result="uncertain",
                    reason="cell not found or not active",
                    step_reached=0,
                    verifier_name="skip",
                )
                continue
            verdict = self._funnel_single(cell, episode_id)
            results[cell_id] = verdict

            # Signal-level 副作用 (Sentinel 直接触发)
            self._apply_signal_side_effect(verdict, cell_id, episode_id)

        return results

    # ------------------------------------------------------------------
    # 漏斗单 cell
    # ------------------------------------------------------------------
    def _funnel_single(self, cell: Cell, episode_id: Optional[str]) -> Verdict:
        # Step 1: 被动信号
        v = self._step1_passive_signals(cell)
        if v is not None:
            self._stats.step1_resolved += 1
            return v

        # Step 2a: 测试验证
        v = self._step2a_test_verify(cell)
        if v is not None:
            self._stats.step2a_resolved += 1
            return v

        # Step 2b: Precondition 核查
        v = self._step2b_precondition_verify(cell)
        if v is not None:
            self._stats.step2b_resolved += 1
            return v

        # Step 3: LLM 仲裁
        v = self._step3_llm_arbitrate(cell)
        self._stats.step3_resolved += 1
        return v

    # ------------------------------------------------------------------
    # Step 1: 被动信号检查
    # ------------------------------------------------------------------
    def _step1_passive_signals(self, cell: Cell) -> Optional[Verdict]:
        """查询该 cell 近 N 个 episode 的引用记录 (从 op log)。

        - 有引用且结果为负 (challenge) → 大概率 decayed → 跳到 Step 3
        - 有引用且结果为正 (reference) → valid
        - 无引用 → 无法判定,进 Step 2
        """
        history = self.tree_store.oplog.get_cell_history(cell.id)
        # 过滤 UPDATE_ENERGY op,看 reason
        ref_count = 0
        chal_count = 0
        for entry in history:
            if entry.op != OpEnum.UPDATE_ENERGY:
                continue
            reason = entry.payload.get("reason", "")
            if reason == "reference":
                ref_count += 1
            elif reason == "challenge":
                chal_count += 1

        if ref_count > 0 and chal_count == 0:
            self._stats.verdicts["valid"] += 1
            return Verdict(
                result="valid",
                reason=f"passive signal: {ref_count} positive references, 0 challenges",
                step_reached=1,
                verifier_name="passive_signal",
            )
        if chal_count > 0 and ref_count == 0:
            # 有挑战但无正面引用 → 跳到 Step 3 确认
            return None  # fall through to Step 2/3
        if ref_count > 0 and chal_count > 0:
            # 混合信号 → 无法判定,进 Step 2
            return None
        # 无引用 → 进 Step 2
        return None

    # ------------------------------------------------------------------
    # Step 2a: 测试验证
    # ------------------------------------------------------------------
    def _step2a_test_verify(self, cell: Cell) -> Optional[Verdict]:
        """检查 cell.evidence 中是否有 test_id:xxx → 尝试跑该测试。"""
        test_ids = self._extract_test_ids(cell)
        if not test_ids:
            return None  # 无 test_id → 进 Step 2b

        if self.test_runner is None:
            return None  # 无 test runner → 进 Step 2b

        all_pass = True
        any_fail = False
        for test_id in test_ids:
            result = self.test_runner.run_test(test_id, self.repo_path)
            if result == "fail":
                any_fail = True
                self._stats.verdicts["decayed"] += 1
                return Verdict(
                    result="decayed",
                    reason=f"test {test_id} failed",
                    step_reached=2,
                    evidence=f"test_id={test_id}, result=fail",
                    verifier_name="test_runner",
                )
            elif result == "unknown":
                all_pass = False

        if all_pass:
            self._stats.verdicts["weak_valid"] += 1
            return Verdict(
                result="weak_valid",
                reason=f"all {len(test_ids)} test(s) passed",
                step_reached=2,
                evidence=", ".join(test_ids),
                verifier_name="test_runner",
            )
        # 部分无法跑 → 进 Step 2b
        return None

    # ------------------------------------------------------------------
    # Step 2b: Precondition 核查
    # ------------------------------------------------------------------
    def _step2b_precondition_verify(self, cell: Cell) -> Optional[Verdict]:
        """遍历 cell.context_preconditions,对有 verify_hint 的执行 verifier。"""
        # 先检查是否有任何带 verify_hint 的 precondition
        has_hints = any(p.verify_hint is not None for p in cell.context_preconditions)
        if not has_hints:
            return None  # 无可验证 precondition → 进 Step 3

        result: CellVerifyResult = self.verifier_registry.verify_cell(
            cell, repo_path=self.repo_path,
        )

        if result.overall == "valid":
            self._stats.verdicts["valid"] += 1
            verifier_names = [
                d.result.evidence or d.kind for d in result.details
            ]
            return Verdict(
                result="valid",
                reason=f"all {len(result.details)} precondition(s) verified",
                step_reached=2,
                evidence="; ".join(verifier_names),
                verifier_name="precondition_verify",
            )
        elif result.overall == "invalid":
            self._stats.verdicts["decayed"] += 1
            failed = [d for d in result.details if d.result.status == "invalid"]
            return Verdict(
                result="decayed",
                reason=f"{len(failed)} precondition(s) failed verification",
                step_reached=2,
                evidence="; ".join(d.result.evidence or d.assertion for d in failed),
                verifier_name="precondition_verify",
            )
        # inconclusive → 进 Step 3
        return None

    # ------------------------------------------------------------------
    # Step 3: LLM 深度裁决
    # ------------------------------------------------------------------
    def _step3_llm_arbitrate(self, cell: Cell) -> Verdict:
        """LLM 最终裁决:输出限定 valid/weak_valid/decayed/uncertain。"""
        system_prompt = (
            "You are a decay sentinel evaluating whether a knowledge cell is still valid. "
            "Given the cell's decision, rationale, preconditions, and evidence, determine "
            "if the cell has decayed (is no longer valid) or is still valid. "
            'Respond with a JSON object: {"result": "valid"|"weak_valid"|"decayed"|"uncertain", "reason": "..."}'
        )
        user_prompt = (
            f"Decision: {cell.decision}\n"
            f"Rationale: {cell.rationale}\n"
            f"Preconditions: {[p.assertion for p in cell.context_preconditions]}\n"
            f"Evidence: {cell.evidence}\n"
            f"Energy: {cell.energy}\n"
            f"Ring: {cell.ring}\n"
        )

        from tree_harness.core.llm_client import parse_llm_json

        raw = self.llm_client.complete(system_prompt, user_prompt)
        parsed = parse_llm_json(raw)

        result = parsed.get("result", "uncertain")
        if result not in ("valid", "weak_valid", "decayed", "uncertain"):
            result = "uncertain"

        reason = parsed.get("reason", "LLM arbitration")

        self._stats.verdicts[result] += 1
        return Verdict(
            result=result,
            reason=reason,
            step_reached=3,
            evidence=raw[:200] if isinstance(raw, str) else None,
            verifier_name="llm_arbitrate",
        )

    # ------------------------------------------------------------------
    # Signal-level 副作用
    # ------------------------------------------------------------------
    def _apply_signal_side_effect(
        self, verdict: Verdict, cell_id: str, episode_id: Optional[str],
    ) -> None:
        """Sentinel 内部触发的信号级副作用。

        - valid → EnergySystem.reference() (+delta_reference)
        - weak_valid → 无
        - decayed → 无 (OuterHarness 执行 quarantine)
        - uncertain → EnergySystem.decay_one(Δ=-0.05) + TreeStore.mark_for_review()
        """
        if verdict.result == "valid":
            self.energy_system.reference(cell_id, episode_id or "")
        elif verdict.result == "uncertain":
            self.energy_system.decay_one(cell_id, delta=-0.05, episode_id=episode_id)
            self.tree_store.mark_for_review(
                cell_id, flag=True, reason="uncertain_verdict", episode_id=episode_id,
            )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_test_ids(cell: Cell) -> List[str]:
        """从 cell.evidence 中提取 test_id:xxx 格式的测试 ID。"""
        ids: List[str] = []
        for ev in cell.evidence:
            m = re.match(r"test_id:(.+)", ev.strip())
            if m:
                ids.append(m.group(1).strip())
        return ids
