"""Cell 数据模型 —— 系统的基本认知单位。

一个 Cell 表示 "在某情境下做出某选择因为某理由" 的 (Context, Decision, Rationale) 三元组。
对应 spec: docs/specs/cell_model.md
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal, List, Any

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
RingLevel = Literal["L0", "L1", "L2", "L3", "L4"]
CellStatus = Literal["active", "quarantined", "superseded", "archived"]
CellSource = Literal["distilled", "user_directive", "seed"]

# ---------------------------------------------------------------------------
# Ring 与 Maturity 映射阈值 (low, high)
# ---------------------------------------------------------------------------
MATURITY_RING_RANGES: dict[str, tuple[float, float]] = {
    "L0": (0.00, 0.15),
    "L1": (0.15, 0.40),
    "L2": (0.40, 0.65),
    "L3": (0.65, 0.85),
    "L4": (0.85, 1.00),
}

# 升层阈值: maturity >= 此值 → 升入该 ring (取自 RING_RANGES 的下界)
PROMOTE_THRESHOLDS: dict[str, float] = {
    "L1": 0.15,   # L0→L1
    "L2": 0.40,   # L1→L2
    "L3": 0.65,   # L2→L3
    "L4": 0.85,   # L3→L4
}

# 降级阈值: maturity 低于此值 → 降一层 (key = 当前 ring)
DEMOTE_THRESHOLDS: dict[str, float] = {
    "L1": 0.05,   # L1→L0
    "L2": 0.30,   # L2→L1
    "L3": 0.55,   # L3→L2
    "L4": 0.75,   # L4→L3
}

# 升层-降层阈值差 = 0.10 (滞回带)

# 年轮从外到内的顺序
RING_ORDER: list[str] = ["L0", "L1", "L2", "L3", "L4"]


# ---------------------------------------------------------------------------
# 嵌套数据结构
# ---------------------------------------------------------------------------
@dataclass
class VerifyHint:
    """机器验证提示,供 Verifiers 使用。"""
    type: Literal["file_grep", "ast_query", "lockfile_query", "test_id_lookup", "env_check"]
    params: dict


@dataclass
class Precondition:
    """cell 生效的前置条件断言。assertion 永远以自然语言保留。"""
    kind: Literal["fact", "config", "dependency", "code_invariant", "test_existence", "convention"]
    assertion: str
    verify_hint: Optional[VerifyHint] = None


# 创建后不可变的字段 (公理六: 细胞不可修改,只能弃用和新生)
_IMMUTABLE_FIELDS = frozenset({"id", "decision", "rationale"})


@dataclass
class Cell:
    """系统的基本认知单位。"""

    id: str
    ring: RingLevel
    maturity: float          # 0.0 ~ 1.0
    energy: float            # 可为负

    # 内容三元组
    context_trigger_task: str       # 触发 task id
    context_domain: str             # 领域标签
    context_preconditions: List[Precondition]
    decision: str                   # NL 决策描述 (创建后不可改)
    rationale: str                  # NL 理由 (创建后不可改)

    # 元数据
    evidence: List[str]             # ["test_id:xxx", "commit:yyy", "file:zzz"]
    domain_tags: List[str]
    status: CellStatus
    source: CellSource

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    superseded_by: Optional[str] = None  # 指向新 cell id

    def __post_init__(self) -> None:
        # 行为契约 4: maturity 硬截断在 [0, 1]
        if self.maturity > 1.0:
            object.__setattr__(self, "maturity", 1.0)
        elif self.maturity < 0.0:
            object.__setattr__(self, "maturity", 0.0)

    def __setattr__(self, name: str, value) -> None:
        # 公理六: id / decision / rationale 创建后不可修改
        if name in _IMMUTABLE_FIELDS and name in self.__dict__:
            raise AttributeError(
                f"'{name}' is immutable after creation (axiom 6: cells cannot be modified)"
            )
        super().__setattr__(name, value)


# ---------------------------------------------------------------------------
# ID 生成
# ---------------------------------------------------------------------------
def generate_cell_id() -> str:
    """生成全局唯一 cell id: cell-{timestamp}-{random_suffix}"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"cell-{ts}-{suffix}"


# ---------------------------------------------------------------------------
# 初始值规则 (cell_model.md 初始值表)
# ---------------------------------------------------------------------------
_SOURCE_DEFAULTS: dict[str, dict] = {
    "distilled":      {"ring": "L0", "maturity": 0.0,  "energy": 0.5},
    "user_directive": {"ring": "L4", "maturity": 0.85, "energy": 1.0},
    "seed":           {"ring": "L0", "maturity": 0.0,  "energy": 0.5},
}


def create_cell(
    *,
    source: CellSource = "distilled",
    trigger_task: str = "",
    domain: str = "",
    decision: str = "",
    rationale: str = "",
    preconditions: Optional[List[Precondition]] = None,
    evidence: Optional[List[str]] = None,
    domain_tags: Optional[List[str]] = None,
    ring: Optional[RingLevel] = None,
    maturity: Optional[float] = None,
    energy: Optional[float] = None,
    cell_id: Optional[str] = None,
) -> Cell:
    """创建 cell,按 source 应用初始值规则。

    | 字段       | distilled      | user_directive |
    |-----------|----------------|----------------|
    | ring      | L0             | L4             |
    | maturity  | 0.0            | 0.85           |
    | energy    | 0.5            | 1.0            |
    | status    | active         | active         |
    """
    defaults = _SOURCE_DEFAULTS[source]
    return Cell(
        id=cell_id if cell_id is not None else generate_cell_id(),
        ring=ring if ring is not None else defaults["ring"],
        maturity=maturity if maturity is not None else defaults["maturity"],
        energy=energy if energy is not None else defaults["energy"],
        context_trigger_task=trigger_task,
        context_domain=domain,
        context_preconditions=preconditions if preconditions is not None else [],
        decision=decision,
        rationale=rationale,
        evidence=evidence if evidence is not None else [],
        domain_tags=domain_tags if domain_tags is not None else [],
        status="active",
        source=source,
    )


# ---------------------------------------------------------------------------
# Phase 2: 轨迹数据结构 (cambium_engine.md / trajectory_adapter.md)
# ---------------------------------------------------------------------------
@dataclass
class StandardStep:
    """单步标准化轨迹 — OuterHarness after_step 的输入。

    TrajectoryAdapter.convert_step() 产出, CambiumEngine.crystallize_step() 消费。
    """
    task_id: str
    episode_id: str
    step_index: int
    repo: str
    action_summary: str             # 本步动作自然语言摘要 (≤1句)
    observation_summary: str        # 本步 observation 摘要
    patch_delta: Optional[str]      # 本步引入的 diff (如有)
    test_results: dict              # {test_id: "pass"/"fail"} (如有)
    outcome_so_far: Literal["pending", "pass", "fail", "error"]
    duration_seconds: float
    token_usage: int


@dataclass
class StandardTrajectory:
    """episode-level 标准化轨迹 — 离线批处理入口用。

    OuterHarness 在线流程不使用此类; 仅用于 CambiumEngine.crystallize() 离线批处理。
    """
    task_id: str
    task_description: str
    repo: str
    base_commit: str
    outcome: Literal["pass", "fail", "error"]
    patches: List[str]
    test_results: dict
    key_actions: List[str]           # 筛选后的关键动作 (≤10条)
    duration_seconds: float
    token_usage: int
    steps: List[StandardStep] = field(default_factory=list)


@dataclass
class CandidateCell:
    """LLM 蒸馏出的候选 cell (Step A 产物, 尚未入树)。

    包含 cell 的内容三元组, 但没有系统分配的字段 (id, ring, energy 等)。
    这些字段在 Step B (dedup → INSERT_NEW) 后由 create_cell() 补全。
    embedding 在 Step A 后由 CambiumEngine 用 embedder 预计算 (dedup/connector 需要)。
    """
    decision: str
    rationale: str
    preconditions: List[Precondition]
    evidence: List[str]
    domain_tags: List[str]
    context_trigger_task: str = ""
    context_domain: str = ""
    embedding: List[float] = field(default_factory=list)
