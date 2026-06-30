# Cambium Engine Spec

## 概述

Cambium Engine 是 Self-Evolution Operator Set 中 `crystallize` 算符的实现策略（同时承担 `connect` 算符在新建 cell 上的初始连边）。它本身**不是 harness 状态的所有者**，只是一个无状态算法服务，由 `OuterHarness.after_step()` 在每步轨迹翻译后调用，从 StandardStep 中蒸馏出结构化 cell 写入 TreeStore。

定位说明：Cambium 不直接维护任何 episode-local 状态，不感知三 hook 的存在；它只回答两件事——"这段轨迹值不值得蒸馏"、"应该连到哪些已有 cell"。是否调用、何时调用、调用频率全部由 OuterHarness 决定。这一约束确保了 outer_harness.md 中算符封闭性 Claim 成立。

## 输入

OuterHarness 在 `after_step` hook 中以**步级**调用 Cambium——每步喂一个 `StandardStep`，by Cambium 决定是否蒸馏。这是 spec 默认的工作节拍。episode-level `crystallize(trajectory)` 保留为离线批处理入口（数据集回放 / 烟测 / 训练管线）。

```python
@dataclass
class StandardStep:
    task_id: str                    # 所属 task（"django__django-16379"）
    episode_id: str                 # 所属 episode
    step_index: int                 # 步序号
    repo: str                       # "django/django"
    action_summary: str             # 本步动作的自然语言摘要（≤ 1 句话）
    observation_summary: str        # 本步 observation 摘要
    patch_delta: Optional[str]      # 本步引入的 diff（如有）
    test_results: dict              # 本步触发的测试 {test_id: pass/fail}（如有）
    outcome_so_far: Literal["pending", "pass", "fail", "error"]
    duration_seconds: float
    token_usage: int


@dataclass
class StandardTrajectory:
    """episode-level 输入；OuterHarness 不使用，仅用于离线批处理。"""
    task_id: str                    # "django__django-16379"
    task_description: str           # issue 摘要
    repo: str                       # "django/django"
    base_commit: str
    outcome: Literal["pass", "fail", "error"]
    patches: list[str]              # 最终 diff
    test_results: dict              # {test_id: pass/fail}
    key_actions: list[str]          # 筛选后的关键动作（≤10条）
    duration_seconds: float
    token_usage: int
    steps: list[StandardStep] = field(default_factory=list)  # 可选展开
```

## 三步管线

```
StandardStep (or StandardTrajectory)
   → [Step A: Crystallize] → [Step B: Dedup] → [Step C: Connect] → cells in tree
```

## 接口定义

```python
class CambiumEngineProtocol(Protocol):
    def __init__(self, tree_store: TreeStore, energy_system: EnergySystem,
                 llm_client: LLMClient, config: CambiumConfig):
        ...

    # --- step-level 入口（OuterHarness.after_step 在三 hook 内使用） ---
    def should_crystallize(self, step: StandardStep) -> bool:
        """准入判断：本步是否值得蒸馏。
        OuterHarness 通过这个布尔短路决定是否调 crystallize_step。
        默认实现：调用 _worth_extracting_step。"""
        ...

    def crystallize_step(self, step: StandardStep) -> list[Cell]:
        """单步蒸馏。完整执行 Step A → B → C，返回本步新生 cell（可能为空）。

        REINFORCE 命中已有 cell 时**不**出现在返回值中——这条按照
        EpisodeReport.new_cells_count 的定义"本 episode 新 crystallize 的 cell 数"。
        REINFORCE 触发的 energy 增量在 Step B 内部通过 energy_system.reference() 完成。
        """
        ...

    def connect_new_cells(self, new_cells: list[Cell]) -> None:
        """批量为 new_cells 建立外向 ray（对应 connect 算符的产物）。

        Step C 已经在 crystallize_step 内对每个 cell 做了一次 connect；
        connect_new_cells 是一个**批量再连边**入口，供 OuterHarness 在一步晶化
        出多个 cell 后做跨 new cell 的相互 connect（避免互相孤立）。
        非必需——若 OuterHarness 不调用，则只依赖 Step C 的单 cell connect。
        """
        ...

    # --- episode-level 入口（离线批处理 / 烟测用，OuterHarness 不调用） ---
    def crystallize(self, trajectory: StandardTrajectory) -> list[Cell]:
        """对完整 trajectory 批量蒸馏；内部等价于对每个 step 顺序调
        crystallize_step（外加 trajectory 级别的 outcome 准入过滤）。"""
        ...

    # --- 内部辅助 ---
    def _worth_extracting_step(self, step: StandardStep) -> bool:
        """step 级准入。"""
        ...

    def _worth_extracting(self, trajectory: StandardTrajectory) -> bool:
        """trajectory 级准入（批处理入口用）。"""
        ...

    def _llm_extract_step(self, step: StandardStep) -> list[CandidateCell]:
        """Step A: LLM 提取候选 cell（步级）"""
        ...

    def _dedup(self, candidate: CandidateCell) -> Union[Literal["INSERT_NEW"], tuple[Literal["REINFORCE"], str]]:
        """Step B: 去重判定"""
        ...

    def _connect(self, new_cell: Cell) -> None:
        """Step C: 建立 ray 连接"""
        ...
```

## 配置

```python
@dataclass
class CambiumConfig:
    """CambiumEngine 顶层配置；持有 dedup 与 connector 的嵌套子配置。

    Dedup / Connector 模块各自构造时从 CambiumConfig.dedup / .connector
    取子配置——这样 cambium_engine 是唯一 config 注入点，不会出现 config
    在多个模块间漂移的问题。
    """
    # 嵌套子配置（详细字段见 dedup.md / connector.md）
    dedup: "DedupConfig" = field(default_factory=lambda: DedupConfig())
    connector: "ConnectorConfig" = field(default_factory=lambda: ConnectorConfig())

    # 新 cell 初值
    initial_energy: float = 0.5
    initial_maturity: float = 0.0
```

> dedup_threshold_exact / dedup_threshold_similar / dedup search_top_k 等字段已迁移到 `DedupConfig`（见 dedup.md）；ray_search_top_k / ray_max_per_cell 字段已迁移到 `ConnectorConfig`（见 connector.md）。

## Step A: Crystallize 准入门槛

step-level（OuterHarness 在三 hook 内使用）：

```python
def _worth_extracting_step(self, step: StandardStep) -> bool:
    if step.outcome_so_far == "error":
        return False
    # pending 步未必无知识——但只有在 patch 或 test 出现状态变化时蒸馏
    if step.outcome_so_far == "pending":
        if not step.patch_delta and not step.test_results:
            return False
    if step.outcome_so_far == "fail" and not self._step_has_clear_lesson(step):
        return False
    if self._is_mechanical_step(step):
        return False
    return True
```

trajectory-level（离线批处理用）：

```python
def _worth_extracting(self, trajectory: StandardTrajectory) -> bool:
    if trajectory.outcome == "error":
        return False
    if trajectory.outcome == "fail" and not self._has_clear_lesson(trajectory):
        return False
    if self._is_mechanical(trajectory):
        return False
    return True
```

判定规则总览：

- `outcome == "error"` → 不提取（agent 环境错误，非知识性问题）
- `outcome == "fail"` 且无明确教训 → 不提取
- 纯机械操作（key_actions 只有 1-2 步简单命令；step 级别检查 action_summary 是否属于 `{ls, cd, cat, pwd}`）→ 不提取
- 其他 → 提取

## Step B: Dedup 逻辑

```
similarity > 0.95 → REINFORCE（完全重复）
similarity ∈ (0.85, 0.95] → LLM 仲裁（same → REINFORCE, different → INSERT_NEW）
similarity <= 0.85 → INSERT_NEW
```

REINFORCE 行为：不创建新 cell，给已有 cell 触发 energy_system.reference()。

## Step C: Connect 规则

1. 对新 cell 做 vec_search (top_k=10, threshold=0.5)
2. 过滤：只保留 ring >= new_cell.ring 的（外→内原则）
3. 取 top 5，计算 weight = similarity * (1 + 0.2 * domain_overlap)
4. 为每个 target 建立 RAY 边

## 蒸馏 Prompt（Step A 使用）

```
从以下 agent 执行记录中提取可复用的决策知识。

执行记录：
- Task: {task_description}
- Outcome: {outcome}
- Key actions: {key_actions}
- Patches: {patches_summary}
- Test results: {test_results_summary}

要求每条知识输出 JSON 格式：
{{
  "decision": "具体做了什么决策",
  "rationale": "为什么这样决策",
  "preconditions": [{{"kind": "...", "assertion": "...", "verify_hint": ...}}],
  "evidence": ["test_id:...", "file:..."],
  "domain_tags": ["tag1", "tag2"]
}}

规则：
- 只提取"下次遇到类似情况能直接复用"的知识
- 跳过纯机械操作（如 git add, cd 到目录）
- 对 config/dependency/code_invariant 类 precondition 尽量给出 verify_hint
- 避免过于具体（绑定特定文件行号）或过于宽泛（适用于任何项目）
```

## 测试用例

1. outcome="error" 的 trajectory → 返回空列表
2. outcome="pass" 的有效 trajectory → 返回 1~3 个 cell
3. 两次输入相同 trajectory → 第二次全部 REINFORCE（不新建）
4. 新 cell 的 ray 方向正确（全部指向同层或更内层）
5. 新 cell 的 ray 数量 ≤ 5
6. REINFORCE 后对应 cell 的 energy 增加了 δ_reference
7. 纯机械 trajectory（只有 `ls` 和 `cat`）→ 不提取
