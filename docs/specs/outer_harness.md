# OuterHarness Spec

## 概述

OuterHarness 是 Tree Harness 系统对外的**唯一入口**。它把一个常规 coding agent harness（如 SWE-agent、Aider、OpenHands）包成一个**结构化自演化的外层 harness**（structurally self-evolving outer harness），在 long-horizon 编码任务上同时控制 stability、context drift、control lag 三项可靠性属性。

OuterHarness 不参与 LLM 调用、不执行 tool、不管理 sandbox——这些由被包裹的 **inner harness** 负责。OuterHarness 自身的职责是通过三个 hook 完成 harness 级别的自演化：

1. **before_step**：基于当前 trajectory 状态动态组装 inner harness 的输入 context
2. **after_step**：观察单步轨迹，触发结构化自演化算符并产出对下一步的纠正信号（control feedback）
3. **after_episode**：以 episode 为节拍触发 harness 状态的批量演化（promotion / decay / consolidation）

## 设计原则

- **Outer harness 而非 memory module**：拥有 prompt 注入点的所有权、验证反馈环的所有权、episode 节拍的所有权三件套。Tree 内部确实包含一个 cell 存储 + 检索子系统，但它是实现细节，不是 framing。
- **结构化自演化 vs 自由改写**：所有 harness 状态的演化必须可被表达为一组**固定算符**的组合（见 Self-Evolution Operator Set）。这条原则区别于让 LLM 自由改写 scaffold 代码的方案，以可解释性与稳定性换取部分灵活性，规避 coupled co-evolutionary Goodhart。
- **Trajectory-conditioned adaptation**：所有 hook 的行为都是 trajectory 状态的函数。before_step 的 budget 分配、after_step 的 verification 抽样、after_episode 的 lignification 都基于轨迹观测做决策。
- **对 inner harness 零侵入**：通过 Protocol 适配，不要求 inner harness 实现任何 Tree 相关接口；inner 可被替换为 SWE-agent / OpenHands / mini-SWE-agent，验证 harness portability。
- **三 hook 单向数据流**：before_step → inner.step → after_step；多 step 后 after_episode。任何 hook 不直接调用其他 hook。
- **所有 Tree 内部模块的访问都收敛到三个 hook**：上层调用者只见 OuterHarness，不需要知道 Cambium / Decay / Lignification 的存在。

## 对标工作

| 类别 | 代表工作 | Tree 与其差异 |
|------|----------|---------------|
| Free-form self-evolving | SIA (arXiv:2605.27276) | SIA 由 Feedback-Agent 自由改写 scaffold 与 weights；Tree 约束演化在固定算符空间内，不动 weights，可解释性更高，规避 Goodhart |
| Harness reliability framework | Harness Card (arXiv:2605.23950) | 该工作定义 stability/context drift/control lag 三属性并量化 HV/MV≈7.8×；Tree 的三 hook 一一对应这三属性（见下文映射） |
| Static harness | SWE-agent / Aider / OpenHands | 这些 harness 每 episode 冷启动、scaffold 不演化；Tree 作为它们的外层包装，不取代它们作为 inner harness 的角色 |
| Continual harness adaptation | Continual Harness (arXiv:2605.09998) | 该路线 adapt prompt 模板与 tool 选择策略；Tree adapt 结构化项目知识图，粒度更细、半衰期更长 |

## 模块定位

OuterHarness 不取代 TreeHarnessRunner，而是**取代它原本承担的"协调编排"职责**。Runner 在新框架下只负责：
- 实验流程驱动（多 task 串行 / checkpoint / 日志）
- OuterHarness 与 InnerHarness 的组装

run_episode 内部的"调 injector → 调 agent → 调 cambium → 调 decay"这套流程，搬到 OuterHarness 的三个 hook 里。

## Self-Evolution Operator Set

Tree Harness 的核心 framing 是 **structurally self-evolving**——harness 在 episode 之间发生状态变化，但所有变化必须可表达为下列**固定算符集**中元素的有限组合。这条约束是 Tree 区别于 SIA 类自由演化路线的核心。

### 算符定义

记 harness 状态为 `H = (C, R, E, ρ)`，其中 C 为 cell 集合、R 为 ray 集合、E 为 energy 向量、ρ 为 ring 分配函数。

| 算符 | 签名 | 语义 | 副作用 |
|------|------|------|--------|
| `crystallize` | `(trajectory_segment) → Cell` | 从轨迹片段中抽取一条 (Context, Decision, Rationale) 三元组形成新 cell，初始 ring=L0、energy=E₀ | 向 C 中插入新元素 |
| `connect` | `(cell_a, cell_b, weight, source_type) → Ray` | 在两个 cell 之间建立单向 ray（外→内），权重 ∈ [0,1]。Ray 同时承担两件事：(a) retrieval 时的相关性传导通道（cell_b 命中可拉高 cell_a 的召回分），(b) **quarantine 信号传播的拓扑底座**（见下文）。 | 向 R 中插入新元素 |
| `promote` | `(cell, target_ring) → Cell'` | 提升 cell 至更内层 ring（受 hysteresis 约束，无跳级） | 修改 ρ(cell) |
| `quarantine` | `(cell, evidence) → Cell'` | 将 cell 标记为 inactive，关联否证证据，断开其外向 active ray。同时沿入向 ray（incoming）一跳传播 warning-weight 加成给邻居 cell——下一次 before_step 在生成 warnings 段时，对受影响邻居优先选取 | 修改 cell.status，修改 R，写入 neighbor_warning_queue |
| `decay` | `(cell, Δ) → Cell'` | 按 ring 对应衰减率削减 cell.energy | 修改 E |

`merge` 与 `split` 视为 `crystallize + quarantine` 的复合算符；`reference` 与 `challenge` 是 `decay` 的反向特例（增能 vs 减能）。

### Ray 的双重职责

Pivot 到 harness framing 后，Ray 必须被同时挂在两个职责上才不沦为旧 memory framing 的 vestigial 抽象：

1. **Connect 算符的产物**：Ray 是 `connect(cell_a, cell_b, ...)` 唯一允许的输出形式。R 集合本身就是 connect 算符的 range，这条让 Ray 在算符层面有不可替代的位置。
2. **Quarantine 的信号传播基底**：当 `quarantine(cell, evidence)` 触发时，沿 cell 的 incoming ray 一跳（仅一跳，避免雪崩）将受影响邻居写入 `neighbor_warning_queue`。下一次 `before_step` 在生成 warnings 段时，从该队列中按 ray.weight 排序优先注入"邻接 cell 同样需要警惕"的提示文本。这条把 Ray 直接绑到 Harness Card 的 **control lag** 属性上（异常→邻居预警的传播延迟，由 ray 拓扑决定上界）。

不变量 **I-Ray**：任何 Ray 必须可同时为 retrieval 与 quarantine propagation 服务；若某条 Ray 在两类用途上权重均为 0，应被 quarantine 算符回收（视为 dead ray）。这条约束防止 Ray 被当作单纯的"检索图边"重新塌缩回 memory framing。

`neighbor_warning_queue` 是 episode-local 状态（不持久化），生命周期与 `pending_warnings` 一致，由 after_episode hook 清理。

### 算符的封闭性

**Claim**：OuterHarness 三个 hook 中触发的任何 harness 状态变更，都可表示为上述 5 个原子算符的有限组合。其他模块（Cambium / Decay Sentinel / Lignification）只是这些算符的策略性调用，不引入新的状态变更类型。

这条性质对外是论文中的 sharp formal claim，对内是回归测试的不变量——任何 PR 引入新的状态变更路径都必须先证明可被现有算符表达，否则需扩展算符集并更新论文 formalism。

### 算符与三 hook 的映射

```
before_step:   read-only over H
after_step:    {crystallize, connect, quarantine}
after_episode: {promote, decay}
```

before_step 是只读 hook，不引入演化。这保证了同一个 (task, step_index) 在重放时能得到一致的注入 context（确定性回放是 oplog 设计的基础）。

### 算符与 Harness Card 三属性的映射

| Harness Card 属性 | 主要承担算符 | 配合机制 |
|-------------------|-----------|---------|
| **Stability**（朝目标稳步前进，不震荡） | `promote` / `decay` | Ring hysteresis（0.10 dead zone）+ Lignification 的 episode 节拍 |
| **Context drift**（任务关键信息丢失） | `crystallize` / `quarantine` | L3/L4 pinned 段无条件注入 + Decay Sentinel funnel verification |
| **Control lag**（异常检测到纠正信号的延迟） | `quarantine` | after_step 内立即生成 warning 注入下一次 before_step（latency=1 step）+ ray 一跳传播给邻居（latency=1 step + 1 hop） |

这张表是论文实验部分的 metric 骨架——每个属性对应一个量化指标（见 metrics.md 中的 Harness-Level Metrics）。

### 不变量

- I-Op1：所有对 TreeStore 的写操作都来自上述 5 个算符之一，违反者视为 bug。
- I-Op2：算符在 oplog 中以语义级别记录（op_type ∈ {CRYSTALLIZE, CONNECT, PROMOTE, QUARANTINE, DECAY}），不记录底层 SQL/Cypher。
- I-Op3：算符执行顺序：
  - **after_step**：`(crystallize → connect → reference[ray.activate 命中的 cell] → quarantine)`。reference 在 quarantine 之前，因为 quarantine 可能切掉 cell 的 incoming ray，影响 reference 累计计数。
  - **after_episode**：`(reference[outcome==pass 时给 injected 的 cell] → decay → promote)`。reference 在 decay 之前——reference 是 challenge 的反向特例，必须先把"本 episode 的 +δ"加进 E，再让 decay 按 ring rate 乘性衰减；颠倒会导致 reference 被同 episode 的 decay 一次性吃掉。promote 在最后——它读取 EnergySystem.update_all_maturity 之后的 maturity 做升降层判定。
  - 任何对该顺序的违背必须在调用 spec 中显式说明（带 reason）。
- I-Pin：ContextBlock 的 pinned_text 与 warnings 段必须被对应 marker（`pin_open_tag/close_tag` / `warning_open_tag/close_tag`）原样包裹；OuterHarness 在 `wrap()` 时读 `inner.capabilities()` 并把 `supports_pin_marker / supports_warning_marker` 写入每条 EpisodeRecord 的 metadata，使下游 H2 (Context Retention Score) 可按 inner 能力分层报告。

## 接口定义

```python
from typing import Protocol, Optional, Any
from dataclasses import dataclass, field

# --- Inner Boundary Types ---
# OuterHarness 与 InnerHarness 之间的最小数据契约。
# inner 可以扩展这些 dataclass 的字段（例如 SWE-agent 在 StepObservation 中
# 加 tool_calls / patch_diff 等），OuterHarness 只读这里声明的核心字段。

@dataclass
class Task:
    """单个 episode 的输入。"""
    task_id: str
    description: str                # 自然语言任务描述（注入 prompt 用）
    repo_path: str                  # 工作 repo 的本地路径（verifier 用）
    metadata: dict = field(default_factory=dict)  # 数据集字段（issue_id 等）


@dataclass
class StepObservation:
    """inner.step() 的返回值——单步执行结果。"""
    action: dict                    # inner 本步采取的动作（语义化字段）
    result: dict                    # 工具/环境返回（stdout/stderr/exit_code/...）
    is_terminal: bool = False       # inner 自报 episode 是否在本步终止
    outcome: Optional[str] = None   # 终止时填 "pass" | "fail" | "error"；中间步为 None
    raw_output: Any = None          # inner 原始输出，TrajectoryAdapter 解析用


class StepState(Protocol):
    """inner harness 维护的运行时状态。
    OuterHarness 只调用下列方法，不假设具体字段。

    augment / advance 返回新 state（不可变契约由 inner 自决；OuterHarness
    一律以返回值覆盖 state 引用）。
    """
    outcome: Optional[str]          # 终止后填 "pass" | "fail" | "error"

    def augment(self, context: "ContextBlock") -> "StepState":
        """把 before_step 产出的 context 融合进 inner 的下一次 LLM 调用。"""
        ...

    def advance(self, observation: StepObservation) -> "StepState":
        """以 step 返回值推进 state。"""
        ...

    def snapshot(self) -> dict:
        """供 StepRecord.state_before 序列化用的浅快照。"""
        ...


# --- Inner Harness 协议（OuterHarness 对 inner 的最小要求） ---
class InnerHarnessProtocol(Protocol):
    def step(self, state: "StepState") -> "StepObservation":
        """执行一步：基于 state 生成 action 并返回 observation"""
        ...

    def is_terminal(self, state: "StepState") -> bool:
        """判断 episode 是否结束"""
        ...

    def reset(self, task: "Task") -> "StepState":
        """开启新 episode，返回初始 state"""
        ...

    def capabilities(self) -> "InnerCapabilities":
        """声明 inner 自身能力，用于 OuterHarness 决定降级策略与实验日志记录"""
        ...


@dataclass
class InnerCapabilities:
    """inner harness 自报能力，OuterHarness 在 wrap 时读取一次"""
    supports_pin_marker: bool         # 是否在 history compaction 中识别 <|PINNED_DO_NOT_COMPACT|>
    supports_warning_marker: bool     # 是否在 history compaction 中识别 <|WARNING_DO_NOT_COMPACT|>
    history_window_tokens: int        # inner 内部 LLM call 的最大窗口（用于注入预算上限）
    has_internal_compaction: bool     # inner 是否会做 history truncation / summarization


# --- 三个 hook 的数据契约 ---
@dataclass
class ContextBlock:
    """before_step 的输出，注入 inner harness 的下一次 LLM 调用

    Pin Marker Protocol:
        pinned_text 与 warnings 段必须以下列标记字符串包裹（marker 由
        OuterHarnessConfig.pin_open_tag / pin_close_tag 配置，默认见下）：

            <|PINNED_DO_NOT_COMPACT|> ... pinned 内容 ... <|/PINNED|>
            <|WARNING_DO_NOT_COMPACT|> ... warnings 内容 ... <|/WARNING|>

        InnerHarness 的 history compaction / summarization 模块若识别该
        marker，必须原样保留所包裹文本（即使触发上下文裁剪），不得改写、
        摘要或丢弃。不识别 marker 的 inner harness 视为"尽力而为"，
        Tree 不强制要求其支持——但实验日志需记录 inner_supports_pin_marker
        布尔值用于结果解释。
    """
    pinned_text: str              # L3/L4 公理段（被 PIN marker 包裹，必注入且不可压）
    relevant_text: str            # L0/L1/L2 相关经验段（无 marker，inner 可自行压缩）
    warnings: list[str]           # 来自上一步 quarantine 的告警（被 WARNING marker 包裹）
    injected_cell_ids: list[str]  # 本次注入的 cell id（用于 episode 末反馈）
    token_count: int              # 估算 token 数
    budget_used: dict             # {"L0":..., "L1":..., "L4":...} 实际占用
    pin_open_tag: str = "<|PINNED_DO_NOT_COMPACT|>"
    pin_close_tag: str = "<|/PINNED|>"
    warning_open_tag: str = "<|WARNING_DO_NOT_COMPACT|>"
    warning_close_tag: str = "<|/WARNING|>"


@dataclass
class StepRecord:
    """after_step 的输入，描述刚完成的一步"""
    episode_id: str
    step_index: int
    state_before: dict
    action: dict
    observation: dict
    cells_referenced: list[str]   # 本步实际用到（被 LLM 关注）的 cell id


@dataclass
class StepReport:
    """after_step 的输出，告知 inner 本步副作用"""
    new_cells: list[str]                # 本步新晶化的 cell id
    quarantined_cells: list[str]        # 本步被验证失败隔离的 cell id
    warnings_for_next_step: list[str]   # 注入到下一次 before_step 的告警


@dataclass
class EpisodeRecord:
    """after_episode 的输入"""
    episode_id: str
    task: "Task"
    outcome: str                  # "pass" | "fail" | "error"
    steps: list[StepRecord]
    duration_seconds: float
    token_usage: int


@dataclass
class EpisodeReport:
    """after_episode 的输出，量化本 episode 的熵释放。

    字段命名与 `metrics.md` 中 entropy_release_per_episode / harness 指标
    一一对应；新增字段必须同步更新 metrics.md。
    """
    promoted: list[tuple[str, str, str]]   # (cell_id, from_ring, to_ring)
    demoted: list[tuple[str, str, str]]
    decayed_below_threshold: list[str]     # 衰减到 quarantine 边缘的 cell id
    # 计数字段（metrics 消费口径）
    new_cells_count: int                   # 本 episode 新 crystallize 的 cell 数
    compressed_count: int                  # 因 merge / supersede 被合并的 cell 数
    quarantined_count: int                 # 本 episode 被 quarantine 算符隔离的 cell 数
    decayed_count: int                     # 本 episode 衰减到阈值以下的 cell 数
    # 算符调用计数（OpLog 聚合，metrics H5 的来源）
    op_counts: dict[str, int]              # {"crystallize", "connect", "promote", "quarantine", "decay"}
    entropy_released: float                # 量化指标，见 metrics.md


# --- OuterHarness 主接口 ---
class OuterHarnessProtocol(Protocol):
    def __init__(
        self,
        tree_store: "TreeStore",
        context_injector: "ContextInjector",
        trajectory_adapter: "TrajectoryAdapter",
        cambium: "CambiumEngine",
        decay_sentinel: "DecaySentinel",
        lignification: "LignificationScheduler",
        energy_system: "EnergySystem",
        config: "OuterHarnessConfig",
    ):
        ...

    def wrap(self, inner: InnerHarnessProtocol) -> "WrappedHarness":
        """把 inner harness 包成具备三 hook 能力的复合 harness"""
        ...

    # 三个 hook（也可由外部直接调用，便于实验对照组）
    def before_step(self, task: "Task", step_index: int, episode_id: str) -> ContextBlock: ...
    def after_step(self, record: StepRecord) -> StepReport: ...
    def after_episode(self, record: EpisodeRecord) -> EpisodeReport: ...


class WrappedHarness(Protocol):
    """OuterHarness.wrap() 的返回，外观与 InnerHarness 兼容但多了 Tree 行为"""
    def run_episode(self, task: "Task") -> tuple[EpisodeRecord, EpisodeReport]:
        ...
```

## 配置

```python
@dataclass
class OuterHarnessConfig:
    # Budget allocation
    total_context_tokens: int = 4000
    pinned_ratio: float = 0.30          # L3+L4 固定占比
    relevant_ratio: float = 0.50        # L0+L1+L2 按相似度
    warnings_ratio: float = 0.20        # quarantine warning 段

    # Verification feedback loop
    enable_inline_warnings: bool = True  # 是否把 quarantine event 注入下一步
    max_warnings_per_step: int = 3

    # Episode tick
    decay_per_episode: bool = True       # 每 episode 末跑一轮全局 decay
    lignification_per_episode: bool = True
    maintenance_funnel_per_episode: bool = True  # 每 episode 末抽样 funnel 验证

    # Verification sampling
    funnel_sample_size: int = 10         # 每 episode 末抽多少个 candidate

    # 熵释放量化（写入 EpisodeReport.entropy_released）
    entropy_weight_compressed: float = 1.0
    entropy_weight_quarantined: float = 2.0
    entropy_weight_decayed: float = 0.5
```

## Hook 1：before_step

**职责**：在 inner harness 发起 LLM call 之前，构造一个 ring-stratified 的 ContextBlock 注入 prompt。

**流程**：

```python
def before_step(self, task, step_index, episode_id):
    # 1. 拉取上一次 after_step 留下的 warnings（首步为空）
    #    key 用 episode_id（与 after_step 写入端对齐；task_id 可能跨 episode 重复）
    warnings_entries = self._pending_warnings.pop(episode_id, [])
    neighbor_entries = self._neighbor_warning_queue.pop(episode_id, [])
    all_warnings = warnings_entries + neighbor_entries

    # 2. 计算 budget 分配
    total = self.config.total_context_tokens
    pinned_budget = int(total * self.config.pinned_ratio)
    relevant_budget = int(total * self.config.relevant_ratio)
    warnings_budget = total - pinned_budget - relevant_budget

    # 3. Pinned 段：拉取所有 L3/L4 active cell（不做相似度过滤）
    pinned_cells = self.tree_store.list_by_ring(["L3", "L4"], status="active")
    pinned_text = self.context_injector.format_pinned(
        pinned_cells, budget=pinned_budget
    )

    # 4. Relevant 段：L0/L1/L2 按相似度 + energy + ring_weight 打分
    relevant_ctx = self.context_injector.retrieve(
        task.description, task.repo,
        ring_filter=["L0", "L1", "L2"],
        token_budget=relevant_budget,
    )

    # 5. Warnings 段：合并直接 + 邻居告警，按 (is_direct, ray.weight, recency) 排序
    warnings_text = self.context_injector.format_warnings(
        all_warnings, budget=warnings_budget
    )

    # 6. 组装
    injected_ids = [c.id for c in pinned_cells] + relevant_ctx.cells
    return ContextBlock(
        pinned_text=pinned_text,
        relevant_text=relevant_ctx.formatted_text,
        warnings=warnings_text,
        injected_cell_ids=injected_ids,
        token_count=...,
        budget_used={...},
    )
```

**关键设计点**：

- L3/L4 是**项目公理**，不参与相似度筛选，**永远在 prompt 里**。这是 harness 和 memory module 的关键差异：memory 永远问"给我相关的"，harness 问"什么必须在场"。
- 三段 budget 静态分配，避免动态算法在长 horizon 下不稳定。
- `injected_cell_ids` 记录到 OuterHarness 内部状态，在 after_episode 时用于 reference event。

## Hook 2：after_step

**职责**：消化刚完成的一步，做三件事——蒸馏新 cell、抽样验证、生成下一步 warnings。

**流程**：

```python
def after_step(self, record):
    new_cells = []
    quarantined = []
    next_warnings = []

    # 1. 调 TrajectoryAdapter 把单步转标准格式
    step_obs = self.trajectory_adapter.convert_step(record)

    # 2. 调 Cambium 尝试 crystallize（不一定每步都产出 cell）
    if self.cambium.should_crystallize(step_obs):
        crystals = self.cambium.crystallize_step(step_obs)
        new_cells.extend(c.id for c in crystals)

    # 3. 抽样 funnel verification
    if self.config.maintenance_funnel_per_episode:
        candidates = self.energy_system.get_decay_candidates(
            limit=self.config.funnel_sample_size
        )
        # funnel_verify 返回 dict[cell_id, Verdict]；Verdict.result ∈
        # {"valid","weak_valid","decayed","uncertain"}。这里只对 result=="decayed"
        # 执行 quarantine 算符；其他 result 由 funnel_verify 内部已经写过
        # energy/maturity 副作用（见 decay_sentinel.md "裁决后果"表）。
        verdicts = self.decay_sentinel.funnel_verify(candidates)
        for cell_id, v in verdicts.items():
            if v.result == "decayed":
                # 真正执行 quarantine 算符（OuterHarness 拥有算符调用权，
                # decay_sentinel 只做判定不做写入）
                self.tree_store.quarantine(
                    cell_id, reason=v.reason, episode_id=record.episode_id,
                )
                quarantined.append(cell_id)
                # 关键：把 quarantine event 翻译成自然语言告警
                if self.config.enable_inline_warnings:
                    next_warnings.append(
                        self._warning_text(cell_id, v.reason)
                    )

    # 4. 写入 pending warnings 供下一次 before_step 消费
    self._pending_warnings[record.episode_id].extend(
        next_warnings[: self.config.max_warnings_per_step]
    )

    return StepReport(
        new_cells=new_cells,
        quarantined_cells=quarantined,
        warnings_for_next_step=next_warnings,
    )
```

**关键设计点**：

- **Verification feedback loop**：quarantine 事件不只是改 DB，还要回写下一步 prompt。这是 harness 才有的能力——memory module 没有"下一步 prompt"这个概念。
- 自然语言告警示例：`"WARNING: The previously injected guideline 'always use nulls_first=True' has been quarantined because verifier 'lockfile_query' detected the project no longer supports MySQL. Disregard it."`
- Crystallize 是 per-step 还是 per-episode 由 Cambium 自己决定（spec 中 `should_crystallize` 控制），OuterHarness 不假设。

## Hook 3：after_episode

**职责**：episode 终止时，跑一轮 ring 升降 + 全局 decay + reference event 回流，并量化熵释放。

**流程**：

```python
def after_episode(self, record):
    # 1. 注入回流：episode pass 时，给本 episode 注入过的 cell 喂 reference event
    if record.outcome == "pass":
        all_injected = self._collect_injected_ids(record.episode_id)
        for cell_id in set(all_injected):
            self.energy_system.reference(cell_id, record.episode_id)

    # 2. 全局 decay tick
    if self.config.decay_per_episode:
        self.energy_system.decay_all(record.episode_id)
        self.energy_system.update_all_maturity(record.episode_id)

    # 3. 木质化（ring 升降 + merge/split）
    promoted, demoted = [], []
    merged_n = 0
    if self.config.lignification_per_episode:
        result = self.lignification.run_maintenance_cycle(record.episode_id)
        promoted = result.promoted
        demoted = result.demoted
        merged_n = len(result.merged)
        # merge 视为 compress 算符的具体形式，计入 compressed_count
        self._episode_compressed_count[record.episode_id] += merged_n

    # 4. 量化熵释放 + 算符调用计数
    compressed_n = self._episode_compressed_count[record.episode_id]
    quarantined_n = self._episode_quarantine_count[record.episode_id]
    decayed_n = self._episode_decayed_count[record.episode_id]
    new_cells_n = self._episode_new_cells_count[record.episode_id]
    entropy = self._compute_entropy_released(
        compressed_n=compressed_n,
        quarantined_n=quarantined_n,
        decayed_n=decayed_n,
    )
    # op_counts 聚合自本 episode 内的 oplog 切片（按 5 算符语义聚合，
    # 底层 op_type → 算符的映射见 oplog.md "算符映射表"）。
    op_counts = self.oplog.count_by_op_type(episode_id=record.episode_id)

    # 5. 清理 episode-local 状态
    self._cleanup_episode_state(record.episode_id)

    return EpisodeReport(
        promoted=promoted,
        demoted=demoted,
        decayed_below_threshold=[...],
        new_cells_count=new_cells_n,
        compressed_count=compressed_n,
        quarantined_count=quarantined_n,
        decayed_count=decayed_n,
        op_counts=op_counts,
        entropy_released=entropy,
    )


def _compute_entropy_released(self, compressed_n, quarantined_n, decayed_n):
    c = self.config
    return (
        c.entropy_weight_compressed * compressed_n
        + c.entropy_weight_quarantined * quarantined_n
        + c.entropy_weight_decayed * decayed_n
    )
```

**关键设计点**：

- 木质化只在 episode 末发生，不在 step 中——这是 harness 的"节拍"，确保 ring 跃迁有自然的 batch 边界。
- 熵释放量化：compressed（重复经验合并）+ quarantined（错误经验淘汰）+ decayed（自然遗忘）。这是 metrics.md 中 `entropy_release_per_episode` 的来源。
- pass 时才喂 reference event，fail 不自动 challenge（failure 不一定是 cell 的错，由 Decay Sentinel 的 funnel verification 独立判定是否惩罚）。

## wrap() 行为契约

```python
def wrap(self, inner):
    outer = self

    class _Wrapped:
        def run_episode(self, task):
            state = inner.reset(task)
            steps = []
            step_index = 0
            episode_id = generate_episode_id()

            while not inner.is_terminal(state):
                # Hook 1
                ctx = outer.before_step(task, step_index, episode_id)
                state = state.augment(context=ctx)  # inner 决定如何用 ctx

                # Inner step
                obs = inner.step(state)

                # 构造 step record
                record = StepRecord(
                    episode_id=episode_id,
                    step_index=step_index,
                    state_before=state.snapshot(),
                    action=obs.action,
                    observation=obs.result,
                    cells_referenced=ctx.injected_cell_ids,
                )

                # Hook 2
                step_report = outer.after_step(record)
                steps.append(record)

                state = state.advance(obs)
                step_index += 1

            # Hook 3
            ep_record = EpisodeRecord(
                episode_id=episode_id,
                task=task,
                outcome=state.outcome,
                steps=steps,
                duration_seconds=...,
                token_usage=...,
            )
            ep_report = outer.after_episode(ep_record)
            return ep_record, ep_report

    return _Wrapped()
```

**关键约束**：

- inner 完全不感知 Tree 的存在，只看到一个被 augment 过的 state。
- `state.augment(context=ctx)` 由 inner harness 自行实现——OuterHarness 不假设 prompt 拼接方式。对 SWE-agent 来说，这意味着把 ctx.pinned_text + ctx.relevant_text + ctx.warnings 拼到 system prompt 或 user message 前缀。

## 与现有模块的依赖关系

```
                    OuterHarness
                   /     |     \
                  /      |      \
        before_step  after_step  after_episode
             |          |  \         |     \
             v          v   v        v      v
       ContextInjector  TrajAdapt Cambium  Energy Lignif
              \         \    /        /
               \         v  v        v
                ---->  TreeStore  <----
                          |
                       (SQLite+Kuzu+OpLog)
```

- ContextInjector 和 TrajectoryAdapter **从独立 Phase 4 模块降级为 OuterHarness 的内部实现**。它们的 spec 仍存在，但定位是"hook 实现细节"。
- Cambium / Decay Sentinel / Lignification / EnergySystem 不变——它们是无状态算法服务，被三个 hook 按需调用。
- TreeStore 是所有数据访问的唯一入口，OuterHarness 不绕过它直接访问 SQLite/KuzuDB。

## 与 TreeHarnessRunner 的关系

新框架下 `TreeHarnessRunner` 简化为：

```python
class TreeHarnessRunner:
    def __init__(self, config):
        # 装配 Tree 内部模块
        self.tree_store = TreeStore(...)
        ...
        # 装配 OuterHarness
        self.outer = OuterHarness(self.tree_store, ..., config.outer_config)
        # 装配 InnerHarness（SWE-agent / Aider / mock）
        self.inner = build_inner_harness(config.inner_config)
        # 组装
        self.wrapped = self.outer.wrap(self.inner)

    def run_sequential(self, tasks):
        results = []
        for task in tasks:
            ep_record, ep_report = self.wrapped.run_episode(task)
            results.append((ep_record, ep_report))
            self._log(ep_record, ep_report)
            if self._should_checkpoint():
                self._checkpoint()
        return results
```

Runner 不再持有 cambium/injector 等模块引用——它们都被 OuterHarness 封装。

## 实验对照能力

OuterHarness 的存在让以下四档 harness 对照成为可能（见 experiment.md）：

| 条件 | 描述 | before_step | after_step | after_episode |
|------|------|-------------|------------|---------------|
| `bare_inner` | 仅 inner harness，无 outer 包装，每 episode 冷启动 | - | - | - |
| `static_outer` | 静态外层 harness：只在 episode 开始时注入固定 system prompt，无演化 | 固定文本 | - | - |
| `freeform_outer` | 自由演化外层 harness（SIA-style）：由 LLM 在 episode 之间自由改写 scaffold | LLM 改写 prompt | LLM 改写 retry/tool 策略 | LLM 改写 scaffold |
| `tree_outer` | Tree Harness：结构化自演化，三 hook 内仅调用 5 个固定算符 | 三段（pinned + relevant + warnings） | crystallize/connect/quarantine | promote/decay |

四档对照锁定的关键比较：
- `bare_inner` vs `static_outer`：验证 outer 包装本身的边际价值
- `static_outer` vs `freeform_outer`：验证 self-evolution 的边际价值
- `freeform_outer` vs `tree_outer`：验证**结构化** self-evolution 相对自由演化的稳定性收益（Goodhart 风险、HV 方差）

这是论文最 sharp 的对照设计——证明"结构化算符空间 + trajectory-conditioned hook" 比 "自由 LLM 改写" 在 long-horizon 任务上更可靠。

## 四档 OuterHarness 实现草图

四档对照在装配上共用同一个 `OuterHarnessProtocol` 接口，差别仅在三 hook 的实现深度与是否持有 Tree 子模块引用。给定如下最小骨架，runner.md 可直接按 `outer_kind` 装配：

### NoOpOuterHarness（bare_inner）

```python
class NoOpOuterHarness:
    """完全透明：wrap(inner) 后等同于直接调 inner.run_episode。"""

    def wrap(self, inner): ...   # 返回直通 inner 的 _Wrapped
    def before_step(self, task, step_index, episode_id) -> ContextBlock:
        return ContextBlock(pinned_text="", relevant_text="",
                            warnings="", injected_cell_ids=[],
                            token_count=0, budget_used={})
    def after_step(self, record) -> StepReport:
        return StepReport([], [], [])
    def after_episode(self, record) -> EpisodeReport:
        return _empty_episode_report()
    def serialize(self): return {}
    def deserialize(self, _): pass
```

### StaticOuterHarness（static_outer）

```python
class StaticOuterHarness:
    """only before_step 注入固定 system prompt 文本（无演化）。"""

    def __init__(self, fixed_prompt: str, config: OuterHarnessConfig):
        self._fixed = fixed_prompt
        self._cfg = config

    def before_step(self, task, step_index, episode_id) -> ContextBlock:
        if step_index > 0:
            # 静态注入只在 episode 开头执行一次（后续 step 由 inner 自己延续 history）
            return _empty_context_block()
        return ContextBlock(
            pinned_text=self._wrap_pin(self._fixed),
            relevant_text="", warnings="",
            injected_cell_ids=[],
            token_count=estimate(self._fixed),
            budget_used={"pinned": estimate(self._fixed)},
        )

    def after_step(self, record) -> StepReport:
        return StepReport([], [], [])         # 不触发任何算符

    def after_episode(self, record) -> EpisodeReport:
        return _empty_episode_report()
```

### FreeformOuterHarness（freeform_outer，SIA-style baseline）

```python
class FreeformOuterHarness:
    """LLM 在 episode 之间自由改写 prompt（不约束算符空间）。"""

    def __init__(self, rewriter_model: str, rewrite_budget: int):
        self._rewriter = LLMClient(rewriter_model)
        self._budget = rewrite_budget
        self._current_prompt = ""            # 跨 episode 累积的 system prompt
        self._episode_traces: list[dict] = []

    def before_step(self, task, step_index, episode_id) -> ContextBlock:
        if step_index > 0:
            return _empty_context_block()
        return ContextBlock(
            pinned_text=self._wrap_pin(self._current_prompt),
            relevant_text="", warnings="",
            injected_cell_ids=[],
            token_count=estimate(self._current_prompt),
            budget_used={"pinned": estimate(self._current_prompt)},
        )

    def after_step(self, record) -> StepReport:
        # 不执行算符——只把轨迹追加进 episode trace 缓冲区
        self._episode_traces[-1]["steps"].append(record)
        return StepReport([], [], [])

    def after_episode(self, record) -> EpisodeReport:
        # 用 LLM 在自由空间内改写 self._current_prompt
        self._current_prompt = self._rewriter.rewrite(
            current=self._current_prompt,
            episode_record=record,
            budget=self._budget,
        )
        # EpisodeReport 中 op_counts 全 0；entropy_released 不可比
        rep = _empty_episode_report()
        rep.entropy_released = float("nan")   # 标记 N/A 给下游分析
        return rep
```

### TreeOuterHarness（tree_outer）

```python
class TreeOuterHarness:
    """结构化自演化，三 hook 内严格只调用 5 个固定算符（见上文流程）。"""

    def __init__(self, tree_store, context_injector, trajectory_adapter,
                 cambium, decay_sentinel, lignification, energy_system,
                 oplog, config: OuterHarnessConfig):
        # 持有全部 Tree 子模块引用——这是与前三档的本质区别
        ...

    def before_step(self, task, step_index, episode_id) -> ContextBlock:
        # 实现见上文 "Hook 1：before_step"
        ...

    def after_step(self, record) -> StepReport:
        # 实现见上文 "Hook 2：after_step"
        # 算符顺序遵守 I-Op3：crystallize → connect → reference → quarantine
        ...

    def after_episode(self, record) -> EpisodeReport:
        # 实现见上文 "Hook 3：after_episode"
        # 算符顺序遵守 I-Op3：reference → decay → promote
        ...
```

**实现层的关键约束**：前三档实现**不允许 import 任何 Tree 子模块**——这条让 runner 的静态依赖检查能机械地区分四档（grep `from .tree_store import` / `from .cambium import` 等）。仅 TreeOuterHarness 可以访问算符服务。

## 测试用例

1. wrap 一个 mock inner（每步固定 action） → run_episode 正常完成，产出 EpisodeRecord + EpisodeReport
2. before_step 在 L3/L4 为空时 → pinned_text 为空字符串但不报错
3. after_step 触发 quarantine → 下一次 before_step 的 warnings 段包含对应告警文本
4. after_step 的 warnings_for_next_step 超过 max_warnings_per_step → 截断到上限
5. after_episode outcome=pass → injected_cell_ids 中所有 cell 的 reference_count 增加
6. after_episode outcome=fail → 不自动 challenge，但 funnel verification 仍可独立 quarantine
7. static_outer 模式 → after_step / after_episode 不触发任何算符
8. tree_outer 模式 → after_episode 至少完成一次 promote + decay 算符调用
9. 同一个 task 跑 10 个 episode，entropy_released 累计值单调非降
10. pending_warnings 在 episode 结束后被清理（不跨 episode 泄漏）

## 不变量

- I1：任何对 Tree 内部模块的调用都从三个 hook 之一发出，不存在"绕过 hook 的直接修改"。
- I2：inner harness 不持有 Tree 引用，只通过 state.augment(context=...) 接收注入。
- I3：pending_warnings 是 episode-local 状态，不跨 episode 泄漏。
- I4：injected_cell_ids 在 after_episode 处理完后清空。
- I5：当 outcome=pass 时，所有 injected cell 至少触发一次 reference event；outcome=fail 时不自动触发任何 challenge event。

## 后续工作

- spec 落地后，`trajectory_adapter.md` 与 `context_injector.md` 需添加"Role in OuterHarness"节，明确它们是 hook 的实现细节。
- `experiment.md` 需采用本文件给出的四档 harness 对照（bare_inner / static_outer / freeform_outer / tree_outer）。
- `metrics.md` 需新增 Harness-Level Metrics 类别（entropy_release_per_episode 等）。
- `runner.md` 需重构为"OuterHarness + InnerHarness 装配器"。
