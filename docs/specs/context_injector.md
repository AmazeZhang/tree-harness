# Context Injector Spec

## 概述

ContextInjector 负责在 inner harness 发起 LLM call 之前，从 Tree 状态 H = (C, R, E, ρ) 中按 ring 分层抽取若干 cell 并格式化为可注入的文本。它是 OuterHarness `before_step` hook 的核心实现。

## Role in OuterHarness

ContextInjector 是 **before_step hook 的内部实现细节**，不再作为独立模块对外暴露。before_step 是只读 hook（不触发任何 Self-Evolution Operator），其全部副作用都局限在 Injector 内部的临时缓存上。

调用入口：

```
OuterHarness.before_step(task, step_index, episode_id)
  └─ pending = _pending_warnings.pop(episode_id, [])
  └─ neighbor = _neighbor_warning_queue.pop(episode_id, [])
  └─ pinned_cells = tree_store.list_by_ring(["L3","L4"], status="active")
  └─ pinned_text = injector.format_pinned(pinned_cells, budget=...)
  └─ relevant_ctx = injector.retrieve(task.description, repo,
                                      ring_filter=["L0","L1","L2"],
                                      token_budget=...)
  └─ warnings_text = injector.format_warnings(pending + neighbor, budget=...)
  └─ return ContextBlock(...)
```

Injector 不感知 Self-Evolution Operator 的存在；它只读取当前 H 快照。这条边界保证 `(episode_id, step_index)` 在重放 oplog 后能得到一致的 ContextBlock（确定性回放）。warnings 队列以 `episode_id` 为键，避免跨 episode 泄漏；同一 task 的不同 episode 不复用对方的 quarantine 告警。

## Budget Allocation Policy

Injector 的核心职责不只是"检索相关 cell"，更是 **prompt budget 的分配器**——这是 outer harness 区别于 memory 检索的关键能力。

三段静态预算：

| 段 | 默认占比 | 内容 | 截断策略 |
|----|---------|------|---------|
| pinned | 30% | L3 + L4 active cells，**无相似度过滤**，按 ring 优先级排序 | 超出预算时按 energy 降序保留 |
| relevant | 50% | L0 + L1 + L2，按 similarity × energy × ring_weight 综合打分 | 超出预算时按打分降序保留 |
| warnings | 20% | 来自上一步 quarantine 的告警，分两类：(a) `pending_warnings`（直接被 quarantine 的 cell），(b) `neighbor_warning_queue`（沿 incoming ray 一跳被传播的邻居 cell） | 超出预算时按 ray.weight × recency 降序保留，直接 quarantine 项优先于邻居项 |

设计要点：

1. **L3/L4 pinned 无条件注入**：项目公理永远在场，不与 query 相似度挂钩。这是 Harness Card 论文 "context drift" 属性的主要承担机制——任务关键信息不会因 query 偏移而被挤出 prompt。
2. **静态分配优于动态分配**：静态比例在长 horizon 下 token 占用可预测；动态算法（如根据 query 类型调整比例）在 1000+ episode 范围内难以收敛到稳定行为。
3. **三段独立排序**：不允许相关性极高的 L1 cell 挤掉 pinned 的 L4 cell。ring 层级是硬约束。
4. **空段安全**：任一段为空时返回空字符串，不报错；预算释放给其他段是**未来扩展点**（当前版本不做）。
5. **双源 warnings 合并**：直接告警与邻居告警共用 20% 预算；合并后按 `(is_direct, ray.weight, recency)` 排序裁剪。这条把 Ray 拓扑直接接到 control lag 指标上——邻居告警的注入延迟由 ray 一跳决定。
6. **Pin Marker 发射责任**：Injector 在序列化 pinned 段时必须把全部 pinned 内容用 `cfg.pin_open_tag` / `cfg.pin_close_tag` 包裹；warnings 段同理用 `cfg.warning_open_tag` / `cfg.warning_close_tag`。**marker 占用的 token 计入 pinned/warnings 段预算**，不另列。这条让 inner harness 的 history compaction 在跨多步累积上下文时能识别并保留 Tree 注入的关键段（见 outer_harness.md I-Pin）。Injector 不验证 inner 是否真支持 marker——能力判断在 OuterHarness `wrap()` 时通过 `inner.capabilities().supports_pin_marker` 读取一次，写入 EpisodeRecord 元数据供分析使用。

Budget 参数定义在 `OuterHarnessConfig` 中（pinned_ratio / relevant_ratio / warnings_ratio），Injector 接受调用方传入的具体 token 数，不自行决策比例。

## 接口定义

```python
@dataclass
class RetrievedContext:
    cells: list[str]           # 被选中的 cell id 列表（按打分降序）
    formatted_text: str        # 格式化后的注入文本（已含 marker？见下方说明）
    token_count: int           # 注入文本的估算 token 数
    retrieval_scores: dict     # {cell_id: score}


@dataclass
class WarningEntry:
    """before_step 收到的单条 warning。"""
    cell_id: str
    text: str                  # 已格式化的自然语言 warning
    is_direct: bool            # True=直接 quarantine 的 cell；False=邻居一跳传播
    ray_weight: float          # 用于排序的 ray 权重（is_direct=True 时为 1.0）
    recency: int               # 距今多少 step 内产生（小者优先）


class ContextInjectorProtocol(Protocol):
    def __init__(self, tree_store: "TreeStore", config: "InjectorConfig"):
        ...

    # --- before_step 直接调用的三个公开方法 ---
    def format_pinned(self, cells: list["Cell"], budget: int) -> str:
        """L3/L4 公理段。按 ring 降序 + energy 降序排序；
        超 budget 时按 energy 降序保留。
        输出文本必须以 cfg.pin_open_tag / cfg.pin_close_tag 包裹，
        marker token 计入 budget。"""
        ...

    def retrieve(
        self,
        task_description: str,
        repo: str,
        ring_filter: list[str],
        token_budget: int,
    ) -> RetrievedContext:
        """L0/L1/L2 相关段。
        - ring_filter: 只检索这些 ring 内的 active cell
        - token_budget: 该段的 token 上限
        返回 RetrievedContext，其中 formatted_text 不带 marker（inner 可压缩）。"""
        ...

    def format_warnings(self, warnings: list["WarningEntry"], budget: int) -> str:
        """Warnings 段。直接告警与邻居告警合并后按
        (is_direct desc, ray_weight desc, recency asc) 排序；
        超 budget 时降序裁剪。
        输出必须以 cfg.warning_open_tag / cfg.warning_close_tag 包裹，
        marker token 计入 budget。"""
        ...

    # --- 内部 ---
    def _score_cell(self, cell: "Cell", similarity: float) -> float:
        """综合打分：similarity * energy * ring_weight"""
        ...

    def _format_cells(self, cells: list["Cell"]) -> str:
        """格式化为 agent 可读的文本（不含 marker）"""
        ...
```

## 配置

```python
@dataclass
class InjectorConfig:
    max_tokens: int = 2000          # 注入文本的 token 上限
    max_cells: int = 10             # 最多注入的 cell 数量
    min_energy: float = 0.0         # 只注入 energy > 0 的 cell
    min_similarity: float = 0.3     # 最低相似度阈值
    ring_weights: dict = field(default_factory=lambda: {
        "L0": 0.5, "L1": 1.0, "L2": 1.5, "L3": 2.0, "L4": 2.5
    })
```

## 检索策略

```python
def retrieve(self, task_description: str, repo: str) -> RetrievedContext:
    # 1. Embedding 检索
    query_emb = embed(task_description)
    candidates = self.tree_store.vec_search(query_emb, top_k=20, threshold=self.config.min_similarity)
    
    # 2. 过滤
    alive = [(cell, sim) for cell, sim in candidates 
             if cell.status == "active" and cell.energy > self.config.min_energy]
    
    # 3. 综合打分
    scored = [(cell, self._score_cell(cell, sim)) for cell, sim in alive]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Token budget 截断
    selected = self._fit_budget(scored)
    
    # 5. 格式化
    text = self._format_cells([cell for cell, _ in selected])
    
    return RetrievedContext(
        cells=[cell for cell, _ in selected],
        formatted_text=text,
        token_count=estimate_tokens(text),
        retrieval_scores={cell.id: score for cell, score in selected}
    )
```

## 打分公式

```python
def _score_cell(self, cell: Cell, similarity: float) -> float:
    ring_weight = self.config.ring_weights[cell.ring]
    energy_factor = max(cell.energy, 0.1)  # 避免零乘
    return similarity * energy_factor * ring_weight
```

高 ring（内层）的 cell 权重更高——核心原则比边材细节更值得注入。

## 输出格式

```
[Project Experience]
Below are relevant lessons from previous work on this repository.

• [L3] When the project supports both PG and MySQL, always specify nulls_first=True in order_by calls.
  Why: PG and MySQL differ in NULL ordering defaults.
  Conditions: Project uses multiple DB backends.

• [L1] The CI pipeline is sensitive to import ordering in test files.
  Why: A custom linter checks import order before pytest runs.
  Conditions: File is under tests/ directory.

[End of Project Experience]
```

格式规则：
- 每个 cell 占 2-4 行
- ring 标签放在开头方便 agent 判断置信度
- Decision 用祈使句
- Rationale 用 "Why:" 引导
- Preconditions 用 "Conditions:" 引导（仅 assertion，不暴露 verify_hint）
- 全部用英文（agent 通常是英文 prompt）

## 注入位置

作为 SWE-agent system prompt 的追加段落。在原始 system prompt 末尾追加 `[Project Experience]...[End]` 块。

## 追踪哪些 cell 被注入

注入后记录 `context.cells` 列表。当 episode 结束时：
- outcome=pass → 被注入的 cell 全部触发 reference_event
- outcome=fail → 不自动 challenge（failure 不一定是 cell 的错）

## 测试用例

1. 空 tree → 返回空 formatted_text
2. tree 有 20 个 cell，task 相关的有 5 个 → 只返回相关的（similarity > threshold）
3. 2 个 cell 内容一样但一个 energy=0.8 一个 energy=0.1 → 高能量的排前面
4. 一个 L4 cell 和一个 L1 cell 相似度相同 → L4 排前面（ring_weight 高）
5. 总 token 超过 max_tokens → 截断到预算内
6. quarantined cell 不出现在结果中
7. 格式化文本包含 `[Project Experience]` 和 `[End of Project Experience]` 标记
