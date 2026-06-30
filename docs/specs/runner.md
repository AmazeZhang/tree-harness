# TreeHarness Runner Spec

## 概述

TreeHarnessRunner 是序贯实验的最外层装配器。它的唯一职责是**组装** OuterHarness 与 InnerHarness、驱动 `wrapped.run_episode(task)` 循环、负责日志/checkpoint。所有自演化逻辑（crystallize / connect / promote / quarantine / decay）都在 OuterHarness 的三个 hook 内执行，Runner 不直接调用 cambium / injector / decay 等内部模块——这保证了 "outer hook = self-evolution operator 的唯一入口" 不变量（见 `outer_harness.md` I1）。

## 设计原则

- **薄装配**：Runner 不持有任何与"自演化"相关的状态，只持有 outer + inner + 日志器。
- **可替换 inner**：通过 `InnerHarness` 协议接入 SWE-agent / OpenHands / mini-swe-agent，用于 portability 实验（HV/MV ratio）。
- **可替换 outer**：通过 `OuterHarness` 协议接入 `tree_outer` / `static_outer` / `freeform_outer` / `bare_inner`（直通），用于四档对照。
- **状态隔离**：每个 `trial` 调用 `outer.reset()` 清空 Tree；同一 trial 内 episode 间共享 Tree 状态（这是序贯实验的核心变量）。

## 接口定义

```python
class TreeHarnessRunner:
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.outer = self._build_outer(config)        # tree_outer / static_outer / ...
        self.inner = self._build_inner(config)        # swe-agent / openhands / mini-swe-agent
        self.wrapped = self.outer.wrap(self.inner)    # 见 outer_harness.md
        self.logger = ExperimentLogger(config.log_dir)
        self.episode_count = 0

    def run_episode(self, task: Task) -> EpisodeRecord:
        """委托给 wrapped harness，自身只做日志与计数"""
        ...

    def run_sequential(self, tasks: list[Task]) -> list[EpisodeRecord]:
        """按时间顺序依次执行 tasks，保留 outer 内部状态"""
        ...

    def reset(self) -> None:
        """trial 起点：清空 outer 状态，inner 重新初始化"""
        ...

    def checkpoint(self, path: str) -> None: ...
    def resume(self, path: str) -> None: ...
```

## 配置

```python
@dataclass
class RunnerConfig:
    # 装配选择
    outer_kind: Literal["bare_inner", "static_outer", "freeform_outer", "tree_outer"]
    inner_kind: Literal["swe-agent", "openhands", "mini-swe-agent"]

    # tree_outer 专用
    db_path: Optional[str] = None
    energy_config: Optional[EnergyConfig] = None
    cambium_config: Optional[CambiumConfig] = None
    injector_config: Optional[InjectorConfig] = None

    # freeform_outer 专用（SIA-style baseline）
    freeform_rewriter_model: Optional[str] = None
    freeform_rewrite_budget: int = 1024

    # 通用
    repo_path: str
    agent_config: dict
    llm_client: LLMClient
    log_dir: str = "./logs"
```

## 装配逻辑

```python
def _build_outer(self, config: RunnerConfig) -> OuterHarness:
    if config.outer_kind == "bare_inner":
        return NoOpOuterHarness()
    if config.outer_kind == "static_outer":
        return StaticOuterHarness()
    if config.outer_kind == "freeform_outer":
        return FreeformOuterHarness(
            rewriter_model=config.freeform_rewriter_model,
            rewrite_budget=config.freeform_rewrite_budget,
        )
    if config.outer_kind == "tree_outer":
        tree_store = TreeStore(config.db_path)
        oplog = OpLog(tree_store)
        energy_system = EnergySystem(config.energy_config, tree_store, oplog)
        cambium = CambiumEngine(tree_store, energy_system, config.llm_client, config.cambium_config)
        decay_sentinel = DecaySentinel(tree_store, energy_system, config.llm_client)
        lignification = LignificationScheduler(tree_store, energy_system, config.llm_client)
        injector = ContextInjector(tree_store, config.injector_config)
        adapter = SWEAgentAdapter()  # TrajectoryAdapter Protocol 的实现
        return TreeOuterHarness(
            tree_store=tree_store, oplog=oplog, energy_system=energy_system,
            cambium=cambium, decay_sentinel=decay_sentinel,
            lignification=lignification, injector=injector, adapter=adapter,
        )
    raise ValueError(config.outer_kind)


def _build_inner(self, config: RunnerConfig) -> InnerHarness:
    if config.inner_kind == "swe-agent":
        return SWEAgentInner(config.agent_config)
    if config.inner_kind == "openhands":
        return OpenHandsInner(config.agent_config)
    if config.inner_kind == "mini-swe-agent":
        return MiniSWEAgentInner(config.agent_config)
    raise ValueError(config.inner_kind)
```

## 单 Episode 流程

```python
def run_episode(self, task: Task) -> EpisodeRecord:
    record = self.wrapped.run_episode(task)
    self.episode_count += 1
    self.logger.write(record)
    return record
```

Runner 在 episode 层级**不做任何业务判断**。所有自演化动作（before_step 注入 / after_step crystallize+connect+quarantine / after_episode promote+decay）均发生在 wrapped harness 内部，由 OuterHarness 的三 hook 驱动。这是把 "self-evolution operator set" 收敛到唯一入口的关键。

## 序贯实验流程

```python
def run_sequential(self, tasks: list[Task]) -> list[EpisodeRecord]:
    records = []
    for task in tasks:                              # tasks 已按时间排序
        record = self.run_episode(task)
        records.append(record)
        if self.episode_count % 10 == 0:
            self.checkpoint(self._auto_checkpoint_path())
    return records
```

注意：
- **不在 Runner 层做 maintenance_interval 调度**。Tree 内部的木质化 / decay tick 由 `after_episode` hook 自身决定何时触发，避免 Runner 与 Tree 内部节奏耦合。
- Runner 负责 checkpoint，但 checkpoint 内容由 OuterHarness 自报（`outer.serialize()`）。

## Checkpoint

```python
def checkpoint(self, path: str) -> None:
    payload = {
        "episode_count": self.episode_count,
        "outer_state": self.outer.serialize(),     # OuterHarness 自报
        "inner_state": self.inner.serialize(),     # InnerHarness 自报
        "timestamp": datetime.utcnow().isoformat(),
    }
    write_json(path, payload)

def resume(self, path: str) -> None:
    payload = read_json(path)
    self.episode_count = payload["episode_count"]
    self.outer.deserialize(payload["outer_state"])
    self.inner.deserialize(payload["inner_state"])
```

## 日志与可观测性

每个 episode 结束后，Runner 写入一行 JSONL：

```python
{
    "episode_index": ...,
    "task_id": ..., "outcome": "pass"|"fail"|"error",
    # Inner 报告
    "token_usage": ..., "duration_seconds": ...,
    # Outer 自演化报告（仅 tree_outer / freeform_outer / static_outer 有）
    "op_counts": {"crystallize": .., "connect": .., "promote": .., "quarantine": .., "decay": ..},
    "entropy_released": ...,
    "ring_distribution": {...},
    "context_retention_score": ...,
    "control_lag_steps": ...,
}
```

字段的精确定义见 `metrics.md` Harness-Level Metrics 一节。Runner 只搬运 OuterHarness/InnerHarness 在 EpisodeRecord 中提供的字段，不自行计算。

## 测试用例

1. `outer_kind="bare_inner"` + mock inner（5 episode 全 pass）→ `op_counts` 全为 0，`entropy_released` 全为 0。
2. `outer_kind="tree_outer"` + mock inner（5 episode 全 pass）→ `op_counts["crystallize"] > 0`，至少一个 cell 被 promote。
3. `outer_kind="freeform_outer"` + mock inner → 无 `op_counts`（freeform 不实现固定算符集），但 EpisodeRecord 包含 rewritten_prompt 字段。
4. trial 起点 `reset()` 后，tree_outer 的 cell 数为 0、`episode_count`=0。
5. checkpoint → resume 后，`episode_count` 与 Tree 内部 cell 数完全一致。
6. portability 实验：固定 `outer_kind="tree_outer"`，切换 `inner_kind` 三次，分别记录 HV/MV ratio（见 `metrics.md` H6）。
7. Runner 不直接 import cambium / injector / decay 模块（静态检查），确保装配薄层不变量。
8. EpisodeRecord 在四档 outer 下 schema 一致（缺失字段填默认值），便于 `experiment.md` 中统一画图。

## 不变量

- **R1**：Runner 不直接调用任何自演化算符——所有 crystallize / connect / promote / quarantine / decay 都经过 `wrapped.run_episode()`。
- **R2**：同一 trial 内不切换 outer/inner 类型；切换必须经过 `reset()`。
- **R3**：日志字段在四档 outer 下 schema 对齐，缺失字段填默认值而非省略键，使得下游分析脚本无需分支。

## 与其他 spec 的关系

- `outer_harness.md`：定义 OuterHarness 协议与三 hook 的语义；Runner 仅消费 `wrap` 返回的 wrapped harness。
- `trajectory_adapter.md`：是 TreeOuterHarness 在 `after_step` 内部使用的实现细节，Runner 不感知。
- `context_injector.md`：是 TreeOuterHarness 在 `before_step` 内部使用的实现细节，Runner 不感知。
- `experiment.md`：调用本 Runner，按 4 个 `outer_kind` 跑序贯实验；portability 子实验固定 `outer_kind="tree_outer"`，遍历 `inner_kind`。
- `metrics.md`：定义 Runner JSONL 输出中每一字段的精确含义。

### 数据类型对齐

- `outer_harness.md` 定义 `EpisodeRecord`（inner 报告的步序列）与 `EpisodeReport`（outer 报告的算符调用与熵释放）两类内部数据结构。
- `wrapped.run_episode()` 返回 `(EpisodeRecord, EpisodeReport)` 二元组。
- Runner 将该二元组**摊平为单行 JSONL**，下游 `experiment.md` / `metrics.md` 中称作 `TaskResult` 的字段集合即此摊平结果。三个名称指向同一组数据，仅命名空间不同。

#### TaskResult 字段映射

下表是 Runner 摊平时的字段对应关系。bare/static/freeform outer 输出的 EpisodeReport 中算符相关字段（op_counts / promoted / demoted / quarantined_count 等）填 0 或空列表（见 R3）。

| TaskResult 字段 | 来源 | 来源字段 | 备注 |
|---|---|---|---|
| `episode_index` | Runner | `self.episode_count` | trial 内单调递增 |
| `task_id` | EpisodeRecord | `task.task_id` | |
| `condition` | Runner | `config.outer_kind` | 四档对照列 |
| `trial` | Runner | `config.trial_id` | 多 trial 实验时填 |
| `inner_kind` | Runner | `config.inner_kind` | portability 实验用 |
| `outcome` | EpisodeRecord | `outcome` | pass/fail/error |
| `duration_seconds` | EpisodeRecord | `duration_seconds` | inner 自报 |
| `token_usage` | EpisodeRecord | `token_usage` | inner 自报 |
| `n_steps` | EpisodeRecord | `len(steps)` | |
| `op_counts` | EpisodeReport | `op_counts` | 5 算符 dict |
| `entropy_released` | EpisodeReport | `entropy_released` | 三类加权和 |
| `new_cells_count` | EpisodeReport | `new_cells_count` | |
| `compressed_count` | EpisodeReport | `compressed_count` | |
| `quarantined_count` | EpisodeReport | `quarantined_count` | |
| `decayed_count` | EpisodeReport | `decayed_count` | |
| `promoted` | EpisodeReport | `promoted` | list[(id,from,to)] |
| `demoted` | EpisodeReport | `demoted` | |
| `ring_distribution` | OuterHarness | `outer.snapshot_ring_distribution()` | episode 末调用 |
| `context_retention_score` | metrics.H2 | episode 后离线计算 | 选填 |
| `control_lag_steps` | metrics.H3 | episode 后离线计算 | 选填 |
| `inner_supports_pin_marker` | EpisodeRecord.metadata | `inner.capabilities().supports_pin_marker` | I-Pin 分层 |
| `inner_supports_warning_marker` | EpisodeRecord.metadata | `inner.capabilities().supports_warning_marker` | I-Pin 分层 |
