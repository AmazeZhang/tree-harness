# Experiment Runner Spec

## 概述

ExperimentRunner 在 long-horizon coding benchmark 上编排 **harness 级别的四档对照实验**，验证 Tree 作为 structurally self-evolving outer harness 相对于 bare / static / freeform 三种基线的优势。

实验设计的核心 framing：**比的不是 memory 系统的检索效果，比的是 harness 的可靠性属性**（stability / context drift / control lag，对齐 Harness Card arXiv:2605.23950 定义）。

## 实验配置

```python
@dataclass
class ExperimentConfig:
    # 四档 harness 对照（详见 outer_harness.md 中 Experiment Comparison 表）
    conditions: list[str] = field(default_factory=lambda: [
        "bare_inner",        # A: 仅 inner harness（SWE-agent 原版），每 episode 冷启动
        "static_outer",      # B: 静态外层包装：episode 开始注入固定 system prompt
        "freeform_outer",    # C: 自由演化外层（SIA-style）：LLM 在 episode 之间改写 scaffold
        "tree_outer",        # D: Tree 结构化自演化外层
    ])

    # Benchmark 参数
    # 长 horizon 是 Tree 优势的前提，优先使用 SWE-bench Pro；Verified 仅作为短任务对照
    dataset: str = "swe-bench-pro"
    fallback_dataset: str = "princeton-nlp/SWE-bench_Verified"
    repos: list[str] = None               # 选定的 repo 子集（None = 全部）
    max_tasks_per_repo: int = None

    # Inner harness（被包裹的下层）
    # 保持一致以满足 Harness Card 论文呼吁的"locked-harness protocol"
    inner_harness: str = "swe-agent"      # 可选: swe-agent / openhands / mini-swe-agent
    model: str = "claude-sonnet-4.5"      # 四组共用同一模型，控制 model 变量
    agent_timeout: int = 600

    # Freeform baseline 的 SIA-style 重写器（C 组使用）
    freeform_rewriter_model: str = "claude-sonnet-4.5"
    freeform_rewrite_budget: int = 5      # 每 N 个 episode 允许一次 scaffold 改写

    # Tree 外层（D 组使用）
    tree_outer_config: dict = None

    # 重复实验
    n_trials: int = 3
    random_seed_base: int = 42

    # Portability 实验（可选）：固定 outer，换 inner
    portability_inner_list: list[str] = None

    # 输出
    output_dir: str = "experiments/"
    log_format: str = "jsonl"
```

## 序贯评估协议

```
对每个选定 repo:
    tasks = 按时间（issue 创建 / commit 时间）排序的所有 task
    对每个 condition (A, B, C, D):
        对每个 trial (seed 1..n):
            fresh_start()  # 重置 harness 状态
            for task in tasks:  # 按时间顺序依次处理
                result = run_single_task(condition, task)
                log(result)
```

关键约束（对齐 Harness Card 的 locked-harness protocol）：

- 四组共用完全相同的 task 序列（控制 task 变量）
- 四组共用完全相同的 inner harness 与 model（控制 model 变量与 inner 变量）
- 每个 task 的处理顺序严格按时间（模拟真实项目演进，体现长 horizon）
- harness 状态（C 组的 scaffold / D 组的 Tree）在 task 间保持，这是 self-evolution 的实验变量
- 不同 trial 间相互独立（重新开始）

## 接口定义

```python
class ExperimentRunnerProtocol(Protocol):
    def __init__(self, config: ExperimentConfig):
        ...

    def run(self) -> ExperimentSummary:
        ...

    def run_repo(self, repo: str) -> RepoResult:
        ...

    def run_condition_trial(self, repo: str, condition: str, trial: int,
                            tasks: list[Task]) -> TrialResult:
        ...

    def run_single_task(self, condition: str, task: Task,
                        harness_state) -> TaskResult:
        ...

    def resume(self, checkpoint_path: str) -> ExperimentSummary:
        ...

    def run_portability(self, fixed_condition: str = "tree_outer",
                        inners: list[str] = None) -> PortabilityResult:
        """固定 outer harness，换 inner harness，验证 portability"""
        ...
```

## TaskResult 结构

```python
@dataclass
class TaskResult:
    task_id: str
    repo: str
    condition: str
    trial: int
    episode_index: int           # 该 task 在序列中的位置
    resolved: bool
    outcome: Literal["pass", "fail", "error", "timeout"]
    duration_seconds: float
    token_usage: int

    # Harness-Level Metrics（详见 metrics.md）
    ring_oscillation_count: int = 0      # 本 episode 内 promoted-then-demoted 次数
    context_retention_score: float = 0.0 # 关键 cell 在 N 步后仍在 prompt 中的概率
    control_lag_steps: float = 0.0       # quarantine → warning 注入的平均 step 距离
    entropy_released: float = 0.0        # 本 episode 释放的熵量

    # 算符调用计数（按 Self-Evolution Operator Set 分类）
    op_counts: dict = None               # {"crystallize":..,"promote":..,...}

    # 调试用快照
    harness_state_snapshot: Optional[dict] = None
```

## 输出目录结构

```
experiments/
├── config.json
├── {repo}/
│   ├── {condition}/                  # bare_inner / static_outer / freeform_outer / tree_outer
│   │   ├── trial_{n}/
│   │   │   ├── episodes.jsonl
│   │   │   ├── harness_snapshots/    # harness 状态快照（每 10 episode）
│   │   │   └── checkpoint.json
│   │   └── aggregate.json
│   └── comparison.json               # 四组对比
└── summary.json
```

## Checkpoint 机制

```json
{
    "repo": "django/django",
    "condition": "tree_outer",
    "trial": 1,
    "last_completed_index": 47,
    "harness_state_path": "harness_snapshots/episode_47.db",
    "timestamp": "2026-06-23T10:30:00Z"
}
```

resume 时从 last_completed_index + 1 继续，加载对应 harness 状态。

## Condition 实现差异

| 组 | before_step | after_step | after_episode | 状态载体 |
|----|-------------|------------|---------------|---------|
| A: bare_inner | inner 自带 prompt | 无 | 无 | 无 |
| B: static_outer | 固定 system prompt 注入 | 无 | 无 | 静态文本（不变） |
| C: freeform_outer | LLM 重写后的 prompt | LLM 决定 retry/tool 策略 | 每 N episode LLM 改写 scaffold | scaffold 代码 + 文本 |
| D: tree_outer | pinned + relevant + warnings 三段 | crystallize/connect/quarantine | promote/decay | TreeStore (SQLite+Kuzu) |

### Baseline B: Static Outer

最小的 outer 包装：在 episode 开始时注入一段固定的 "project description" system prompt（从 repo README 自动抽取），不做任何演化。目的是分离"任何 outer 包装"和"self-evolution"的边际价值。

### Baseline C: Freeform Outer (SIA-style)

复现 SIA (arXiv:2605.27276) 路线的简化版：

```python
class FreeformOuter:
    """每 N episode 让一个 Feedback-Agent 自由改写 scaffold（prompt / retry 策略）"""

    def after_episode(self, record):
        if self.episode_count % self.config.rewrite_budget == 0:
            recent_trajectories = self._collect_last_n(self.config.rewrite_budget)
            new_scaffold = self.feedback_agent.rewrite(
                current_scaffold=self.scaffold,
                trajectories=recent_trajectories,
            )
            self.scaffold = new_scaffold  # 直接覆盖
```

注意：本组不复现 SIA 的 weight update 路径——本实验只比较 harness-level 演化策略。

### Baseline D: Tree Outer

完整 Tree Harness，三 hook 全开，5 个算符均启用。配置详见 `outer_harness.md`。

## Benchmark 选择

**主 benchmark**：SWE-bench Pro。理由：

- Long-horizon 任务（多文件、跨 commit、长依赖链）才能体现 self-evolution 的累积价值
- 单 task 的 step 数足够多，控制 lag 等指标才有信号
- 主榜单 frontier-model 仍未饱和，对比空间大

**对照 benchmark**：SWE-bench Verified。理由：

- 短 horizon，预期 Tree 相对 bare_inner 优势更小，**验证我们 framing 的边界**（即 Tree 不是普适最优，而是 long-horizon 场景下的最优）
- 是 negative result 的来源——这是论文诚信度的关键

**可选扩展**：RepoBench-Long / Multi-SWE-bench / SWE-Bench-Live，验证 generalization。

## SWE-bench Repo 选择标准

选择子集的标准：

- 同 repo 内 task 数量 >= 20（足够体现 self-evolution 累积）
- task 时间跨度 >= 6 个月（展示长期效果与项目演化）
- 包含架构变化或 dependency 重大变更（展示 quarantine 算符价值）
- 代码语言为 Python（统一 inner harness 能力）

## 统计分析

每个指标报告：

- 均值 ± 标准差（across trials）
- Bootstrap 95% 置信区间
- 配对 t-test：(D vs A), (D vs B), (D vs C)，以及最关键的 **(D vs C)** 即结构化 vs 自由演化
- Effect size (Cohen's d)

**HV/MV 分解**：在 portability 实验中，按 Harness Card 协议计算 HV（同 inner、不同 outer）与 MV（同 outer、不同 inner）的方差比，验证 Tree 是否能在维持低 HV 的同时获得高 D-vs-A gap。

## 测试用例

1. 配置 1 个 repo + 5 个 task + 1 trial → 四组都能完整跑通
2. 四个 condition 使用相同 task 序列、相同 inner harness、相同 model
3. checkpoint 写入后 resume → 从正确位置继续
4. timeout task → outcome="timeout"，不影响后续 task
5. tree_outer condition → TaskResult.op_counts 中 crystallize > 0
6. bare_inner condition → TaskResult.op_counts 全为 0
7. static_outer condition → op_counts 全为 0 但 context 注入非空
8. freeform_outer condition → 每 rewrite_budget 个 episode 有一次 scaffold change 事件记录
9. episodes.jsonl 行数 = task 数量
10. summary.json 包含四组的 resolve_rate + harness-level metrics 对比
11. portability 实验：固定 tree_outer，3 种 inner 都能完成至少 1 个 task
