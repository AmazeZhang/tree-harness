# Trajectory Adapter Spec

## 概述

TrajectoryAdapter 将特定 inner harness（如 SWE-agent / OpenHands / Aider）的原始输出转换为 Tree 内部模块（Cambium / Decay Sentinel）能消费的标准格式。它本身不参与任何决策，只做格式翻译与轨迹清洗。

## Role in OuterHarness

TrajectoryAdapter 是 **after_step hook 的内部实现细节**，不再作为独立模块对外暴露。它的调用入口收敛到 `OuterHarness.after_step(record: StepRecord)` 内：

```
OuterHarness.after_step(record)
  └─ trajectory_adapter.convert_step(record) → StandardStep
       └─ cambium.crystallize_step(step) → list[Cell]   (operator: crystallize)
       └─ cambium.connect_new_cells(...)                (operator: connect)
       └─ decay_sentinel.funnel_verify(candidates)
            └─ if quarantine → trajectory_adapter.format_quarantine_warning(...)
                              → OuterHarness 写入 _pending_warnings
```

Adapter 自身**不调用任何 Self-Evolution Operator**——算符调用由 OuterHarness 协调。Adapter 在数据流向上只承担两个职责：(a) 把 inner harness 异构输出翻译成 StandardTrajectory / StandardStep；(b) 把 quarantine 算符的执行结果翻译成自然语言 warning 文本。

## Verification Feedback Loop

Adapter 与 Decay Sentinel 共同实现 Harness Card 论文定义的 **control lag = 1 step** 性质：

1. 单步轨迹经 Adapter 翻译为 StandardStep；
2. OuterHarness 在 after_step 内调用 Decay Sentinel 对抽样 cell 做 funnel verification；
3. 若 verification 触发 `quarantine` 算符，Adapter 提供 `format_quarantine_warning(cell, evidence) → str` 把算符语义翻译为自然语言告警；
4. OuterHarness 把告警写入 `_pending_warnings[episode_id]`；
5. **同步沿被 quarantine cell 的 incoming ray 一跳传播**：OuterHarness 从 TreeStore 取邻居 cell 列表，调用 Adapter 的 `format_neighbor_warning(neighbor_cell, quarantined_cell, ray) → str`，写入 `neighbor_warning_queue[episode_id]`；
6. 下一次 before_step 读取两个队列（直接告警 + 邻居告警），合并注入 ContextBlock.warnings 段（双源预算合并见 `context_injector.md`）；
7. inner harness 的下一次 LLM call 即看到告警，闭环延迟 = 1 step；邻居告警的拓扑延迟额外 = 1 hop（仅一跳，避免雪崩）。

告警文本规范：

**直接告警（format_quarantine_warning）**
- 必须明确引用被 quarantine 的 cell 的 Decision 字段原文
- 必须给出 verification 失败原因（verifier 名 + evidence 摘要）
- 必须以祈使句结尾，明确告知 inner agent "disregard / re-evaluate"

**邻居告警（format_neighbor_warning）**
- 必须引用被 quarantine 的源 cell（不引用邻居自己的 Decision，避免错杀）
- 必须说明邻居 cell 与源 cell 的 ray 关系（依赖 / 派生 / 共同前提）
- 必须以"verify before relying"语气结尾，提示而非禁用——邻居 cell 未被直接否证，仅是连带预警

示例（直接告警）：
```
WARNING: A previously injected guideline ("always pass nulls_first=True in order_by")
has been quarantined by verifier `lockfile_query` — current pyproject.toml shows the
project no longer supports MySQL, making the guideline obsolete. Disregard it for the
remainder of this episode.
```

示例（邻居告警）：
```
ADJACENT WARNING: The cell "use Django's Q objects for ordering null fields" is connected
(ray weight=0.72, source=derived-from) to a guideline that was just quarantined. The
adjacent cell itself has not been refuted, but verify its preconditions against the
current repo state before relying on it.
```

## 标准输出格式

```python
@dataclass
class StandardTrajectory:
    task_id: str                    # "django__django-16379"
    task_description: str           # issue 原文摘要
    repo: str                       # "django/django"
    base_commit: str                # task 对应的 repo commit
    outcome: Literal["pass", "fail", "error"]
    patches: list[str]             # 最终提交的 diff（可能多个文件）
    test_results: dict             # {"test_ordering_null": "pass", ...}
    key_actions: list[str]         # 筛选后的关键动作描述（≤10条）
    duration_seconds: float         # 执行时长
    token_usage: int                # LLM token 消耗
```

## 接口定义

```python
class TrajectoryAdapterProtocol(Protocol):
    # --- after_step 调用 ---
    def convert_step(self, record: "StepRecord") -> "StandardStep":
        """单步翻译：StepRecord → StandardStep（见 cambium_engine.md）。
        负责字段映射 + key_action 单句化；不裁剪决策语义。"""
        ...

    # --- after_episode 调用（offline 批量复测时使用） ---
    def convert(self, raw_output: dict) -> "StandardTrajectory":
        """将 inner harness 的完整 episode 输出转为 StandardTrajectory。
        OuterHarness 在线流程不调用本方法；Cambium 的 episode-level crystallize
        (offline batch) 才使用该入口。"""
        ...

    def filter_key_actions(self, raw_actions: list[dict]) -> list[str]:
        """从原始动作列表中筛选关键动作（≤10 条），convert 内部调用"""
        ...

    # --- after_step 内 OuterHarness 发出 quarantine 后调用 ---
    def format_quarantine_warning(
        self, cell: "Cell", evidence: str, verifier_name: str,
    ) -> str:
        """把 quarantine 算符的执行结果翻译为自然语言告警。
        约束见上文 "告警文本规范 - 直接告警"。"""
        ...

    def format_neighbor_warning(
        self, neighbor: "Cell", quarantined: "Cell", ray: dict,
    ) -> str:
        """沿 incoming ray 一跳传播时使用，生成邻居预警文本。
        约束见上文 "告警文本规范 - 邻居告警"。
        ray dict 字段：{from_id, to_id, weight, source_type, last_activated}"""
        ...
```

Adapter 不持有对 TreeStore / Cambium / DecaySentinel 的引用——它是纯翻译层。所有依赖 cell 字段的格式化输入由 OuterHarness 在调用时显式传入。

## SWE-Agent Adapter 实现要点

SWE-agent 的 trajectory 输出格式（`trajectory.json`）：

```json
{
    "instance_id": "django__django-16379",
    "model_name_or_path": "...",
    "model_patch": "diff --git a/...",
    "history": [
        {"role": "system", "content": "..."},
        {"role": "assistant", "content": "...", "action": "find_file ..."},
        {"role": "user", "content": "...(observation)..."}
    ],
    "exit_status": "submitted",
    "test_result": {"PASS_TO_PASS": [...], "FAIL_TO_PASS": [...]}
}
```

转换映射：
| StandardTrajectory 字段 | SWE-agent 来源 |
|------------------------|----------------|
| task_id | instance_id |
| task_description | 从 SWE-bench dataset 查 |
| repo | instance_id 解析（`__` 分隔） |
| base_commit | 从 SWE-bench dataset 查 |
| outcome | exit_status + test_result 综合判定 |
| patches | model_patch |
| test_results | test_result 展开 |
| key_actions | history 过滤 |
| duration_seconds | 从执行日志提取 |
| token_usage | 从 API 日志提取 |

## Key Actions 过滤规则

从 history 中过滤掉机械操作，保留决策性动作：

**保留**：
- `edit`（编辑文件）
- `create`（创建文件）
- `find_file` / `search_dir`（有目的的搜索）
- `python -c "..."`（验证性执行）
- 任何导致 agent 改变方向的观察

**过滤掉**：
- `cd`、`ls`、`pwd`（导航）
- `cat`、`head`、`tail`（纯查看）
- `git status`、`git diff`（纯状态查看）
- 重复的相同命令

**格式化**：每条动作压缩为一句话描述，如：
- `"编辑 django/db/models/sql/compiler.py 第 45 行，添加 nulls_first 参数"`
- `"搜索项目中所有 order_by 调用，发现 12 处"`
- `"运行测试 test_ordering_null，结果 PASS"`

最多保留 10 条。

## 测试用例

1. 正常 SWE-agent 输出 → 正确转换所有字段
2. exit_status="submitted" + FAIL_TO_PASS 全 pass → outcome="pass"
3. exit_status="submitted" + FAIL_TO_PASS 有 fail → outcome="fail"
4. exit_status="error" → outcome="error"
5. history 有 30 条 → key_actions 最多 10 条
6. history 全是 ls/cat → key_actions 为空列表
7. patches 为空（agent 没提交）→ 正常处理，outcome 由 test 决定
