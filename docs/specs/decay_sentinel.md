# Decay Sentinel Spec

## 概述

Decay Sentinel 是 Self-Evolution Operator Set 中 `quarantine` 算符的实现策略（同时承担 `decay` 算符在低能量候选筛选上的 Step 0 信号源）。它本身**不是 harness 状态的所有者**，只是一个无状态算法服务，由 `OuterHarness.after_step()` 调用：拿到候选 cell 列表后跑漏斗式验证，把"应被 quarantine"的裁决结果回传给 OuterHarness，由后者执行算符（写状态 + 触发 ray 一跳传播）。

定位说明：Sentinel 不直接修改 TreeStore，不写 oplog，不生成 warning 文本——这三件事都属于算符本身的副作用，由 OuterHarness 在收到裁决后完成。Sentinel 只做"判定"，不做"执行"。这一职责切分保证了 outer_harness.md 不变量 I-Op1。

## 漏斗结构

```
Step 0: 能量阈值 / 沉寂时间标记 → 候选列表
    ↓
Step 1: 被动信号检查（近期引用结果）
    ↓ (无法判定的)
Step 2: 确定性验证
    2a: test_id → 跑测试
    2b: precondition verify_hint → grep/AST/lockfile
    ↓ (仍无法判定的)
Step 3: LLM 深度裁决
```

## 接口定义

```python
class DecaySentinelProtocol(Protocol):
    def __init__(self, tree_store: TreeStore, energy_system: EnergySystem,
                 verifier_registry: VerifierRegistry, llm_client: LLMClient):
        ...
    
    def funnel_verify(self, candidate_ids: list[str]) -> dict[str, Verdict]:
        """对一批候选 cell 执行漏斗验证，返回 {cell_id: verdict}"""
        ...
    
    def _step1_passive_signals(self, cell: Cell) -> Optional[Verdict]:
        """检查近期引用结果"""
        ...
    
    def _step2a_test_verify(self, cell: Cell) -> Optional[Verdict]:
        """跑关联测试"""
        ...
    
    def _step2b_precondition_verify(self, cell: Cell) -> Optional[Verdict]:
        """机械核查 preconditions"""
        ...
    
    def _step3_llm_arbitrate(self, cell: Cell) -> Verdict:
        """LLM 最终裁决"""
        ...
```

注意：Sentinel **不实现 apply_verdict**。Verdict 的副作用（quarantine 算符、ray 切断、energy 更新）由 `OuterHarness.after_step` 在收到 verdict dict 后调用 `TreeStore.quarantine` / `EnergySystem.reference` 完成；这一切分对应 outer_harness.md I-Op1（写入唯一由算符触发）。Sentinel 自身可写入"信号性"副作用（如 reference / energy 微调）——但这些副作用应表达为对 `EnergySystem` 的调用，而非直接访问 TreeStore。

## Verdict 定义

```python
@dataclass
class Verdict:
    result: Literal["valid", "weak_valid", "decayed", "uncertain"]
    reason: str
    step_reached: int  # 在哪一步做出的裁定 (1/2/3)
    evidence: Optional[str] = None  # 支撑裁决的证据
```

## Verdict.result → 后续动作映射

| verdict.result | Sentinel 内部副作用（在 funnel_verify 内执行） | OuterHarness 在 after_step 接到 verdict 后执行 |
|---|---|---|
| `valid` | `EnergySystem.reference(cell_id)`（+δ_reference） | 不动 |
| `weak_valid` | 无 | 不动 |
| `decayed` | 无（不写 TreeStore） | `TreeStore.quarantine(cell_id, reason)` → 自动断出向 ray、写 oplog QUARANTINE/SEVER_RAY；生成 warning 文本入 pending_warnings |
| `uncertain` | `EnergySystem.decay_one(cell_id, Δ=-0.05)` + `TreeStore.mark_for_review(cell_id, True, reason="uncertain_verdict")` | 不动 |

切分原则：**减能/标 review** 走 EnergySystem（属 `decay` 算符的特例 challenge），由 Sentinel 在判定时直接触发；**真正改变 cell.status + 拓扑** 的 `quarantine` 算符由 OuterHarness 调用 TreeStore 执行——这条对齐 outer_harness.md I-Op1。

## verifiers → Verdict.result 映射

Step 2b 调用 `VerifierRegistry.verify` 返回 `"pass" | "fail" | "unknown"`。Step 2a 跑 test 时同样得到三态。映射规则：

| verifier 输出 | 来源 step | Verdict.result | step_reached |
|---|---|---|---|
| 全部 `pass` | 2a 测试 pass | `weak_valid` | 2 |
| 全部 `pass` | 2b 所有 hint 通过 | `valid` | 2 |
| 任一 `fail` | 2a 或 2b | `decayed` | 2 |
| 全 `unknown` 或混合 `pass/unknown` | 2a/2b 无法定调 | 进入 Step 3 LLM 仲裁 | 3 |

Step 3 LLM 输出限定四值之一；无法解析时降级为 `uncertain`。

## Step 1: 被动信号检查

查询该 cell 近 N 个 episode 的引用记录（从 op log）：
- 有引用且结果为负（challenge） → 大概率 decayed → 跳到 Step 3 确认
- 有引用且结果为正（reference） → valid，退出
- 无引用 → 无法判定，进 Step 2

## Step 2a: 测试验证

检查 cell.evidence 中是否有 `test_id:xxx`：
- 有 → 尝试跑该测试
- pass → weak_valid
- fail → decayed
- 无法跑（测试不存在了） → 进 Step 2b

## Step 2b: Precondition 核查

遍历 cell.context_preconditions：
- 有 verify_hint → 执行对应 verifier
- 全部通过 → valid
- 任一明确失败 → decayed
- 部分无法判定 → 进 Step 3

## Verifier Registry

```python
class VerifierRegistry:
    def verify(self, hint: VerifyHint, repo_path: str) -> Literal["pass", "fail", "unknown"]:
        """根据 hint 类型分发到具体 verifier"""
        ...

class FileGrepVerifier:
    def verify(self, path: str, pattern: str, repo_path: str) -> str:
        """ripgrep 验证文件中是否存在 pattern"""
        ...

class LockfileQueryVerifier:
    def verify(self, pkg: str, constraint: str, repo_path: str) -> str:
        """检查 requirements.txt / package.json 中的版本"""
        ...

class TestIdLookupVerifier:
    def verify(self, test_id: str, repo_path: str) -> str:
        """检查测试是否存在"""
        ...
```

## 测试用例

1. 一个 energy=-0.3 的 cell，有 test_id，测试 pass → verdict=weak_valid
2. 一个 energy=-0.3 的 cell，有 test_id，测试 fail → verdict=decayed
3. 一个 cell 有 precondition verify_hint=file_grep，文件中不存在 pattern → verdict=decayed
4. 一个 cell 无 test_id 无 verify_hint → 进 Step 3 LLM 裁决
5. verdict=decayed 后：cell.status=quarantined, energy=0, 出向 ray 全部 severed
6. verdict=valid 后：cell.energy 增加了 δ_reference
7. 漏斗统计正确：记录每层的处理数量和裁决分布
