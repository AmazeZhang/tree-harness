# Ring Promotion Spec

## 概述

RingPromotion 是 Self-Evolution Operator Set 中 `promote` 算符内部的决策子模块——它接受 LignificationScheduler 传来的候选列表，按 `maturity → ring` 映射并叠加滞回（0.10 dead zone）/ 最小成熟期约束后，返回真正应被 promote 的 cell 列表。它**不直接修改 TreeStore**，决策结果交回 LignificationScheduler，再由 OuterHarness 调用 promote 算符落盘。

定位说明：滞回是 Harness Card **stability** 属性的物理实现；ring 层级是硬约束（无跳级）。Maturity 在新 framing 下被明确重定位为 **promote 算符的滞后窗口/cooldown 计数器**，不是独立健康轴——这条与 lignification.md 的 framing 保持一致。

## 阈值定义

阈值常量**唯一权威源**在 `cell_model.md`，本 spec 仅 import，不重复定义：

```python
from cell_model import RING_THRESHOLDS, RING_DEMOTE_THRESHOLDS

# RING_THRESHOLDS["L0→L1"] = 0.15   ...   "L3→L4" = 0.85
# RING_DEMOTE_THRESHOLDS["L1→L0"] = 0.05 ...   "L4→L3" = 0.75
# 升层-降层阈值差 = 0.10（滞回带；hysteresis_min_gap 不变量）
```

## Ring Capacity Policy

L3/L4 是 ContextInjector pinned 段（30% 预算）的注入源。长期运行下若 active L3 cell 持续累积，pinned 段按 energy 降序截断时会**等价于让低能量 L3 公理被隐性丢弃**——这是 context drift 在 pinned 段内部的回流，违反 Harness Card 属性边界。容量上限是该问题的硬约束。

容量检查时机：在 `evaluate_all()` 的 promote 阶段，每次 cell 拟升入 L3/L4 前先查目标 ring 当前 active 数量；若已达 `ring_capacity[target]`，按 `overflow_policy` 处理：

| overflow_policy | 行为 | 适用场景 |
|---|---|---|
| `force_promote`（默认） | L3 满时将候选直接升 L4（跳过 L3→L4 的 100-episode 等待）；L4 满时退化到 `demote_oldest` | 信任成熟度判定，让最稳定 cell 自然上浮 |
| `demote_oldest` | 把目标 ring 中 maturity 最低的 active cell 降一级，腾位给新入 | 严守容量、保留新陈代谢 |
| `block_new` | 阻塞升入，候选 cell 的 maturity 冻结在阈值下沿，等位 | 仅 debug 用，会破坏算符封闭性下的语义 |

**不变量 I-Cap**：
- 任意时刻 `|{c : c.ring=L3, c.status=active}| ≤ ring_capacity["L3"]`
- 任意时刻 `|{c : c.ring=L4, c.status=active}| ≤ ring_capacity["L4"]`
- 容量溢出导致的降级/跨级升必须经 promote 算符记录到 OpLog（带 `reason=overflow`），不允许绕过算符直接改 ρ。

**Metric 接入**：metrics.md H5 (Operator Call Distribution) 中应增设 `promote.reason ∈ {normal, overflow_force, overflow_demote}` 维度；overflow 比例若 > 10% 提示 capacity 配置过低或 crystallize 阈值过松，是实验调参信号。

## 接口定义

```python
@dataclass
class PromotionConfig:
    min_maturity_age: dict = field(default_factory=lambda: {
        "L0→L1": 3,     # 至少 3 episode
        "L1→L2": 10,    # 至少 10 episode
        "L2→L3": 30,    # 至少 30 episode
        "L3→L4": 100,   # 至少 100 episode
    })
    diversity_bonus: float = 0.1   # 跨 trigger 类型引用多样性奖励

    # --- L3/L4 容量上限（防 pinned 段长期膨胀挤占 30% 预算）---
    ring_capacity: dict = field(default_factory=lambda: {
        "L3": 60,   # active L3 cell 上限（典型 SWE 项目公理量级）
        "L4": 20,   # active L4 cell 上限（user_directive + 极少量自动晋升）
    })
    overflow_policy: Literal["force_promote", "demote_oldest", "block_new"] = "force_promote"
    # force_promote: L3 满时下一个 L3→L4 候选强制升 L4；L4 满时回写 demote_oldest
    # demote_oldest: 当前 ring 满时把该 ring 中 maturity 最低的 active cell 降一级
    # block_new: 当前 ring 满时阻止升入并把候选 cell 的 maturity 冻结（保守模式，仅 debug 用）


class RingPromotionProtocol(Protocol):
    def __init__(self, config: PromotionConfig, tree_store: TreeStore, oplog: OpLog):
        ...

    def evaluate_all(self, episode_id: str) -> PromotionReport:
        """
        扫描所有 active cell，执行升/降层判定。
        返回报告（promoted, demoted, blocked 列表）
        """
        ...

    def should_promote(self, cell: Cell, episode_count_since_creation: int) -> Optional[str]:
        """
        判断单个 cell 是否应该升层。
        返回目标 ring（如 "L2"）或 None。
        检查：
        1. maturity >= threshold
        2. episode_count >= min_maturity_age
        3. diversity check（可选）
        """
        ...

    def should_demote(self, cell: Cell) -> Optional[str]:
        """
        判断单个 cell 是否应该降层。
        返回目标 ring 或 None。
        检查：maturity < demote_threshold
        """
        ...

```

注意：RingPromotion **只产出 PromotionReport，不调用 tree_store.promote / demote**。真正的算符落盘由 `LignificationScheduler.check_promotions / check_demotions` 在 `run_maintenance_cycle` 内完成（见 lignification.md）；RingPromotion 是 `promote` 算符的**决策子模块**，Lignification 是**执行子模块**，二者切分对应 outer_harness.md I-Op1（算符调用唯一入口）。具体调用链：

```
OuterHarness.after_episode
  └─ Lignification.run_maintenance_cycle(episode_id)
       └─ Lignification.check_promotions()
            ├─ RingPromotion.evaluate_all(episode_id)     ← 决策
            │     └─ for each candidate: should_promote / should_demote
            └─ for (cell_id, _, target_ring) in report.promoted:
                  TreeStore.promote(cell_id, ..., reason=...)   ← 执行（唯一入口）
```

## PromotionReport 结构

```python
@dataclass
class PromotionReport:
    promoted: list[tuple[str, str, str]]   # [(cell_id, from_ring, to_ring)]
    demoted: list[tuple[str, str, str]]    # [(cell_id, from_ring, to_ring)]
    blocked: list[tuple[str, str, str]]    # [(cell_id, target_ring, reason)]
```

## 滞回机制

```
                demote_threshold    promote_threshold
L1→L0: 0.05                                           L0→L1: 0.15
L2→L1: 0.30                                           L1→L2: 0.40
L3→L2: 0.55                                           L2→L3: 0.65
L4→L3: 0.75                                           L3→L4: 0.85

         |----dead zone----|
         demote            promote
```

当 maturity 处于 [demote_threshold, promote_threshold) 区间时，不发生任何升/降层操作（dead zone）。这保证了 cell 不会因微小波动而频繁切换层级。

## 最小成熟期（防早熟）

即使 maturity 跨过了 promote_threshold，如果 cell 创建以来经过的 episode 数少于 `min_maturity_age`，则阻止升层并记录 blocked。

设计原因：新创建的 cell 可能因初始连续引用导致 maturity 快速爬升，但还未经过充分验证。

## 多样性判定（可选增强）

```python
def _diversity_score(self, cell: Cell) -> float:
    """
    统计引用该 cell 的不同 trigger_task 来源数量。
    多样性越高，说明该知识在多种情境下都有效。
    score = unique_trigger_count / total_reference_count
    """
    ...
```

当 diversity_score >= 0.5 时给予 maturity 额外 +diversity_bonus，加速成熟。该逻辑在 EnergySystem.update_maturity 中应用，此处只做判定和报告。

## 跳级禁止

cell 只能逐级升/降，不允许跳级（如 L1 直接跳 L3）。如果 maturity 急剧变化导致跨两级阈值，也只执行一次升级到相邻层，下一 episode 再判定是否继续升。

## 与其他模块的交互

| 调用方 | 调用时机 | 调用的方法 |
|--------|---------|-----------|
| EnergySystem.update_all_maturity | OuterHarness.after_episode 内 | 先更新所有 active cell 的 maturity |
| LignificationScheduler.check_promotions / check_demotions | maturity 更新后 | ring_promotion.evaluate_all(episode_id) → PromotionReport |
| LignificationScheduler 自身 | 拿到 PromotionReport 后 | tree_store.promote / demote（唯一执行点） |

## 测试用例

1. maturity=0.39 的 L1 cell → should_promote 返回 None
2. maturity=0.41 的 L1 cell，age=12 → should_promote 返回 "L2"
3. maturity=0.41 的 L1 cell，age=5 → should_promote 返回 None（blocked: min_age）
4. maturity=0.29 的 L2 cell → should_demote 返回 "L1"
5. maturity=0.31 的 L2 cell → should_demote 返回 None（在 dead zone 内）
6. maturity=0.90 的 L2 cell → should_promote 返回 "L3"（不是 "L4"，禁止跳级）
7. evaluate_all 在 10 个 cell 中正确识别 2 个 promote + 1 个 demote + 1 个 blocked
8. evaluate_all 自身**不**触发任何 TreeStore 写入——只产出 PromotionReport，oplog 中没有新的 PROMOTE/DEMOTE 记录（落盘由 LignificationScheduler 调 tree_store.promote/demote 完成，见 lignification.md）
9. 连续 5 个 episode maturity 在 [0.30, 0.40) 之间震荡 → 无升降层发生（滞回生效）
