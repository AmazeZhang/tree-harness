# Energy System Spec

## 概述

EnergySystem 是 Self-Evolution Operator Set 中 `decay` 算符的实现策略（同时承担其反向特例 `reference` / `challenge` 的能量增减）。它另外为 `promote` 算符提供门控信号——promote 决策需要读取 cell.energy 作为优先级标量。EnergySystem 本身**不是 harness 状态的所有者**，只是一个无状态算法服务，由 OuterHarness 在 after_step（reference / challenge）与 after_episode（全局 decay tick + maturity 更新）两个 hook 内分别调用。

定位说明：能量在新 framing 下不再叙述为"细胞生命周期"，而是 promote / quarantine 决策的优先级标量。它承担 Harness Card **stability** 属性中"按时间常数稳步衰减、抗短期噪声"的部分；生物隐喻保留为可读性 analogy，但不作为 spec 的主语义。

## 参数

```python
@dataclass
class EnergyConfig:
    # δ 增量
    delta_reference: float = 0.10    # 一次成功引用
    delta_challenge: float = -0.15   # 一次挑战/否定（不对称）
    
    # 每 episode 乘性衰减
    decay_rates: dict = field(default_factory=lambda: {
        "L0": 0.30,
        "L1": 0.10,
        "L2": 0.03,
        "L3": 0.01,
        "L4": 0.002,
    })
    
    # maturity 更新参数
    alpha: float = 0.05    # 引用驱动的成熟速率
    beta: float = 0.02     # 自然成熟惩罚
    e_norm: float = 1.0    # 归一化尺度
    
    # 腐朽候选阈值
    energy_threshold: float = -0.20
    idle_thresholds: dict = field(default_factory=lambda: {
        "L0": 2, "L1": 8, "L2": 30, "L3": 100, "L4": 500,
    })
```

## 接口定义

```python
class EnergySystemProtocol(Protocol):
    def __init__(self, config: EnergyConfig, tree_store: TreeStore, oplog: OpLog):
        ...
    
    def reference(self, cell_id: str, episode_id: str) -> None:
        """cell 被成功引用：energy += delta_reference"""
        ...
    
    def challenge(self, cell_id: str, episode_id: str) -> None:
        """cell 被挑战/否定：energy += delta_challenge"""
        ...

    def decay_one(self, cell_id: str, delta: float, episode_id: str,
                  reason: str = "uncertain_verdict") -> None:
        """对单个 cell 施加增量 Δ（可正可负）。
        典型场景：DecaySentinel 在 Verdict=uncertain 时调用 decay_one(cell_id, -0.05)。
        实现：tree_store.update_energy(cell_id, new=old+Δ, reason=reason, episode_id=...)
        """
        ...

    def decay_all(self, episode_id: str) -> None:
        """对所有 active cell 执行自然衰减：energy *= (1 - decay_rate[ring])"""
        ...

    def update_maturity(self, cell_id: str) -> None:
        """更新单个 cell 的 maturity：maturity += α * tanh(E/E_norm) - β * decay_rate"""
        ...

    def update_all_maturity(self, episode_id: str) -> None:
        """对所有 active cell 更新 maturity"""
        ...

    def get_decay_candidates(self, limit: Optional[int] = None,
                             order_by: str = "energy_asc") -> list[str]:
        """
        返回满足腐朽候选条件的 cell id（energy < energy_threshold 或
        idle_episodes ≥ idle_thresholds[ring]）。
        - limit: 最多返回多少个（用于 OuterHarness.funnel_sample_size 抽样）
        - order_by: "energy_asc"（最低能量优先）| "idle_desc"（最久沉寂优先）
        """
        ...
    
    def get_promotion_candidates(self) -> list[tuple[str, str]]:
        """返回所有 maturity 跨阈值的 cell: [(cell_id, target_ring)]"""
        ...
    
    def get_demotion_candidates(self) -> list[tuple[str, str]]:
        """返回所有 maturity 低于降级阈值的 cell: [(cell_id, target_ring)]"""
        ...
```

## 核心公式

### 能量更新（每 episode）

```
E += δ_reference * reference_count - |δ_challenge| * challenge_count
E *= (1 - decay_rate[ring])
```

### 成熟度更新（每 episode）

```
maturity += α * tanh(E / E_norm) - β * decay_rate[ring]
maturity = clip(maturity, 0.0, 1.0)
```

### 升层判定（带滞回）

```
promote if:  maturity >= ring_threshold[current_ring → next_ring]
demote if:   maturity < ring_demote_threshold[current_ring → prev_ring]
```

升降级阈值差为 0.10，防止震荡。

## 特殊规则

1. `source == "user_directive"` 的 cell：decay_rate = 0，不自然衰减
2. energy 可以为负（进入腐朽候选区）
3. 单个 episode 中同一 cell 可以被 reference 多次（每次都加 δ）
4. decay_all 在每个 episode 结束时调用一次

## 调用链与副作用

EnergySystem **不直接写 SQLite/KuzuDB，也不直接写 oplog**——所有能量/成熟度变更都通过 TreeStore facade 落盘，由 TreeStore 内部保证 oplog 与档案表的双写一致性：

```
EnergySystem.reference(cell_id, episode_id)
  ├─ read:  tree_store.get_cell(cell_id) → old_energy
  └─ write: tree_store.update_energy(cell_id, new=old+δ_ref,
                                     reason="reference", episode_id=...)
            ↑ TreeStore 内部：先 oplog.append("UPDATE_ENERGY", {...})
                            再 sqlite.update_cell(cell_id, energy=new)

EnergySystem.challenge(...)  → tree_store.update_energy(..., reason="challenge")
EnergySystem.decay_one(...)  → tree_store.update_energy(..., reason=...)

EnergySystem.decay_all(episode_id):
  for c in tree_store.list_active_cells():
      new = c.energy * (1 - decay_rates[c.ring])
      tree_store.update_energy(c.id, new, reason="natural_decay",
                               episode_id=episode_id)

EnergySystem.update_maturity / update_all_maturity:
  → tree_store.update_maturity(cell_id, new_maturity, episode_id)
```

这一约束对齐 tree_store.md "facade 不变量"——算法服务（EnergySystem / Cambium / Sentinel / Lignification）禁止以 `tree_store.sqlite.xxx` 或 `tree_store.kuzu.xxx` 形式绕过 facade。

## 测试用例

1. 新 cell (energy=0.5) 被 reference 一次 → energy = 0.5 + 0.10 = 0.60（衰减前）
2. L1 cell (energy=0.6) 经过一个 episode 无事件 → energy = 0.6 * 0.9 = 0.54
3. cell 被 challenge 3 次 → energy 下降 0.45
4. user_directive cell 经过 100 episode → energy 不变（decay_rate=0）
5. maturity 从 0.39 经一次正能量 episode → 可能跨 0.40 → promote 候选
6. maturity 从 0.41 经多次负能量 episode → 跌到 0.29 → demote 候选（0.30 以下）
7. maturity 在 0.38 时不触发 demote（滞回：demote 阈值是 0.30 而非 0.40）
8. 模拟 20 episode 纯衰减（无引用）→ L0 cell energy 趋近 0，L4 cell 几乎不变
9. energy < -0.20 的 cell 出现在 get_decay_candidates 中
