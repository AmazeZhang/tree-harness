# 开发规格与实现顺序

## 开发方法论

采用**规格驱动编程（Spec-Driven Development）**：
1. 每个模块先写 spec 文档（接口定义 + 行为契约 + 测试用例）
2. 按 spec 实现
3. 用 spec 中定义的测试用例验证

## 实现阶段

### Phase 0: 基础数据层（无外部依赖）

**目标**：TreeStore 能存取 cell 和 ray，op log 能记录和回放。

| 步骤 | 产出 | spec 文件 |
|------|------|-----------|
| 0.1 | Cell 数据模型 (dataclass) | `specs/cell_model.md` |
| 0.2 | SQLiteBackend (CRUD + vec search) | `specs/sqlite_backend.md` |
| 0.3 | KuzuBackend (node/edge CRUD + graph query) | `specs/kuzu_backend.md` |
| 0.4 | OpLog (append + replay) | `specs/oplog.md` |
| 0.5 | TreeStore (统一接口，协调双库) | `specs/tree_store.md` |

**验证标准**：能手动创建 cell、插入 ray、记录 op log、从 op log 重建状态。

### Phase 1: 能量系统（纯数学计算）

**目标**：energy 和 maturity 的更新逻辑正确运行。

| 步骤 | 产出 | spec 文件 |
|------|------|-----------|
| 1.1 | EnergySystem (update/reference/challenge/decay) | `specs/energy_system.md` |
| 1.2 | RingPromotion (maturity → ring 映射 + 滞回) | `specs/ring_promotion.md` |

**验证标准**：模拟 100 episode 的 energy 序列，验证 ring 升降逻辑、半衰期符合预期、无震荡。

### Phase 2: 形成层引擎（核心蒸馏管线）

**目标**：给定 trajectory，能产出 cell 并正确接线。

| 步骤 | 产出 | spec 文件 |
|------|------|-----------|
| 2.1 | CambiumEngine (crystallize 主流程) | `specs/cambium_engine.md` |
| 2.2 | Dedup (embedding 去重 + LLM 仲裁) | `specs/dedup.md` |
| 2.3 | Connector (ray 接线逻辑) | `specs/connector.md` |

**验证标准**：用 mock trajectory 灌入，验证 cell 产出数量合理、重复不新建、ray 方向正确。

### Phase 3: 腐朽检测（漏斗验证）

**目标**：低能量 cell 能被正确识别和裁决。

| 步骤 | 产出 | spec 文件 |
|------|------|-----------|
| 3.1 | DecaySentinel (漏斗主流程) | `specs/decay_sentinel.md` |
| 3.2 | Verifiers (Step 2a test runner, Step 2b grep/AST) | `specs/verifiers.md` |

**验证标准**：构造已知过时和未过时的 cell，验证漏斗分层正确、false-decay 率为 0。

### Phase 4: 木质化调度（合并/分裂）

**目标**：eligible cell 能被正确聚类合并或分裂。

| 步骤 | 产出 | spec 文件 |
|------|------|-----------|
| 4.1 | LignificationScheduler (promote/merge/split) | `specs/lignification.md` |

**验证标准**：模拟一组 maturity 跨阈值的 cell，验证 promote 原地改字段、merge 产出新 cell + SUPERSEDES。

### Phase 5: 集成层（对接真实 agent）

**目标**：完整跑通一个 episode（task → agent → trajectory → tree → next task）。

| 步骤 | 产出 | spec 文件 |
|------|------|-----------|
| 5.1 | TrajectoryAdapter (SWE-agent → StandardTrajectory) | `specs/trajectory_adapter.md` |
| 5.2 | ContextInjector (Tree → Agent context) | `specs/context_injector.md` |
| 5.3 | TreeHarnessRunner (主循环编排) | `specs/runner.md` |

**验证标准**：在 SWE-bench 的 1 个 task 上端到端跑通，确认 cell 入树、energy 更新、context 注入。

### Phase 6: 评测实验

**目标**：在 SWE-bench Verified 上跑完整序贯实验。

| 步骤 | 产出 | spec 文件 |
|------|------|-----------|
| 6.1 | ExperimentRunner (三组对比 + 日志) | `specs/experiment.md` |
| 6.2 | Metrics (resolve rate, tree health, token usage) | `specs/metrics.md` |
| 6.3 | Visualization (成长曲线, 树可视化) | — |

## 模块依赖关系

```
Phase 0 (TreeStore)
    ↓
Phase 1 (EnergySystem)  ←── Phase 2 (Cambium) 需要 energy.reference()
    ↓                           ↓
Phase 3 (Decay)          Phase 4 (Lignification)
    ↓                           ↓
    └───────────┬───────────────┘
                ↓
        Phase 5 (集成)
                ↓
        Phase 6 (实验)
```

## 每个 Phase 的 Definition of Done

- [ ] spec 文档写完（接口 + 行为 + 测试用例）
- [ ] 实现代码通过 spec 中所有测试用例
- [ ] 代码有 type hints，关键函数有 docstring
- [ ] 可独立运行（不依赖后续 Phase）

## 可替换性设计原则

每个模块通过 Protocol/ABC 定义接口，实现类可独立替换：
- `TreeStore` → 换存储引擎不影响上层
- `TrajectoryAdapter` → 换 agent 只需写新 adapter
- `ContextInjector` → 换注入策略不影响 tree 逻辑
- `EnergySystem` → 换能量公式不影响存储
- `CambiumEngine` → 换蒸馏 LLM 不影响其他模块
