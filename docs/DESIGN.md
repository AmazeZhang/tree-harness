# Tree Harness 设计文档

## 概述

Tree Harness 是一个项目级认知中间件，为 coding agent 提供长期稳定的跨 episode 知识管理。它解决两个核心问题：agent 长期使用中的上下文污染（对话越长效果越差）和跨 session 失忆（新对话什么都不记得）。

## 核心比喻

将 agent 对一个项目的认知比作一棵树：
- 外层（形成层）不断长出新细胞
- 经过验证的知识逐渐木质化，从边材硬化为心材
- 过时的知识如同腐朽，被检测并隔离
- 整棵树始终保持健康的结构：外层活跃更替，内层稳固支撑

## 系统定位

```
[Task] → [Harness: 调度/工具/环境] → [Agent: 推理/决策/执行]
                    ↑
         [Tree Harness: 认知管理]
              ↕ 提供上下文 / 接收 trajectory
```

Harness 三层职责：
1. 执行环境（repo checkout, 工具注册, 沙盒隔离）— 不管
2. 流程编排（接任务, 调 agent, 收结果, 跑测试）— 不管
3. **认知管理**（跨 episode 知识积累/检索/维护）— **Tree Harness 的领地**

## 存储边界

**只存蒸馏后的 cell**，不存原始 trajectory/对话/tool call。

一个 episode 可能有 50 次工具调用，但只有从中蒸馏出的 0~N 条可复用认知会进入树。原始 trajectory 是临时输入，用完即弃。

## 设计哲学

### 七条公理

1. **生长即熵增，健康即有序熵增** — 问题不是阻止积累，而是让积累有结构
2. **树的内容归属项目，法则归属 harness** — harness 是可复用的"园丁服务"
3. **外层是瞬时表相，内层是持久支撑** — 维护的是"为什么是这样"的历史沉淀
4. **矢量方向必须来自外部** — 没有外部 fitness signal 的树 = 癌变
5. **死亡是设计的一部分** — 好死亡 = 木质化（结晶），坏死亡 = 腐朽（淤积）
6. **细胞不可修改，只能弃用和新生** — 完整演变史 + 当前查询清晰性
7. **Harness 是主动维护型系统** — 即使 agent 不在也在巡视/压缩

## 核心本体

### Cell（细胞）

系统的基本认知单位：**(Context, Decision, Rationale)** 三元组。

```json
{
  "id": "cell-2026-06-16-001",
  "ring": "L1",
  "maturity": 0.35,
  "energy": 0.8,
  "context": {
    "trigger_task": "django#1234",
    "domain": "ORM/sorting",
    "preconditions": [
      {"kind": "fact", "assertion": "项目同时支持 PG 和 MySQL", "verify_hint": null},
      {"kind": "code_invariant", "assertion": "存在跨数据库的排序测试", "verify_hint": {"type": "test_id_lookup", "test_id": "test_ordering_null"}}
    ]
  },
  "decision": "在所有涉及 ORM order_by 的代码中显式添加 nulls_first=True",
  "rationale": "PG 和 MySQL 对 NULL 排序的默认行为不同",
  "evidence": ["test_id:test_ordering_null", "commit:abc123"],
  "domain_tags": ["ORM", "sorting", "cross-db"],
  "status": "active",
  "source": "distilled",
  "created_at": "2026-06-16T10:00:00Z",
  "superseded_by": null
}
```

### Ring Layer（年轮层）

| 层级 | 名称 | 含义 | 半衰期 |
|------|------|------|--------|
| L0 | cambium（形成层） | 刚诞生，未验证 | ~2 episode |
| L1 | sapwood（边材） | 活跃知识 | ~6.6 episode |
| L2 | transition（过渡材） | 稳定模式 | ~22.8 episode |
| L3 | heartwood（心材） | 核心经验 | ~69 episode |
| L4 | core（树芯） | 基础认知 | ~346.5 episode |

### Ray（射线）

跨年轮的单向因果连接，方向为外→内（"我引用/依赖了你"）。

### Energy（能量）

驱动细胞生命周期的核心数值，由引用、挑战、自然衰减三种力共同作用。

## 五大模块

| 模块 | 职责 |
|------|------|
| Cambium Engine | 从 trajectory 蒸馏新 cell + 建立 ray 连接 |
| Lignification Scheduler | 监控 maturity → 触发升层/合并/分裂 |
| Tropism Calibrator | 接收外部信号 → 归因到 cell → 调整 energy |
| Decay Sentinel | 低能量 cell → 漏斗式验证 → 裁定存亡 |
| Radial Conduit Keeper | 维护 ray 拓扑完整性 |

## 技术栈

| 组件 | 选型 | 备胎 |
|------|------|------|
| 细胞存储 | SQLite + JSONB | Postgres |
| 向量检索 | sqlite-vec | FAISS / Chroma |
| 年轮拓扑 | KuzuDB（嵌入式图 DB） | Neo4j |
| 版本化 | append-only op log | — |
| LLM 调用 | temperature=0 + 全量缓存 | — |
| 抽象层 | TreeOps → TreeStore → {SQLiteBackend, KuzuBackend} | — |

## 实验设计概要

**对比组**：SWE-agent 原版 / SWE-agent + RAG memory / SWE-agent + Tree Harness

**评测协议**：SWE-bench Verified 按 repo 分组、按时间排序、序贯处理

**核心指标**：resolve rate vs episode number 的纵向成长曲线
