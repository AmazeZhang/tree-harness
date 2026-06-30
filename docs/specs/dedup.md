# Dedup Spec

## 概述

Dedup 是 Self-Evolution Operator Set 中 `crystallize` 算符内部的去重子例程——具体承担 Cambium Engine 三步管线中的 Step B，判定候选 cell 是否与 tree 中已有 active cell 重复。重复 → REINFORCE 已有 cell（触发 `decay` 算符的反向特例 reference，加 δ_reference）；不重复 → 允许 INSERT_NEW（让 Step C 继续 `connect`）。

定位说明：Dedup 不是独立 harness 状态所有者，只是一个无状态子例程。它由 `CambiumEngine.crystallize_step()` 在 Step A 抽出 CandidateCell 后调用一次，自身不创建 cell、不写 oplog、不动 ray 拓扑——它的输出只是一个 DedupResult 信号，由 CambiumEngine 决定执行 INSERT_NEW 还是触发 `energy_system.reference()`。这一切分让 `crystallize` 算符的"是否实际产生新 cell"完全可追溯：每个 INSERT_NEW 都对应一次 Dedup 判定为非重复，每个 REINFORCE 都对应一次 Dedup 命中已有 cell——OpLog 的 CRYSTALLIZE 与 REFERENCE 事件总能精确归因。

## 判定流程

```
candidate cell
    ↓
vec_search(candidate.embedding, top_k=5)
    ↓
取最高 similarity score
    ↓
score > 0.95 → REINFORCE（完全重复，直接强化已有 cell）
score ∈ (0.85, 0.95] → LLM 仲裁（语义级判定是否真正重复）
score <= 0.85 → INSERT_NEW（足够新颖）
```

## 接口定义

```python
from typing import Union, Literal
from dataclasses import dataclass

@dataclass
class DedupResult:
    action: Literal["INSERT_NEW", "REINFORCE"]
    matched_cell_id: Optional[str] = None       # REINFORCE 时的目标 cell
    similarity_score: Optional[float] = None    # 最高匹配分数
    reason: Optional[str] = None                # LLM 仲裁理由（仅灰区时有值）


class DedupProtocol(Protocol):
    def __init__(self, tree_store: TreeStore, llm_client: LLMClient, config: DedupConfig):
        ...

    def check(self, candidate: CandidateCell) -> DedupResult:
        """
        对候选 cell 执行去重判定。
        返回 DedupResult 指示 INSERT_NEW 或 REINFORCE。
        """
        ...

    def _vec_match(self, embedding: list[float]) -> list[tuple[Cell, float]]:
        """向量检索找最相似的已有 cell"""
        ...

    def _llm_arbitrate(self, candidate: CandidateCell, matched: Cell, score: float) -> Literal["same", "different"]:
        """LLM 仲裁灰区情况"""
        ...
```

## 配置

```python
@dataclass
class DedupConfig:
    threshold_exact: float = 0.95       # 以上 = 完全重复
    threshold_similar: float = 0.85     # 以上 = 灰区，需 LLM 仲裁
    search_top_k: int = 5               # vec_search 返回数量
    only_active: bool = True            # 只与 active cell 比较
```

## LLM 仲裁 Prompt

```
判断以下两条知识是否表达了相同的决策：

已有知识：
- Decision: {matched.decision}
- Rationale: {matched.rationale}
- Domain: {matched.context_domain}

新候选：
- Decision: {candidate.decision}
- Rationale: {candidate.rationale}
- Domain: {candidate.context_domain}

Similarity score: {score}

请判定：
- "same"：表达同一个决策，只是措辞不同或细节略有差异
- "different"：虽然表面相似，但涉及不同的场景/条件/解决方案

输出格式：{"verdict": "same"|"different", "reason": "..."}
```

## 行为契约

1. 只与 status="active" 的 cell 比较（quarantined/superseded 的 cell 不参与去重）
2. 当 REINFORCE 时，调用方（CambiumEngine）负责触发 energy_system.reference()
3. LLM 仲裁使用 temperature=0，结果可缓存
4. 如果 vec_search 返回空列表（tree 为空）→ 直接 INSERT_NEW
5. 灰区仲裁只看 top-1 匹配的 cell（不逐一比较所有灰区结果）

## CandidateCell 结构

```python
@dataclass
class CandidateCell:
    decision: str
    rationale: str
    context_trigger_task: str
    context_domain: str
    preconditions: list[dict]
    evidence: list[str]
    domain_tags: list[str]
    embedding: list[float]   # 预计算的向量
```

## 性能考量

- vec_search 是 O(N) 扫描（sqlite-vec），当 cell 总量 < 10K 时延迟 < 10ms
- LLM 仲裁是最贵的操作，通过两级阈值将 LLM 调用量限制在总候选的 ~10-20%
- LLM 仲裁结果缓存 key = hash(candidate.decision + matched.decision)

## 测试用例

1. tree 为空时提交候选 → INSERT_NEW
2. 插入 cell A，提交完全相同的候选 → score > 0.95 → REINFORCE(A)
3. 插入 cell A，提交措辞不同但语义相同的候选 → score ∈ (0.85, 0.95] → LLM 判 same → REINFORCE(A)
4. 插入 cell A，提交表面相似但情境不同的候选 → LLM 判 different → INSERT_NEW
5. 插入 cell A，提交完全不同的候选 → score < 0.85 → INSERT_NEW
6. quarantined cell 不参与去重比较
7. DedupResult 正确携带 matched_cell_id 和 similarity_score
