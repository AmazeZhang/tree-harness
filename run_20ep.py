#!/usr/bin/env python3
"""20 Episode 累积测试 — Qwen + BGE 真实 embedder。

输出:
  - 实时 episode 进度到 stdout
  - episodes_trial0.jsonl  (episode 级 TaskResult + LLM stats)
  - llm_calls.jsonl        (每次 LLM 调用明细)
  - final_report.txt       (最终分析报告)
"""
import time
import os
import json
import tempfile

from tree_harness.core.llm_client import RealLLMClient
from tree_harness.core.embedding import SentenceTransformerEmbedder
from tree_harness.modules.runner import TreeHarnessRunner, RunnerConfig
from tree_harness.modules.outer_harness import Task
from tree_harness.modules.metrics import (
    resolve_rate, cumulative_resolve_curve, ring_distribution,
    ray_connectivity_rate, active_dead_ratio, op_count_distribution,
    lignification_compression, centrality_gini, take_snapshot,
)

# ── 日志目录 ──
LOG_DIR = os.path.join(tempfile.gettempdir(), f"tree_20ep_{int(time.time())}")
os.makedirs(LOG_DIR, exist_ok=True)
LLM_LOG = os.path.join(LOG_DIR, "llm_calls.jsonl")
REPORT_PATH = os.path.join(LOG_DIR, "final_report.txt")

# ── 预热 BGE ──
print("Preloading BGE embedder...", flush=True)
SentenceTransformerEmbedder("BAAI/bge-base-zh-v1.5")
print("  cached.\n", flush=True)

# ── LLM client (带 token 日志) ──
llm = RealLLMClient.from_env("qwen", log_path=LLM_LOG)

# ── Runner ──
runner = TreeHarnessRunner(RunnerConfig(
    outer_kind="tree_outer",
    inner_kind="mock",
    llm_client=llm,
    log_dir=LOG_DIR,
    repo_path=".",
    embedder_kind="sentence-transformer",
    embedder_model="BAAI/bge-base-zh-v1.5",
))

# ── 20 个不同 task (模拟 SWE-bench 风格) ──
tasks = [
    Task("django-001", "Fix ORM order_by NULL sorting in Django queryset", "."),
    Task("django-002", "Fix migration conflict when adding nullable field to user model", "."),
    Task("django-003", "Fix admin form validation for nullable foreign key fields", "."),
    Task("django-004", "Fix serializer to handle empty queryset without raising exception", "."),
    Task("django-005", "Fix timezone-aware datetime comparison in query filters", "."),
    Task("django-006", "Fix model save when unique_together constraint involves null field", "."),
    Task("django-007", "Fix select_related causing duplicate rows in admin list view", "."),
    Task("django-008", "Fix template rendering for nested context variables in loops", "."),
    Task("django-009", "Fix middleware order causing auth token loss on redirect", "."),
    Task("django-010", "Fix CSRF token validation failing on AJAX POST requests", "."),
    Task("django-011", "Fix database connection pool exhaustion under concurrent load", "."),
    Task("django-012", "Fix formset validation when extra forms are empty", "."),
    Task("django-013", "Fix model clean method not called during bulk_update", "."),
    Task("django-014", "Fix reverse URL resolution for namespaced include patterns", "."),
    Task("django-015", "Fix prefetch_related producing N+1 queries with custom managers", "."),
    Task("django-016", "Fix content type caching causing stale template lookups", "."),
    Task("django-017", "Fix signal handler firing twice on test database transactions", "."),
    Task("django-018", "Fix annotation error when combining Count and Sum in aggregate", "."),
    Task("django-019", "Fix model str method causing unicode error in admin logs", "."),
    Task("django-020", "Fix queryset chaining losing distinct after values_list", "."),
]

# ── 运行 ──
print(f"Log dir: {LOG_DIR}", flush=True)
print(f"LLM call log: {LLM_LOG}", flush=True)
print(f"Running 20 episodes with Qwen + BGE...\n", flush=True)

snapshots = []
t_start = time.time()

for i, task in enumerate(tasks):
    t0 = time.time()
    r = runner.run_episode(task)
    elapsed = time.time() - t0
    total_elapsed = time.time() - t_start

    dist = ring_distribution(runner.outer.tree_store)
    total_cells = sum(dist.values())

    # 采集快照
    from datetime import datetime, timezone
    snap = take_snapshot(
        tree_store=runner.outer.tree_store,
        oplog=runner.outer.oplog,
        episode_index=i,
        timestamp=datetime.now(timezone.utc).isoformat(),
        resolved=r.resolved,
        cumulative_rate=resolve_rate(runner.results),
        token_usage=llm.total_tokens,
        duration_seconds=total_elapsed,
        entropy_released=r.entropy_released,
    )
    snapshots.append(snap)

    ops = r.op_counts
    conn = ops.get("CONNECT", 0)
    prom = ops.get("PROMOTE", 0)
    quar = ops.get("QUARANTINE", 0)
    crys = ops.get("CRYSTALLIZE", 0)
    dec = ops.get("DECAY", 0)

    print(
        f"Ep{i+1:2d} [{elapsed:5.1f}s] cells={total_cells:2d} "
        f"dist={dist} | "
        f"crystal={crys} connect={conn} promote={prom} "
        f"quarantine={quar} decay={dec} | "
        f"new={r.new_cells_count} entropy={r.entropy_released:.1f} | "
        f"llm={llm.call_count} tokens={llm.total_tokens}",
        flush=True,
    )

total_time = time.time() - t_start
print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

# ── 最终分析 ──
print("\n" + "=" * 70, flush=True)
print("  最终分析报告", flush=True)
print("=" * 70, flush=True)

lines = []
def p(s=""):
    print(s, flush=True)
    lines.append(s)

p()
p("--- 1. 树结构 ---")
cells = runner.outer.tree_store.sqlite.list_active()
dead_cells = runner.outer.tree_store.sqlite.list_cells(status="dead")
p(f"Active cells: {len(cells)}")
p(f"Dead cells:  {len(dead_cells)}")
p(f"Total ever:  {len(cells) + len(dead_cells)}")
p()

p("--- 2. Ring 分布 ---")
dist = ring_distribution(runner.outer.tree_store)
p(f"  {dist}")
p()

p("--- 3. Cell 详情 ---")
for idx, c in enumerate(sorted(cells, key=lambda x: x.created_at)):
    p(f"  [{idx+1}] Ring={c.ring} Energy={c.energy:.3f} Maturity={c.maturity:.3f}")
    p(f"      Decision: {c.decision[:100]}")
    p(f"      Tags: {c.domain_tags}")
    p()

p("--- 4. 算符统计 ---")
ops = op_count_distribution(runner.outer.oplog)
p(f"  Op distribution: {ops}")
p()

p("--- 5. 结构健康度 ---")
conn_rate = ray_connectivity_rate(runner.outer.tree_store)
ad_ratio = active_dead_ratio(runner.outer.tree_store)
comp = lignification_compression(runner.outer.oplog)
gini = centrality_gini(runner.outer.tree_store)
p(f"  Ray connectivity:    {conn_rate:.3f}  (目标 > 0.8)")
p(f"  Active/Dead ratio:   {ad_ratio:.1f}  (目标 > 2.0)")
p(f"  Lignification compr: {comp:.3f}")
p(f"  Centrality Gini:     {gini:.3f}  (0=均匀, 1=集中)")
p()

p("--- 6. 效果指标 ---")
rates = cumulative_resolve_curve(runner.results)
p(f"  Cumulative resolve curve: {[f'{r:.2f}' for r in rates]}")
p(f"  Final resolve rate: {resolve_rate(runner.results):.2f}")
p()

p("--- 7. LLM 消耗 ---")
p(f"  Total LLM calls:    {llm.call_count}")
p(f"  Cache hits:         {llm.cache_hit_count}")
p(f"  Actual API calls:   {llm.call_count - llm.cache_hit_count}")
p(f"  Prompt tokens:      {llm.prompt_tokens:,}")
p(f"  Completion tokens:  {llm.completion_tokens:,}")
p(f"  Total tokens:       {llm.total_tokens:,}")
if llm.call_count > 0:
    hit_rate = llm.cache_hit_count / llm.call_count * 100
    avg_tokens = llm.total_tokens / (llm.call_count - llm.cache_hit_count) if (llm.call_count - llm.cache_hit_count) > 0 else 0
    p(f"  Cache hit rate:     {hit_rate:.1f}%")
    p(f"  Avg tokens/call:    {avg_tokens:.0f}")
p()

p("--- 8. 日志文件 ---")
p(f"  Episode log: {os.path.join(LOG_DIR, 'episodes_trial0.jsonl')}")
p(f"  LLM call log: {LLM_LOG}")
p(f"  Report:      {REPORT_PATH}")
p()

# 写报告文件
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(lines))

print(f"\nReport saved to: {REPORT_PATH}", flush=True)
