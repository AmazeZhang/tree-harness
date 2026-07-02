#!/usr/bin/env python3
"""5 Episode 快速验证 — 确认 L0→L1 promotion 触发 + episode_id 正确传递。"""
import time
import os
import tempfile

from tree_harness.core.llm_client import RealLLMClient
from tree_harness.core.embedding import SentenceTransformerEmbedder
from tree_harness.modules.runner import TreeHarnessRunner, RunnerConfig
from tree_harness.modules.outer_harness import Task

LOG_DIR = os.path.join(tempfile.gettempdir(), f"tree_verify_{int(time.time())}")
os.makedirs(LOG_DIR, exist_ok=True)

# 预热 BGE
print("Preloading BGE...", flush=True)
SentenceTransformerEmbedder("BAAI/bge-base-zh-v1.5")
print("  cached.\n", flush=True)

llm = RealLLMClient.from_env("qwen", log_path=os.path.join(LOG_DIR, "llm_calls.jsonl"))

runner = TreeHarnessRunner(RunnerConfig(
    outer_kind="tree_outer",
    inner_kind="mock",
    llm_client=llm,
    log_dir=LOG_DIR,
    repo_path=".",
    embedder_kind="sentence-transformer",
    embedder_model="BAAI/bge-base-zh-v1.5",
))

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
]

print(f"Log dir: {LOG_DIR}")
print(f"Running {len(tasks)} episodes with Qwen + BGE...\n")

for i, task in enumerate(tasks):
    t0 = time.time()
    r = runner.run_episode(task)
    elapsed = time.time() - t0

    cells = runner.outer.tree_store.sqlite.list_active()
    dist = {}
    for c in cells:
        dist[c.ring] = dist.get(c.ring, 0) + 1

    ops = r.op_counts
    raw_ops = runner.outer.oplog.count_by_raw_op()

    print(
        f"Ep{i+1:2d} [{elapsed:5.1f}s] cells={len(cells):2d} dist={dist} | "
        f"crystal={ops.get('CRYSTALLIZE',0)} connect={ops.get('CONNECT',0)} "
        f"promote(算符)={ops.get('PROMOTE',0)} quarantine={ops.get('QUARANTINE',0)} | "
        f"raw: PROMOTE={raw_ops['PROMOTE']} MERGE={raw_ops['MERGE']} "
        f"DEMOTE={raw_ops['DEMOTE']} | "
        f"tokens={llm.total_tokens}",
        flush=True,
    )

    # 检查 maturity 最高的 cell
    if cells:
        top = max(cells, key=lambda c: c.maturity)
        print(f"         top cell: ring={top.ring} maturity={top.maturity:.4f} energy={top.energy:.4f}",
              flush=True)

# 最终验证
print("\n" + "=" * 60)
print("  验证结果")
print("=" * 60)

cells = runner.outer.tree_store.sqlite.list_active()
raw = runner.outer.oplog.count_by_raw_op()

print(f"\n底层 op 统计:")
for op, cnt in sorted(raw.items()):
    if cnt > 0:
        print(f"  {op:25s}: {cnt}")

print(f"\n真实 ring promotion (PROMOTE op): {raw['PROMOTE']}")
print(f"Merge 操作 (MERGE op):            {raw['MERGE']}")
print(f"Demote 操作 (DEMOTE op):          {raw['DEMOTE']}")

promoted_cells = [c for c in cells if c.ring != "L0"]
print(f"\n非 L0 cell 数量: {len(promoted_cells)}")
for c in promoted_cells:
    print(f"  {c.id[:20]}... ring={c.ring} maturity={c.maturity:.4f}")

# 验证 episode_id 正确传递
all_promote_entries = runner.outer.oplog.get_entries(op_filter="PROMOTE")
has_none_ep = any(e.episode_id is None for e in all_promote_entries)
print(f"\nPROMOTE ops 总数: {len(all_promote_entries)}")
print(f"episode_id=None 的 PROMOTE ops: {sum(1 for e in all_promote_entries if e.episode_id is None)}")
print(f"episode_id 正确传递: {'YES' if not has_none_ep else 'NO (BUG!)'}")
