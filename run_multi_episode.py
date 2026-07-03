#!/usr/bin/env python
"""多 episode 实验: 验证 Tree 知识积累 + 跨 episode 注入。

3 个同域任务 (Python 工具函数 edge case 修复) 序贯运行:
  Ep1: sort_items — None 值导致 sorted() crash
  Ep2: merge_sorted — 空列表导致 IndexError
  Ep3: deduplicate — 不保留顺序

验证:
  1. new_cells > 0 (CambiumEngine 真正蒸馏出知识)
  2. ring_dist 非零 (cell 存入 Tree)
  3. 后续 episode 的注入包含前序知识
"""
import os
import sys
import tempfile
import textwrap
import json

from dotenv import load_dotenv
load_dotenv()

# 配置 litellm 连接 Qwen (给 mini-swe-agent 用)
os.environ["OPENAI_API_KEY"] = os.environ["QWEN_API_KEY"]
os.environ["OPENAI_API_BASE"] = os.environ["QWEN_BASE_URL"]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tree_harness.modules.runner import TreeHarnessRunner, RunnerConfig
from tree_harness.adapters.mini_swe_inner import MiniSWEConfig
from tree_harness.modules.outer_harness import Task
from tree_harness.core.llm_client import RealLLMClient
from tree_harness.modules.energy_system import EnergyConfig
from tree_harness.modules.lignification import LignificationConfig


def create_repo_bug1() -> tuple[str, str, str]:
    """Bug 1: sort_items 不处理 None。"""
    repo = tempfile.mkdtemp(prefix="tree_ep1_")
    with open(os.path.join(repo, "utils.py"), "w") as f:
        f.write(textwrap.dedent("""\
            def sort_items(items):
                return sorted(items)
        """))
    with open(os.path.join(repo, "test_utils.py"), "w") as f:
        f.write(textwrap.dedent("""\
            from utils import sort_items
            def test_sort():
                result = sort_items([3, None, 1, None, 2])
                assert result == [1, 2, 3, None, None], f"Got {result}"
            if __name__ == "__main__":
                test_sort()
                print("All tests passed!")
        """))
    os.system(f"cd {repo} && git init && git add -A && git commit -m 'init' -q 2>/dev/null")
    return repo, "utils.py", "test_utils.py"


def create_repo_bug2() -> tuple[str, str, str]:
    """Bug 2: merge_sorted 不处理 None 值 (同类型 bug, 不同函数)。"""
    repo = tempfile.mkdtemp(prefix="tree_ep2_")
    with open(os.path.join(repo, "utils.py"), "w") as f:
        f.write(textwrap.dedent("""\
            def merge_sorted(a, b):
                result = []
                i, j = 0, 0
                while i < len(a) and j < len(b):
                    if a[i] <= b[j]:
                        result.append(a[i])
                        i += 1
                    else:
                        result.append(b[j])
                        j += 1
                result.extend(a[i:])
                result.extend(b[j:])
                return result
        """))
    with open(os.path.join(repo, "test_utils.py"), "w") as f:
        f.write(textwrap.dedent("""\
            from utils import merge_sorted
            def test_merge():
                # Normal case works
                assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]
                # None values should be placed at the end
                assert merge_sorted([1, None], [2, 3]) == [1, 2, 3, None], f"Got {merge_sorted([1, None], [2, 3])}"
            if __name__ == "__main__":
                test_merge()
                print("All tests passed!")
        """))
    os.system(f"cd {repo} && git init && git add -A && git commit -m 'init' -q 2>/dev/null")
    return repo, "utils.py", "test_utils.py"


def create_repo_bug3() -> tuple[str, str, str]:
    """Bug 3: deduplicate 不保留顺序。"""
    repo = tempfile.mkdtemp(prefix="tree_ep3_")
    with open(os.path.join(repo, "utils.py"), "w") as f:
        f.write(textwrap.dedent("""\
            def deduplicate(items):
                return list(set(items))
        """))
    with open(os.path.join(repo, "test_utils.py"), "w") as f:
        f.write(textwrap.dedent("""\
            from utils import deduplicate
            def test_dedup():
                result = deduplicate([3, 1, 2, 3, 1])
                assert result == [3, 1, 2], f"Got {result}"
            if __name__ == "__main__":
                test_dedup()
                print("All tests passed!")
        """))
    os.system(f"cd {repo} && git init && git add -A && git commit -m 'init' -q 2>/dev/null")
    return repo, "utils.py", "test_utils.py"


def make_task(task_id: str, desc: str, repo: str) -> Task:
    return Task(task_id=task_id, description=desc, repo_path=repo)


def main():
    print("=" * 70)
    print("多 episode 实验: Tree 知识积累验证")
    print("=" * 70)

    # 创建 3 个 bug repo
    repo1, _, test1 = create_repo_bug1()
    repo2, _, test2 = create_repo_bug2()
    repo3, _, test3 = create_repo_bug3()
    print(f"\n[Setup] 3 bug repos created")

    # 验证 bug 确实存在
    for i, (repo, test) in enumerate([(repo1, test1), (repo2, test2), (repo3, test3)], 1):
        ret = os.system(f"cd {repo} && python {test} 2>&1 > /dev/null")
        print(f"  Bug {i}: {'FAIL (bug confirmed)' if ret != 0 else 'PASS (warning!)'}")

    # 配置 mini-swe-agent
    swe_config = {
        "model_name": "openai/Qwen3.7-Max-DogFooding",
        "step_limit": 15,
        "cost_limit": 2.0,
        "wall_time_limit_seconds": 300,
        "env_kind": "local",
        "env_timeout": 30,
    }

    # 配置 Runner — 关键: 使用 RealLLMClient + sentence-transformer
    llm_client = RealLLMClient.from_env("qwen")

    config = RunnerConfig(
        outer_kind="tree_outer",
        inner_kind="mini-swe-agent",
        mini_swe_config=swe_config,
        repo_path=".",  # 每个 task 有自己的 repo_path
        db_path="./logs/multi_episode/tree.db",  # 持久化到文件, 避免进程退出后丢失
        llm_client=llm_client,
        embedder_kind="sentence-transformer",
        embedder_model="BAAI/bge-base-zh-v1.5",
        log_dir="./logs/multi_episode",
        trial_id=0,
        # 自定义配置: 加速 ring promotion, 仅用于本实验 (不影响默认值)
        energy_config=EnergyConfig(alpha=0.15),  # 3x 默认, maturity 增长更快
        lignification_config=LignificationConfig(min_maturity_age={
            "L0→L1": 1,   # 允许 1 episode 后升层 (默认 3)
            "L1→L2": 10,
            "L2→L3": 30,
            "L3→L4": 100,
        }),
    )

    runner = TreeHarnessRunner(config)
    print(f"\n[Runner] outer={type(runner.outer).__name__}, inner={type(runner.inner).__name__}")
    print(f"  llm_client={type(runner.llm_client).__name__}")
    print(f"  embedder={type(runner.embedder).__name__} (dim={runner.embedder.dim})")

    # 定义 3 个任务
    tasks = [
        make_task(
            "ep1-sort-none",
            "Fix the bug in utils.py: sort_items() crashes when the list contains None values. "
            "The function should handle None by placing them at the end. "
            f"Run {test1} to verify your fix.",
            repo1,
        ),
        make_task(
            "ep2-merge-none",
            "Fix the bug in utils.py: merge_sorted() crashes when the list contains None values. "
            "The function should handle None by placing them at the end. "
            f"Run {test2} to verify your fix.",
            repo2,
        ),
        make_task(
            "ep3-dedup-order",
            "Fix the bug in utils.py: deduplicate() does not preserve the original order of elements. "
            "It should keep the first occurrence of each element and preserve insertion order. "
            f"Run {test3} to verify your fix.",
            repo3,
        ),
    ]

    # 序贯运行
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n{'─' * 70}")
        print(f"[Episode {i}] task_id={task.task_id}")
        print(f"  desc: {task.description[:80]}...")
        print(f"  repo: {task.repo_path}")
        print(f"{'─' * 70}")

        # 每次更新 mini-swe-agent 的 env_cwd
        runner.config.mini_swe_config = {**swe_config, "env_cwd": task.repo_path}
        # 重建 inner 以使用新 cwd
        runner.inner = runner._build_inner(runner.config)
        runner.wrapped = runner.outer.wrap(runner.inner)

        result = runner.run_episode(task)
        results.append(result)

        print(f"\n[Result {i}]")
        print(f"  resolved:      {result.resolved}")
        print(f"  outcome:       {result.outcome}")
        print(f"  n_steps:       {result.n_steps}")
        print(f"  new_cells:     {result.new_cells_count}")
        print(f"  quarantined:   {result.quarantined_count}")
        print(f"  promoted:      {result.promoted}")
        print(f"  demoted:       {result.demoted}")
        print(f"  ring_dist:     {result.ring_distribution}")
        print(f"  op_counts:     {result.op_counts}")
        print(f"  raw_ops:       {result.raw_op_counts}")
        print(f"  entropy:       {result.entropy_released:.2f}")

        # 验证修复
        test_file = [test1, test2, test3][i - 1]
        repo = [repo1, repo2, repo3][i - 1]
        ret = os.system(f"cd {repo} && python {test_file} 2>&1")
        print(f"  verify:        {'PASS' if ret == 0 else 'FAIL'}")

    # 总结
    print(f"\n{'=' * 70}")
    print("[Summary]")
    print(f"{'=' * 70}")
    for i, r in enumerate(results, 1):
        print(f"  Ep{i}: resolved={r.resolved}, steps={r.n_steps}, "
              f"new_cells={r.new_cells_count}, "
              f"ring_dist={r.ring_distribution}")

    # LLM 统计
    llm = runner.llm_client
    if hasattr(llm, "total_tokens"):
        print(f"\n[LLM Stats]")
        print(f"  calls:         {llm.call_count}")
        print(f"  cache_hits:    {llm.cache_hit_count}")
        print(f"  prompt_tokens: {llm.prompt_tokens}")
        print(f"  completion:    {llm.completion_tokens}")
        print(f"  total_tokens:  {llm.total_tokens}")

    # 检查 Tree 内容
    print(f"\n[Tree Content]")
    outer = runner.outer
    if hasattr(outer, "tree_store"):
        sqlite = outer.tree_store.sqlite
        all_cells = sqlite.list_by_status("active")
        print(f"  total active cells: {len(all_cells)}")
        for c in all_cells:
            print(f"    [{c.ring}] {c.decision[:80]}")
            print(f"          rationale: {c.rationale[:80]}")
            print(f"          domain: {c.context_domain}, tags: {c.domain_tags}")
            print(f"          energy: {c.energy:.2f}, maturity: {c.maturity:.2f}")
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()
