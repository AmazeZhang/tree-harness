#!/usr/bin/env python
"""端到端验证: MiniSWEAgentInner + TreeHarnessRunner + Qwen。

创建一个简单的 Python bug 修复 task, 验证:
1. mini-swe-agent inner harness 能正常 reset + step
2. Tree context 注入到 agent messages
3. agent 完成任务, episode 正常终止
"""
import os
import sys
import tempfile
import textwrap

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 配置 litellm 连接 Qwen
os.environ["OPENAI_API_KEY"] = os.environ["QWEN_API_KEY"]
os.environ["OPENAI_API_BASE"] = os.environ["QWEN_BASE_URL"]

# 确认 src 在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tree_harness.modules.runner import TreeHarnessRunner, RunnerConfig
from tree_harness.adapters.mini_swe_inner import MiniSWEConfig
from tree_harness.modules.outer_harness import Task


def create_bug_repo() -> str:
    """创建一个有 bug 的简单 Python 项目。"""
    repo = tempfile.mkdtemp(prefix="tree_e2e_repo_")

    # bug: sort 没处理 None 值, 会 crash
    with open(os.path.join(repo, "utils.py"), "w") as f:
        f.write(textwrap.dedent("""\
            def sort_items(items):
                \"\"\"Sort items, but crashes on None values.\"\"\"
                return sorted(items)

            def filter_active(items):
                \"\"\"Filter active items.\"\"\"
                return [x for x in items if x is not None]
        """))

    with open(os.path.join(repo, "test_utils.py"), "w") as f:
        f.write(textwrap.dedent("""\
            from utils import sort_items

            def test_sort_with_none():
                # This should not crash
                result = sort_items([3, None, 1, None, 2])
                assert result == [1, 2, 3, None, None], f"Got {result}"

            if __name__ == "__main__":
                test_sort_with_none()
                print("All tests passed!")
        """))

    # 初始化 git repo (mini-swe-agent 可能需要)
    os.system(f"cd {repo} && git init && git add -A && git commit -m 'init' -q 2>/dev/null")

    return repo


def main():
    print("=" * 60)
    print("MiniSWEAgentInner 端到端验证")
    print("=" * 60)

    # 1. 创建 bug repo
    repo_path = create_bug_repo()
    print(f"\n[1] Bug repo created: {repo_path}")

    # 验证 bug 确实存在
    ret = os.system(f"cd {repo_path} && python test_utils.py 2>&1")
    if ret == 0:
        print("  WARNING: test passed, bug may not exist")
    else:
        print("  OK: test fails (bug confirmed)")

    # 2. 配置 mini-swe-agent
    swe_config = MiniSWEConfig(
        model_name="openai/Qwen3.7-Max-DogFooding",
        step_limit=15,
        cost_limit=2.0,
        wall_time_limit_seconds=300,
        env_kind="local",
        env_cwd=repo_path,
        env_timeout=30,
    )
    print(f"\n[2] MiniSWE config: model={swe_config.model_name}, step_limit={swe_config.step_limit}")

    # 3. 配置 Runner
    config = RunnerConfig(
        outer_kind="tree_outer",
        inner_kind="mini-swe-agent",
        mini_swe_config={
            "model_name": swe_config.model_name,
            "step_limit": swe_config.step_limit,
            "cost_limit": swe_config.cost_limit,
            "wall_time_limit_seconds": swe_config.wall_time_limit_seconds,
            "env_kind": swe_config.env_kind,
            "env_cwd": swe_config.env_cwd,
            "env_timeout": swe_config.env_timeout,
        },
        repo_path=repo_path,
        embedder_kind="deterministic",
        log_dir="./logs/e2e_mini_swe",
        trial_id=0,
    )
    print(f"\n[3] RunnerConfig: outer=tree_outer, inner=mini-swe-agent")

    # 4. 构建 Runner
    runner = TreeHarnessRunner(config)
    print(f"\n[4] Runner built: outer={type(runner.outer).__name__}, inner={type(runner.inner).__name__}")

    # 5. 创建 Task
    task = Task(
        task_id="e2e-001",
        description=(
            "Fix the bug in utils.py: sort_items() crashes when the list contains None values. "
            "The function should handle None by placing them at the end. "
            "Run test_utils.py to verify your fix."
        ),
        repo_path=repo_path,
    )
    print(f"\n[5] Task: {task.description[:80]}...")

    # 6. 运行 episode
    print(f"\n[6] Running episode...")
    print("-" * 60)
    try:
        result = runner.run_episode(task)
        print("-" * 60)

        # 7. 打印结果
        print(f"\n[7] Result:")
        print(f"  resolved:     {result.resolved}")
        print(f"  outcome:      {result.outcome}")
        print(f"  n_steps:      {result.n_steps}")
        print(f"  duration:     {result.duration_seconds:.1f}s")
        print(f"  new_cells:    {result.new_cells_count}")
        print(f"  quarantined:  {result.quarantined_count}")
        print(f"  promoted:     {result.promoted}")
        print(f"  ring_dist:    {result.ring_distribution}")
        print(f"  inner_kind:   {result.inner_kind}")

        # 8. 验证 bug 是否修复
        print(f"\n[8] Verifying fix...")
        ret = os.system(f"cd {repo_path} && python test_utils.py 2>&1")
        if ret == 0:
            print("  PASS: test passed, bug fixed!")
        else:
            print("  FAIL: test still fails")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done")


if __name__ == "__main__":
    main()
