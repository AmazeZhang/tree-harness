#!/usr/bin/env python
"""SWE-bench 实验: Tree Harness 包装 mini-swe-agent。

每个 SWE-bench instance = 一个 episode。Tree Harness 跨 instance 积累知识
(cells, rings, context injection)，验证进化性。

用法:
  python run_swebench_tree.py --slice 0:5 --step-limit 30
  python run_swebench_tree.py --slice 0:10 --step-limit 250
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 环境变量: 加载 .env, 设置 Qwen 为 OpenAI 兼容
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("QWEN_API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.environ.get("QWEN_BASE_URL", "")
os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
# HF 缓存放到工作区内 (避免沙箱权限问题)
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".cache", "huggingface")

QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen3.7-Max-DogFooding")
MODEL_NAME = f"openai/{QWEN_MODEL}"

# ---------------------------------------------------------------------------
# 导入 (在环境变量设置之后)
# ---------------------------------------------------------------------------
from datasets import load_dataset
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    get_sb_environment,
)

from tree_harness.modules.runner import TreeHarnessRunner, RunnerConfig
from tree_harness.adapters.mini_swe_inner import MiniSWEConfig
from tree_harness.modules.outer_harness import Task
from tree_harness.modules.energy_system import EnergyConfig
from tree_harness.modules.lignification import LignificationConfig
from tree_harness.core.llm_client import RealLLMClient


def extract_patch(env, inner) -> str:
    """从 Docker 容器提取 git diff (优先) 或 agent submission (兜底)。"""
    # 方式1: 在 Docker 容器中跑 git diff (最可靠)
    try:
        out = env.execute({"command": "git diff"}, timeout=30)
        patch = out.get("output", "").strip()
        if patch and "diff --git" in patch:
            return patch
    except Exception:
        pass

    # 方式2: 从 agent 的 exit message 提取 submission
    try:
        agent = inner._agent
        if agent and agent.messages:
            last = agent.messages[-1]
            if last.get("role") == "exit":
                submission = last.get("extra", {}).get("submission", "")
                if submission.strip():
                    return submission.strip()
    except Exception:
        pass

    return ""


def run_swebench_tree(
    subset: str = "verified",
    split: str = "test",
    slice_spec: str = "0:5",
    step_limit: int = 30,
    output_dir: str = "./logs/swebench_tree",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))
    instances = sorted(instances, key=lambda x: x["instance_id"])

    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]

    print(f"Running on {len(instances)} instances with step_limit={step_limit}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {output_path}")
    print()

    # 构建 SWE-bench 配置 (用于 Docker 环境和 model)
    swe_config = get_config_from_spec("swebench")
    # 覆盖模型为 Qwen
    swe_config["model"]["model_name"] = MODEL_NAME
    swe_config["model"]["model_kwargs"] = {"drop_params": False, "parallel_tool_calls": True}
    # 增大 pull_timeout (x86 镜像在 arm64 上拉取较慢)
    swe_config["environment"]["pull_timeout"] = 600

    # 创建 LLM client (用于 CambiumEngine crystallize, DecaySentinel 等)
    llm_client = RealLLMClient(
        api_key=os.environ.get("QWEN_API_KEY", ""),
        base_url=os.environ.get("QWEN_BASE_URL", ""),
        model=QWEN_MODEL,
        log_path=str(output_path / "llm_calls.jsonl"),
    )

    # 创建 Tree Harness Runner (持久化 TreeStore, 跨 episode 积累)
    config = RunnerConfig(
        outer_kind="tree_outer",
        inner_kind="mini-swe-agent",
        mini_swe_config={
            "model_name": MODEL_NAME,
            "step_limit": step_limit,
            "cost_limit": 0,           # 禁用 (MSWEA_COST_TRACKING=ignore_errors)
            "wall_time_limit_seconds": 600,
        },
        db_path=str(output_path / "tree.db"),
        log_dir=str(output_path),
        trial_id=0,
        llm_client=llm_client,
        embedder_kind="sentence-transformer",
        embedder_model="BAAI/bge-base-zh-v1.5",
        # 加速 ring promotion (实验用, 不影响默认值)
        energy_config=EnergyConfig(alpha=0.15),
        lignification_config=LignificationConfig(
            min_maturity_age={"L0→L1": 1, "L1→L2": 10, "L2→L3": 30, "L3→L4": 100}
        ),
    )
    runner = TreeHarnessRunner(config)

    # preds.json (SWE-bench 评测格式)
    preds_path = output_path / "preds.json"
    preds: dict = {}

    # 统计
    resolved_count = 0
    start_time = time.time()

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        print(f"{'='*60}")
        print(f"Instance {i+1}/{len(instances)}: {instance_id}")
        print(f"  repo: {instance.get('repo', '?')}")
        print(f"{'='*60}")

        env = None
        try:
            # 1. 创建 Docker 环境 (每个 instance 独立容器)
            instance_config = copy.deepcopy(swe_config)
            env = get_sb_environment(instance_config, instance)
            print(f"  Docker container: {env.container_id[:12]}")

            # 2. 创建 model (带 SWE-bench 模板配置)
            model = get_model(config=copy.deepcopy(swe_config.get("model", {})))

            # 3. 注入到 MiniSWEAgentInner
            runner.inner.set_environment(env)
            runner.inner.set_model(model)

            # 4. 运行 episode
            task = Task(
                description=instance["problem_statement"],
                task_id=instance_id,
                repo_path=instance.get("repo", ""),
            )
            result = runner.run_episode(task)

            # 5. 提取 patch
            patch = extract_patch(env, runner.inner)

            # 6. 保存
            preds[instance_id] = {
                "model_name_or_path": MODEL_NAME,
                "instance_id": instance_id,
                "model_patch": patch,
            }
            preds_path.write_text(json.dumps(preds, indent=2))

            # 统计
            is_resolved = result.resolved and bool(patch)
            if is_resolved:
                resolved_count += 1

            ring_dist = runner.outer.snapshot_ring_distribution()
            print(f"  outcome: {result.outcome}")
            print(f"  resolved: {is_resolved}")
            print(f"  steps: {result.n_steps}")
            print(f"  patch: {len(patch)} chars")
            print(f"  new_cells: {result.new_cells_count}")
            print(f"  ring_dist: {ring_dist}")
            print(f"  promoted: {result.promoted}")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            preds[instance_id] = {
                "model_name_or_path": MODEL_NAME,
                "instance_id": instance_id,
                "model_patch": "",
            }
            preds_path.write_text(json.dumps(preds, indent=2))
        finally:
            # 清理 Docker 容器
            if env:
                try:
                    env.cleanup()
                except Exception:
                    pass

        elapsed = time.time() - start_time
        print(f"  elapsed: {elapsed:.0f}s")

    # 汇总
    total_elapsed = time.time() - start_time
    ring_dist = runner.outer.snapshot_ring_distribution()
    total_cells = sum(ring_dist.values())

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Instances:    {len(instances)}")
    print(f"Resolved:     {resolved_count}/{len(instances)}")
    print(f"Total cells:  {total_cells}")
    print(f"Ring dist:    {ring_dist}")
    print(f"Time:         {total_elapsed:.0f}s")
    print(f"Preds:        {preds_path}")

    # 保存汇总
    summary = {
        "subset": subset,
        "split": split,
        "slice": slice_spec,
        "step_limit": step_limit,
        "model": MODEL_NAME,
        "total_instances": len(instances),
        "resolved": resolved_count,
        "ring_distribution": ring_dist,
        "total_cells": total_cells,
        "elapsed_seconds": total_elapsed,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary:      {output_path / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tree Harness on SWE-bench")
    parser.add_argument("--subset", default="verified", help="SWE-bench subset")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--slice", default="0:5", help="Instance slice (e.g., '0:5')")
    parser.add_argument("--step-limit", type=int, default=30, help="Max steps per instance")
    parser.add_argument("--output", default="./logs/swebench_tree", help="Output directory")
    args = parser.parse_args()

    run_swebench_tree(
        subset=args.subset,
        split=args.split,
        slice_spec=args.slice,
        step_limit=args.step_limit,
        output_dir=args.output,
    )
