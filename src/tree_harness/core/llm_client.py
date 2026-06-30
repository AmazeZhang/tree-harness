"""LLM 调用抽象层 —— temperature=0, 全量缓存 (DESIGN.md 约束)。

LLMClient 是 CambiumEngine 和 DecaySentinel 的共享依赖:
- Cambium Step A: crystallize prompt → 返回决策知识 JSON
- Cambium Step B: dedup arbitration → 返回 keep_new / keep_old / merge
- DecaySentinel (Phase 3): verifier prompt → 返回验证结果

所有实现必须满足两个约束:
1. temperature = 0 — 确定性输出 (相同输入 → 相同输出)
2. 全量缓存 — 重复调用不产生额外开销
"""
from __future__ import annotations

import hashlib
import json
from typing import Protocol


class LLMClient(Protocol):
    """LLM 调用接口 — 所有 LLM 交互的唯一入口。"""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """单轮 completion 调用,返回 LLM 文本响应。

        约束 (DESIGN.md):
        - temperature = 0 (确定性输出)
        - 全量缓存 (相同输入 → 相同输出,重复调用不产生额外开销)
        """
        ...


class DeterministicLLMClient:
    """确定性 LLM 桩 — 用于测试和开发阶段。

    特性:
    - 全量缓存: 相同 (system_prompt, user_prompt) → 相同响应
    - 预注入: 可按 system_prompt 子串匹配注入特定响应
    - 默认响应: 未匹配的调用返回 default_response

    用法示例:
        client = DeterministicLLMClient()
        client.inject("crystallize", json.dumps({
            "decision": "Always validate input",
            "rationale": "Prevents crash",
            "preconditions": [],
            "evidence": [],
            "domain_tags": ["validation"],
        }))
        client.inject("dedup", json.dumps({"action": "INSERT_NEW"}))
    """

    def __init__(self, default_response: str = "{}"):
        self._default = default_response
        self._cache: dict[str, str] = {}
        self._injected: dict[str, str] = {}
        self._call_count = 0
        self._cache_hit_count = 0

    def inject(self, system_prompt_substring: str, response: str) -> None:
        """预注入特定 system prompt 的响应 (子串匹配)。

        匹配优先级: 先注入的先匹配 (FIFO)。
        """
        self._injected[system_prompt_substring] = response
        # 注入新响应后清空缓存,确保后续调用使用新注入
        self._cache.clear()

    @property
    def call_count(self) -> int:
        """总调用次数 (含缓存命中)。"""
        return self._call_count

    @property
    def cache_hit_count(self) -> int:
        """缓存命中次数 (未实际计算的调用)。"""
        return self._cache_hit_count

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        cache_key = hashlib.sha256(
            (system_prompt + "\x00" + user_prompt).encode()
        ).hexdigest()

        if cache_key in self._cache:
            self._call_count += 1
            self._cache_hit_count += 1
            return self._cache[cache_key]

        # 匹配注入的响应
        response = self._default
        for substring, injected_response in self._injected.items():
            if substring in system_prompt:
                response = injected_response
                break

        self._cache[cache_key] = response
        self._call_count += 1
        return response


def parse_llm_json(response: str) -> dict:
    """解析 LLM 返回的 JSON 响应。

    处理常见 LLM 输出问题:
    - 前后空白
    - markdown 代码块包裹 (```json ... ```)
    - 单条或多条 JSON 对象
    - 无法解析时返回空 dict (不抛异常)
    """
    text = response.strip()
    # 去除 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        # 去首行 (```json) 和末行 (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
