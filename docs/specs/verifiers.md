# Verifiers Spec

## 概述

Verifiers 是 Decay Sentinel 漏斗 Step 2 的可插拔验证器集合，承担 `quarantine` 算符的**确定性证据收集层**——它们不调用 LLM，对 cell 的 preconditions 做 grep / AST / git diff / lockfile 之类的事实核查，把结果返还 Decay Sentinel 用于裁决。

定位说明：Verifier 是 `quarantine` 算符的子例程，不是独立模块。它本身不调算符、不写 TreeStore，只回答"该 cell 的前置假设是否仍成立"。这一切分让 Decay Sentinel 的漏斗 Step 0/1/2/3 都可独立度量误判率，是 Harness Card **control lag** 属性回归测试的基础。

## 架构

```
DecaySentinel (漏斗 Step 2)
    ↓
VerifierRegistry.verify(cell)
    ↓ 路由到对应 verifier
┌─────────────────────────────────────┐
│ FileGrepVerifier     (file_grep)    │
│ LockfileQueryVerifier(lockfile)     │
│ TestIdLookupVerifier (test_id)      │
│ AstQueryVerifier     (ast_query)    │
│ EnvCheckVerifier     (env_check)    │
│ GitDiffVerifier      (git_diff)     │  ← SWE-bench 特有
└─────────────────────────────────────┘
```

## 接口定义

```python
from typing import Protocol, Literal
from dataclasses import dataclass

@dataclass
class VerifyResult:
    status: Literal["valid", "invalid", "inconclusive"]
    evidence: Optional[str] = None    # 验证依据描述
    cost: float = 0.0                 # 执行耗时(seconds)


class VerifierProtocol(Protocol):
    def can_handle(self, hint_type: str) -> bool:
        """该 verifier 是否处理此 hint type"""
        ...

    def verify(self, hint: dict, repo_path: str) -> VerifyResult:
        """
        对单个 verify_hint 执行验证。
        hint: {"type": "file_grep", "path": "...", "pattern": "..."}
        repo_path: 代码仓库的本地路径
        返回 valid/invalid/inconclusive
        """
        ...


class VerifierRegistryProtocol(Protocol):
    def __init__(self, verifiers: list[Verifier], repo_path: str):
        ...

    def verify_cell(self, cell: Cell) -> CellVerifyResult:
        """
        验证 cell 的所有 preconditions。
        对每个带 verify_hint 的 precondition 调用对应 verifier。
        汇总结果。
        """
        ...

    def register(self, verifier: Verifier) -> None:
        """注册新的 verifier"""
        ...
```

## CellVerifyResult 结构

```python
@dataclass
class CellVerifyResult:
    cell_id: str
    overall: Literal["valid", "invalid", "inconclusive"]
    details: list[PreconditionVerifyResult]
    total_cost: float

@dataclass
class PreconditionVerifyResult:
    precondition_index: int
    kind: str
    assertion: str
    result: VerifyResult
```

## 汇总规则

```python
def _aggregate(self, results: list[VerifyResult]) -> str:
    if any(r.status == "invalid" for r in results):
        return "invalid"      # 任一 precondition 失效 → cell 无效
    if all(r.status == "valid" for r in results):
        return "valid"        # 全部通过 → cell 有效
    return "inconclusive"     # 有些无法验证 → 不确定
```

- 一票否决：任何一个 precondition 验证为 invalid，整个 cell 判 invalid
- 全票通过：所有可验证的 precondition 都 valid → cell valid
- 其他情况：inconclusive → 进入 Step 3 LLM 裁决

## 各 Verifier 实现

### FileGrepVerifier

```python
class FileGrepVerifier:
    """处理 verify_hint type = 'file_grep'"""

    def verify(self, hint, repo_path):
        # hint: {"type": "file_grep", "path": "settings.py", "pattern": "ENGINE.*postgres"}
        target = os.path.join(repo_path, hint["path"])
        if not os.path.exists(target):
            return VerifyResult("inconclusive", "file not found")
        
        result = subprocess.run(
            ["grep", "-E", hint["pattern"], target],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return VerifyResult("valid", f"matched: {result.stdout[:100]}")
        return VerifyResult("invalid", "pattern not found")
```

### LockfileQueryVerifier

```python
class LockfileQueryVerifier:
    """处理 verify_hint type = 'lockfile_query'"""

    def verify(self, hint, repo_path):
        # hint: {"type": "lockfile_query", "pkg": "django", "constraint": ">=3.0"}
        # 查找 requirements.txt / setup.cfg / pyproject.toml / package.json
        version = self._find_package_version(hint["pkg"], repo_path)
        if version is None:
            return VerifyResult("inconclusive", "package not found in lockfiles")
        if self._satisfies(version, hint["constraint"]):
            return VerifyResult("valid", f"{hint['pkg']}=={version}")
        return VerifyResult("invalid", f"{hint['pkg']}=={version} does not satisfy {hint['constraint']}")
```

### TestIdLookupVerifier

```python
class TestIdLookupVerifier:
    """处理 verify_hint type = 'test_id_lookup'"""

    def verify(self, hint, repo_path):
        # hint: {"type": "test_id_lookup", "test_id": "test_ordering_null"}
        # 在 repo 中搜索 test 函数定义
        result = subprocess.run(
            ["grep", "-r", f"def {hint['test_id']}", repo_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return VerifyResult("valid", f"test exists: {result.stdout.split(chr(10))[0]}")
        return VerifyResult("invalid", f"test {hint['test_id']} not found")
```

**注意**：TestIdLookup 只验证测试存在性。实际跑测试（Step 2a）由 DecaySentinel 直接处理，不经过 VerifierRegistry。

### AstQueryVerifier

```python
class AstQueryVerifier:
    """处理 verify_hint type = 'ast_query'"""

    def verify(self, hint, repo_path):
        # hint: {"type": "ast_query", "query": "find Model fields with null=True..."}
        # AST 查询较复杂，初版用 grep 近似 + 标记 inconclusive
        # 后续可接入 tree-sitter
        return VerifyResult("inconclusive", "ast_query not yet implemented, defer to Step 3")
```

初版将 ast_query 标记为 inconclusive，交给 Step 3 LLM 处理。这是 MVP 策略：先跑通漏斗，后续再增强 Step 2 覆盖率。

### EnvCheckVerifier

```python
class EnvCheckVerifier:
    """处理 verify_hint type = 'env_check'"""

    def verify(self, hint, repo_path):
        # hint: {"type": "env_check", "var": "DATABASE_URL", "expected": "postgres://..."}
        value = os.environ.get(hint["var"])
        if value is None:
            return VerifyResult("inconclusive", f"env var {hint['var']} not set")
        if hint.get("expected") and hint["expected"] in value:
            return VerifyResult("valid", f"{hint['var']}={value[:50]}")
        return VerifyResult("invalid", f"{hint['var']} does not match expected")
```

### GitDiffVerifier（SWE-bench 特有）

```python
class GitDiffVerifier:
    """处理 verify_hint type = 'git_diff'"""

    def verify(self, hint, repo_path):
        # hint: {"type": "git_diff", "path": "django/db/models/query.py", "since_commit": "abc123"}
        # 检查文件自某 commit 以来是否被修改
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", hint["since_commit"], "--", hint["path"]],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return VerifyResult("inconclusive", "git diff failed")
        if hint["path"] in result.stdout:
            return VerifyResult("invalid", f"{hint['path']} modified since {hint['since_commit'][:8]}")
        return VerifyResult("valid", f"{hint['path']} unchanged since {hint['since_commit'][:8]}")
```

## 安全约束

1. 所有外部命令设置 timeout（默认 5~10s），超时 → inconclusive
2. grep/git 命令只执行读操作，不修改任何文件
3. 路径拼接使用 os.path.join + 路径遍历检查（禁止 `..` 越界）
4. 敏感环境变量（含 SECRET/TOKEN/KEY）的值不写入 evidence

## 扩展方式

新增 verifier 只需：
1. 实现 VerifierProtocol (can_handle + verify)
2. 调用 registry.register(new_verifier) 注册

不需要修改 DecaySentinel 或 VerifierRegistry 代码。

## 测试用例

1. FileGrepVerifier：文件存在且 pattern 匹配 → valid
2. FileGrepVerifier：文件存在但 pattern 不匹配 → invalid
3. FileGrepVerifier：文件不存在 → inconclusive
4. LockfileQueryVerifier：package 存在且满足版本约束 → valid
5. LockfileQueryVerifier：package 版本不满足约束 → invalid
6. TestIdLookupVerifier：test 函数存在 → valid
7. TestIdLookupVerifier：test 函数不存在 → invalid
8. AstQueryVerifier：任何输入 → inconclusive（初版策略）
9. GitDiffVerifier：文件未变 → valid；文件已变 → invalid
10. Registry 汇总：一个 invalid + 两个 valid → overall invalid
11. Registry 汇总：全 valid → overall valid
12. Registry 汇总：无 invalid 但有 inconclusive → overall inconclusive
13. 无 verify_hint 的 precondition → 跳过，不影响汇总结果
14. 命令超时 → inconclusive（不崩溃）
