"""Verifiers — DecaySentinel 漏斗 Step 2 的可插拔验证器集合。

承担 quarantine 算符的确定性证据收集层:不调用 LLM,
对 cell 的 preconditions 做 grep / AST / git diff / lockfile 事实核查。

对应 spec: docs/specs/verifiers.md
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Protocol, runtime_checkable

from tree_harness.core.cell_model import Cell, Precondition, VerifyHint


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class VerifyResult:
    """单个 verify_hint 的验证结果。"""
    status: Literal["valid", "invalid", "inconclusive"]
    evidence: Optional[str] = None
    cost: float = 0.0


@dataclass
class PreconditionVerifyResult:
    """单个 precondition 的验证结果。"""
    precondition_index: int
    kind: str
    assertion: str
    result: VerifyResult


@dataclass
class CellVerifyResult:
    """整个 cell 所有 precondition 的汇总验证结果。"""
    cell_id: str
    overall: Literal["valid", "invalid", "inconclusive"]
    details: List[PreconditionVerifyResult] = field(default_factory=list)
    total_cost: float = 0.0


# ---------------------------------------------------------------------------
# Verifier 协议
# ---------------------------------------------------------------------------
@runtime_checkable
class VerifierProtocol(Protocol):
    def can_handle(self, hint_type: str) -> bool: ...
    def verify(self, hint: dict, repo_path: str) -> VerifyResult: ...


# ---------------------------------------------------------------------------
# 安全辅助
# ---------------------------------------------------------------------------
_SENSITIVE_PATTERN = re.compile(r"(SECRET|TOKEN|KEY|PASSWORD)", re.IGNORECASE)


def _safe_path(repo_path: str, sub_path: str) -> Optional[str]:
    """拼接路径并防止目录遍历 (禁止 .. 越界)。"""
    full = os.path.join(repo_path, sub_path)
    real = os.path.realpath(full)
    repo_real = os.path.realpath(repo_path)
    if not real.startswith(repo_real):
        return None
    return real


# ---------------------------------------------------------------------------
# FileGrepVerifier
# ---------------------------------------------------------------------------
class FileGrepVerifier:
    """处理 verify_hint type = 'file_grep'。"""

    def can_handle(self, hint_type: str) -> bool:
        return hint_type == "file_grep"

    def verify(self, hint: dict, repo_path: str) -> VerifyResult:
        target = _safe_path(repo_path, hint.get("path", ""))
        if target is None or not os.path.exists(target):
            return VerifyResult("inconclusive", "file not found or path traversal blocked")
        pattern = hint.get("pattern", "")
        try:
            result = subprocess.run(
                ["grep", "-E", pattern, target],
                capture_output=True, text=True, timeout=5,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult("inconclusive", "grep timed out")
        except FileNotFoundError:
            # grep not available — fallback to Python regex
            return self._python_grep(target, pattern)
        if result.returncode == 0:
            snippet = result.stdout.strip().split("\n")[0][:100]
            return VerifyResult("valid", f"matched: {snippet}")
        return VerifyResult("invalid", "pattern not found in file")

    @staticmethod
    def _python_grep(path: str, pattern: str) -> VerifyResult:
        try:
            with open(path, "r", errors="replace") as f:
                for line in f:
                    if re.search(pattern, line):
                        return VerifyResult("valid", f"matched: {line.strip()[:100]}")
            return VerifyResult("invalid", "pattern not found in file")
        except Exception as e:
            return VerifyResult("inconclusive", f"read error: {e}")


# ---------------------------------------------------------------------------
# LockfileQueryVerifier
# ---------------------------------------------------------------------------
class LockfileQueryVerifier:
    """处理 verify_hint type = 'lockfile_query'。"""

    _LOCKFILES = (
        "requirements.txt", "setup.cfg", "pyproject.toml", "package.json",
    )

    def can_handle(self, hint_type: str) -> bool:
        return hint_type == "lockfile_query"

    def verify(self, hint: dict, repo_path: str) -> VerifyResult:
        pkg = hint.get("pkg", "")
        constraint = hint.get("constraint", "")
        version = self._find_package_version(pkg, repo_path)
        if version is None:
            return VerifyResult("inconclusive", f"package {pkg} not found in lockfiles")
        if self._satisfies(version, constraint):
            return VerifyResult("valid", f"{pkg}=={version}")
        return VerifyResult(
            "invalid", f"{pkg}=={version} does not satisfy {constraint}"
        )

    def _find_package_version(self, pkg: str, repo_path: str) -> Optional[str]:
        for lockfile in self._LOCKFILES:
            path = os.path.join(repo_path, lockfile)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue
            version = self._extract_version(pkg, content, lockfile)
            if version is not None:
                return version
        return None

    @staticmethod
    def _extract_version(pkg: str, content: str, lockfile: str) -> Optional[str]:
        pkg_lower = pkg.lower()
        if lockfile == "requirements.txt":
            for line in content.splitlines():
                line = line.strip()
                if line.lower().startswith(pkg_lower):
                    # e.g. "django>=3.0" or "django==4.2"
                    m = re.search(r"[=<>!~]+\s*([\d.]+)", line)
                    return m.group(1) if m else None
        elif lockfile == "pyproject.toml":
            # TOML dependency: "django>=4.2" or django = "4.2" or django>=4.2
            m = re.search(
                rf'{re.escape(pkg)}["\']?\s*[=<>!~]*\s*["\']?([\d.]+)', content, re.IGNORECASE
            )
            return m.group(1) if m else None
        elif lockfile == "setup.cfg":
            m = re.search(
                rf'{re.escape(pkg)}["\']?\s*[=<>!~]+\s*([\d.]+)', content, re.IGNORECASE
            )
            return m.group(1) if m else None
        elif lockfile == "package.json":
            # JSON is more structured, but simple regex works for version strings
            m = re.search(
                rf'"{re.escape(pkg)}"\s*:\s*"([^"]+)"', content, re.IGNORECASE
            )
            return m.group(1) if m else None
        return None

    @staticmethod
    def _satisfies(version: str, constraint: str) -> bool:
        """Simple version constraint check.

        Supports: >=X, >X, ==X, <=X, <X, ~=X, bare version (exact).
        """
        constraint = constraint.strip()
        if not constraint:
            return True
        m = re.match(r"(>=|<=|==|~=|>|<)?\s*([\d.]+)", constraint)
        if not m:
            return True
        op, required = m.group(1) or "==", m.group(2)
        v_parts = [int(x) for x in version.split(".") if x.isdigit()]
        r_parts = [int(x) for x in required.split(".") if x.isdigit()]
        # pad
        max_len = max(len(v_parts), len(r_parts))
        v_parts += [0] * (max_len - len(v_parts))
        r_parts += [0] * (max_len - len(r_parts))
        if op == ">=":
            return v_parts >= r_parts
        elif op == ">":
            return v_parts > r_parts
        elif op == "<=":
            return v_parts <= r_parts
        elif op == "<":
            return v_parts < r_parts
        elif op == "~=":
            # compatible release: same major.minor
            if len(r_parts) >= 2:
                return v_parts[:2] == r_parts[:2] and v_parts >= r_parts
            return v_parts >= r_parts
        else:  # ==
            return v_parts == r_parts


# ---------------------------------------------------------------------------
# TestIdLookupVerifier
# ---------------------------------------------------------------------------
class TestIdLookupVerifier:
    """处理 verify_hint type = 'test_id_lookup'。

    只验证测试存在性。实际跑测试 (Step 2a) 由 DecaySentinel 直接处理。
    """

    def can_handle(self, hint_type: str) -> bool:
        return hint_type == "test_id_lookup"

    def verify(self, hint: dict, repo_path: str) -> VerifyResult:
        test_id = hint.get("test_id", "")
        try:
            result = subprocess.run(
                ["grep", "-r", "--include=*.py", f"def {test_id}", repo_path],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult("inconclusive", "grep timed out")
        except FileNotFoundError:
            return self._python_search(repo_path, test_id)
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().split("\n")[0]
            return VerifyResult("valid", f"test exists: {test_id} in {os.path.basename(first_line.split(':')[0])}")
        return VerifyResult("invalid", f"test {test_id} not found")

    @staticmethod
    def _python_search(repo_path: str, test_id: str) -> VerifyResult:
        pattern = f"def {test_id}"
        for dirpath, _dirs, files in os.walk(repo_path):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", errors="replace") as f:
                        for line in f:
                            if pattern in line:
                                return VerifyResult(
                                    "valid", f"test exists: {test_id} in {fname}"
                                )
                except Exception:
                    continue
        return VerifyResult("invalid", f"test {test_id} not found")


# ---------------------------------------------------------------------------
# AstQueryVerifier
# ---------------------------------------------------------------------------
class AstQueryVerifier:
    """处理 verify_hint type = 'ast_query'。

    初版用 grep 近似 + 标记 inconclusive,后续可接入 tree-sitter。
    """

    def can_handle(self, hint_type: str) -> bool:
        return hint_type == "ast_query"

    def verify(self, hint: dict, repo_path: str) -> VerifyResult:
        return VerifyResult(
            "inconclusive", "ast_query not yet implemented, defer to Step 3"
        )


# ---------------------------------------------------------------------------
# EnvCheckVerifier
# ---------------------------------------------------------------------------
class EnvCheckVerifier:
    """处理 verify_hint type = 'env_check'。"""

    def can_handle(self, hint_type: str) -> bool:
        return hint_type == "env_check"

    def verify(self, hint: dict, repo_path: str) -> VerifyResult:
        var_name = hint.get("var", "")
        expected = hint.get("expected")
        value = os.environ.get(var_name)
        if value is None:
            return VerifyResult("inconclusive", f"env var {var_name} not set")
        if _SENSITIVE_PATTERN.search(var_name):
            return VerifyResult("valid", f"{var_name} is set (value redacted)")
        if expected and expected in value:
            return VerifyResult("valid", f"{var_name}={value[:50]}")
        if expected is None:
            return VerifyResult("valid", f"{var_name} is set")
        return VerifyResult("invalid", f"{var_name} does not match expected")


# ---------------------------------------------------------------------------
# GitDiffVerifier
# ---------------------------------------------------------------------------
class GitDiffVerifier:
    """处理 verify_hint type = 'git_diff' (SWE-bench 特有)。"""

    def can_handle(self, hint_type: str) -> bool:
        return hint_type == "git_diff"

    def verify(self, hint: dict, repo_path: str) -> VerifyResult:
        path = hint.get("path", "")
        since_commit = hint.get("since_commit", "")
        try:
            result = subprocess.run(
                ["git", "-C", repo_path, "diff", "--name-only", since_commit, "--", path],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult("inconclusive", "git diff timed out")
        except FileNotFoundError:
            return VerifyResult("inconclusive", "git not available")
        if result.returncode != 0:
            return VerifyResult("inconclusive", "git diff failed")
        if path in result.stdout:
            return VerifyResult(
                "invalid", f"{path} modified since {since_commit[:8]}"
            )
        return VerifyResult(
            "valid", f"{path} unchanged since {since_commit[:8]}"
        )


# ---------------------------------------------------------------------------
# VerifierRegistry
# ---------------------------------------------------------------------------
_DEFAULT_VERIFIERS: List[VerifierProtocol] = [
    FileGrepVerifier(),
    LockfileQueryVerifier(),
    TestIdLookupVerifier(),
    AstQueryVerifier(),
    EnvCheckVerifier(),
    GitDiffVerifier(),
]


class VerifierRegistry:
    """验证器注册表 — 分发 verify_hint 到对应 verifier 并汇总结果。"""

    def __init__(
        self,
        verifiers: Optional[List[VerifierProtocol]] = None,
        repo_path: str = ".",
    ):
        self._verifiers: List[VerifierProtocol] = list(verifiers) if verifiers else list(_DEFAULT_VERIFIERS)
        self.repo_path = repo_path

    def register(self, verifier: VerifierProtocol) -> None:
        self._verifiers.append(verifier)

    def verify_hint(self, hint: dict, repo_path: Optional[str] = None) -> VerifyResult:
        """对单个 verify_hint 执行验证,返回结果。"""
        rp = repo_path or self.repo_path
        hint_type = hint.get("type", "")
        for v in self._verifiers:
            if v.can_handle(hint_type):
                return v.verify(hint, rp)
        return VerifyResult("inconclusive", f"no verifier for type '{hint_type}'")

    def verify_cell(self, cell: Cell, repo_path: Optional[str] = None) -> CellVerifyResult:
        """验证 cell 的所有 preconditions,汇总结果。"""
        rp = repo_path or self.repo_path
        details: List[PreconditionVerifyResult] = []
        total_cost = 0.0

        for idx, precond in enumerate(cell.context_preconditions):
            if precond.verify_hint is None:
                continue  # 无 verify_hint 的 precondition 跳过

            hint_dict = {
                "type": precond.verify_hint.type,
                **precond.verify_hint.params,
            }
            result = self.verify_hint(hint_dict, rp)
            total_cost += result.cost
            details.append(PreconditionVerifyResult(
                precondition_index=idx,
                kind=precond.kind,
                assertion=precond.assertion,
                result=result,
            ))

        overall = self._aggregate([d.result for d in details])
        return CellVerifyResult(
            cell_id=cell.id,
            overall=overall,
            details=details,
            total_cost=total_cost,
        )

    @staticmethod
    def _aggregate(results: List[VerifyResult]) -> Literal["valid", "invalid", "inconclusive"]:
        """汇总规则: 一票否决 + 全票通过 + 其他不确定。"""
        if not results:
            return "valid"  # 无可验证 precondition → 默认 valid
        if any(r.status == "invalid" for r in results):
            return "invalid"
        if all(r.status == "valid" for r in results):
            return "valid"
        return "inconclusive"
