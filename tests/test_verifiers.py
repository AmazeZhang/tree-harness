"""Verifiers 测试 —— 对应 docs/specs/verifiers.md 测试用例。"""
import os
import tempfile

import pytest

from tree_harness.core.cell_model import Cell, Precondition, VerifyHint, create_cell
from tree_harness.modules.verifiers import (
    VerifyResult,
    CellVerifyResult,
    PreconditionVerifyResult,
    FileGrepVerifier,
    LockfileQueryVerifier,
    TestIdLookupVerifier,
    AstQueryVerifier,
    EnvCheckVerifier,
    GitDiffVerifier,
    VerifierRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_repo(tmp_path):
    """创建临时仓库目录。"""
    return str(tmp_path)


@pytest.fixture
def grep_file(tmp_repo):
    """创建一个包含已知内容的文件。"""
    path = os.path.join(tmp_repo, "settings.py")
    with open(path, "w") as f:
        f.write("DATABASE_ENGINE = 'postgres'\n")
        f.write("DATABASE_HOST = 'localhost'\n")
    return path


# ---------------------------------------------------------------------------
# FileGrepVerifier
# ---------------------------------------------------------------------------
class TestFileGrepVerifier:
    def test_file_exists_pattern_matches(self, tmp_repo, grep_file):
        v = FileGrepVerifier()
        result = v.verify(
            {"type": "file_grep", "path": "settings.py", "pattern": "ENGINE.*postgres"},
            tmp_repo,
        )
        assert result.status == "valid"
        assert "matched" in result.evidence

    def test_file_exists_pattern_not_match(self, tmp_repo, grep_file):
        v = FileGrepVerifier()
        result = v.verify(
            {"type": "file_grep", "path": "settings.py", "pattern": "ENGINE.*mysql"},
            tmp_repo,
        )
        assert result.status == "invalid"
        assert "not found" in result.evidence

    def test_file_not_exists(self, tmp_repo):
        v = FileGrepVerifier()
        result = v.verify(
            {"type": "file_grep", "path": "nonexistent.py", "pattern": "foo"},
            tmp_repo,
        )
        assert result.status == "inconclusive"
        assert "not found" in result.evidence

    def test_path_traversal_blocked(self, tmp_repo):
        v = FileGrepVerifier()
        result = v.verify(
            {"type": "file_grep", "path": "../../../etc/passwd", "pattern": "root"},
            tmp_repo,
        )
        assert result.status == "inconclusive"


# ---------------------------------------------------------------------------
# LockfileQueryVerifier
# ---------------------------------------------------------------------------
class TestLockfileQueryVerifier:
    def test_package_satisfies_constraint(self, tmp_repo):
        req_path = os.path.join(tmp_repo, "requirements.txt")
        with open(req_path, "w") as f:
            f.write("django>=3.0\n")
        v = LockfileQueryVerifier()
        result = v.verify(
            {"type": "lockfile_query", "pkg": "django", "constraint": ">=3.0"},
            tmp_repo,
        )
        assert result.status == "valid"
        assert "django" in result.evidence

    def test_package_does_not_satisfy(self, tmp_repo):
        req_path = os.path.join(tmp_repo, "requirements.txt")
        with open(req_path, "w") as f:
            f.write("django==2.2\n")
        v = LockfileQueryVerifier()
        result = v.verify(
            {"type": "lockfile_query", "pkg": "django", "constraint": ">=3.0"},
            tmp_repo,
        )
        assert result.status == "invalid"
        assert "does not satisfy" in result.evidence

    def test_package_not_found(self, tmp_repo):
        req_path = os.path.join(tmp_repo, "requirements.txt")
        with open(req_path, "w") as f:
            f.write("flask==2.0\n")
        v = LockfileQueryVerifier()
        result = v.verify(
            {"type": "lockfile_query", "pkg": "django", "constraint": ">=3.0"},
            tmp_repo,
        )
        assert result.status == "inconclusive"
        assert "not found" in result.evidence

    def test_pyproject_toml(self, tmp_repo):
        pyproject = os.path.join(tmp_repo, "pyproject.toml")
        with open(pyproject, "w") as f:
            f.write('[project]\ndependencies = ["django>=4.2"]\n')
        v = LockfileQueryVerifier()
        result = v.verify(
            {"type": "lockfile_query", "pkg": "django", "constraint": ">=4.0"},
            tmp_repo,
        )
        assert result.status == "valid"


# ---------------------------------------------------------------------------
# TestIdLookupVerifier
# ---------------------------------------------------------------------------
class TestTestIdLookupVerifier:
    def test_test_exists(self, tmp_repo):
        test_file = os.path.join(tmp_repo, "test_things.py")
        with open(test_file, "w") as f:
            f.write("def test_ordering_null():\n    assert True\n")
        v = TestIdLookupVerifier()
        result = v.verify(
            {"type": "test_id_lookup", "test_id": "test_ordering_null"},
            tmp_repo,
        )
        assert result.status == "valid"
        assert "test_ordering_null" in result.evidence

    def test_test_not_found(self, tmp_repo):
        v = TestIdLookupVerifier()
        result = v.verify(
            {"type": "test_id_lookup", "test_id": "test_nonexistent"},
            tmp_repo,
        )
        assert result.status == "invalid"
        assert "not found" in result.evidence


# ---------------------------------------------------------------------------
# AstQueryVerifier
# ---------------------------------------------------------------------------
class TestAstQueryVerifier:
    def test_always_inconclusive(self, tmp_repo):
        v = AstQueryVerifier()
        result = v.verify(
            {"type": "ast_query", "query": "find Model fields with null=True"},
            tmp_repo,
        )
        assert result.status == "inconclusive"
        assert "not yet implemented" in result.evidence


# ---------------------------------------------------------------------------
# EnvCheckVerifier
# ---------------------------------------------------------------------------
class TestEnvCheckVerifier:
    def test_env_set_and_matches(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgres://localhost:5432")
        v = EnvCheckVerifier()
        result = v.verify(
            {"type": "env_check", "var": "DATABASE_URL", "expected": "postgres://"},
            ".",
        )
        assert result.status == "valid"

    def test_env_not_set(self, monkeypatch):
        monkeypatch.delenv("MY_MISSING_VAR", raising=False)
        v = EnvCheckVerifier()
        result = v.verify(
            {"type": "env_check", "var": "MY_MISSING_VAR"},
            ".",
        )
        assert result.status == "inconclusive"

    def test_sensitive_var_redacted(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "abc123")
        v = EnvCheckVerifier()
        result = v.verify(
            {"type": "env_check", "var": "SECRET_KEY"},
            ".",
        )
        assert result.status == "valid"
        assert "redacted" in result.evidence


# ---------------------------------------------------------------------------
# GitDiffVerifier
# ---------------------------------------------------------------------------
class TestGitDiffVerifier:
    def test_file_unchanged(self, tmp_repo):
        # Init git repo with a file
        with open(os.path.join(tmp_repo, "README.md"), "w") as f:
            f.write("# Test Repo\n")
        os.system(f"cd '{tmp_repo}' && git init -q && git add -A && git commit -q -m initial")
        v = GitDiffVerifier()
        result = v.verify(
            {"type": "git_diff", "path": "README.md", "since_commit": "HEAD"},
            tmp_repo,
        )
        assert result.status == "valid"
        assert "unchanged" in result.evidence

    def test_file_changed(self, tmp_repo):
        with open(os.path.join(tmp_repo, "README.md"), "w") as f:
            f.write("# Test Repo\n")
        os.system(f"cd '{tmp_repo}' && git init -q && git add -A && git commit -q -m initial")
        # Modify a file after commit
        with open(os.path.join(tmp_repo, "app.py"), "w") as f:
            f.write("print('hello')\n")
        os.system(f"cd '{tmp_repo}' && git add -A && git commit -q -m 'add app'")
        # Now app.py should show as changed since initial commit
        initial_commit = os.popen(
            f"cd '{tmp_repo}' && git rev-list --max-parents=0 HEAD"
        ).read().strip()
        v = GitDiffVerifier()
        result = v.verify(
            {"type": "git_diff", "path": "app.py", "since_commit": initial_commit},
            tmp_repo,
        )
        assert result.status == "invalid"
        assert "modified" in result.evidence


# ---------------------------------------------------------------------------
# VerifierRegistry
# ---------------------------------------------------------------------------
class TestVerifierRegistry:
    def test_aggregate_one_invalid_plus_valids(self, tmp_repo, grep_file):
        """一个 invalid + 两个 valid → overall invalid。"""
        preconds = [
            Precondition(
                kind="config",
                assertion="DB engine is postgres",
                verify_hint=VerifyHint(type="file_grep",
                                      params={"path": "settings.py", "pattern": "ENGINE.*postgres"}),
            ),
            Precondition(
                kind="config",
                assertion="DB host is mysql",
                verify_hint=VerifyHint(type="file_grep",
                                      params={"path": "settings.py", "pattern": "HOST.*mysql"}),
            ),
        ]
        cell = create_cell(
            trigger_task="t1", domain="d1",
            decision="use postgres", rationale="pg is better",
            preconditions=preconds,
        )
        reg = VerifierRegistry(repo_path=tmp_repo)
        result = reg.verify_cell(cell)
        assert result.overall == "invalid"

    def test_aggregate_all_valid(self, tmp_repo, grep_file):
        """全 valid → overall valid。"""
        preconds = [
            Precondition(
                kind="config",
                assertion="DB engine is postgres",
                verify_hint=VerifyHint(type="file_grep",
                                      params={"path": "settings.py", "pattern": "ENGINE.*postgres"}),
            ),
            Precondition(
                kind="config",
                assertion="DB host is localhost",
                verify_hint=VerifyHint(type="file_grep",
                                      params={"path": "settings.py", "pattern": "HOST.*localhost"}),
            ),
        ]
        cell = create_cell(
            trigger_task="t1", domain="d1",
            decision="use postgres", rationale="pg is better",
            preconditions=preconds,
        )
        reg = VerifierRegistry(repo_path=tmp_repo)
        result = reg.verify_cell(cell)
        assert result.overall == "valid"

    def test_aggregate_no_invalid_but_inconclusive(self, tmp_repo, grep_file):
        """无 invalid 但有 inconclusive → overall inconclusive。"""
        preconds = [
            Precondition(
                kind="config",
                assertion="DB engine is postgres",
                verify_hint=VerifyHint(type="file_grep",
                                      params={"path": "settings.py", "pattern": "ENGINE.*postgres"}),
            ),
            Precondition(
                kind="code_invariant",
                assertion="some AST check",
                verify_hint=VerifyHint(type="ast_query", params={"query": "find something"}),
            ),
        ]
        cell = create_cell(
            trigger_task="t1", domain="d1",
            decision="use postgres", rationale="pg is better",
            preconditions=preconds,
        )
        reg = VerifierRegistry(repo_path=tmp_repo)
        result = reg.verify_cell(cell)
        assert result.overall == "inconclusive"

    def test_no_verify_hint_preconditions_skipped(self, tmp_repo):
        """无 verify_hint 的 precondition → 跳过,不影响汇总。"""
        preconds = [
            Precondition(kind="convention", assertion="follow PEP 8"),
        ]
        cell = create_cell(
            trigger_task="t1", domain="d1",
            decision="use postgres", rationale="pg is better",
            preconditions=preconds,
        )
        reg = VerifierRegistry(repo_path=tmp_repo)
        result = reg.verify_cell(cell)
        assert result.overall == "valid"
        assert len(result.details) == 0  # 没有可验证的 precondition

    def test_register_custom_verifier(self, tmp_repo):
        """注册自定义 verifier。"""
        class CustomVerifier:
            def can_handle(self, hint_type):
                return hint_type == "custom"
            def verify(self, hint, repo_path):
                return VerifyResult("valid", "custom check passed")

        reg = VerifierRegistry(repo_path=tmp_repo)
        reg.register(CustomVerifier())
        result = reg.verify_hint({"type": "custom"})
        assert result.status == "valid"

    def test_no_verifier_for_type(self, tmp_repo):
        """无对应 verifier → inconclusive。"""
        reg = VerifierRegistry(repo_path=tmp_repo)
        result = reg.verify_hint({"type": "unknown_type"})
        assert result.status == "inconclusive"
        assert "no verifier" in result.evidence
