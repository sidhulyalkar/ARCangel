from __future__ import annotations

import pytest

from scripts.llm_patch_worker import _extract_diff, _validate_diff


def test_extracts_fenced_unified_diff() -> None:
    text = """Here is the patch:\n```diff\ndiff --git a/src/arc3lab/policy/x.py b/src/arc3lab/policy/x.py\n--- a/src/arc3lab/policy/x.py\n+++ b/src/arc3lab/policy/x.py\n@@ -1 +1 @@\n-A=1\n+A=2\n```\n"""
    diff = _extract_diff(text)
    assert diff.startswith("diff --git")
    assert _validate_diff(diff) == ("src/arc3lab/policy/x.py",)


def test_rejects_patch_that_touches_judge_owned_code() -> None:
    diff = """diff --git a/src/arc3lab/arena/scoring.py b/src/arc3lab/arena/scoring.py\n--- a/src/arc3lab/arena/scoring.py\n+++ b/src/arc3lab/arena/scoring.py\n@@ -1 +1 @@\n-A=1\n+A=2\n"""
    with pytest.raises(ValueError, match="forbidden"):
        _validate_diff(diff)


def test_rejects_support_only_patch() -> None:
    diff = """diff --git a/tests/test_new.py b/tests/test_new.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/test_new.py\n@@ -0,0 +1 @@\n+def test_x(): assert True\n"""
    with pytest.raises(ValueError, match="cognition"):
        _validate_diff(diff)


def test_accepts_cognition_plus_support_patch() -> None:
    diff = """diff --git a/src/arc3lab/memory/new.py b/src/arc3lab/memory/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/src/arc3lab/memory/new.py\n@@ -0,0 +1 @@\n+VALUE = 1\ndiff --git a/tests/test_new.py b/tests/test_new.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/test_new.py\n@@ -0,0 +1 @@\n+def test_x(): assert True\n"""
    assert _validate_diff(diff) == (
        "src/arc3lab/memory/new.py",
        "tests/test_new.py",
    )
