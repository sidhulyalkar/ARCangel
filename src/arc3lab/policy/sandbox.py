from __future__ import annotations

import ast
import collections
import heapq
import itertools
import math
import sys
from typing import Any


class SandboxError(RuntimeError):
    pass


_DISALLOWED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)
_DISALLOWED_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "help",
    "breakpoint",
}


def _safe_range(*args: int) -> range:
    out = range(*args)
    if len(out) > 20_000:
        raise SandboxError("range too large")
    return out


def _validate(tree: ast.AST, *, max_nodes: int = 2200) -> None:
    nodes = list(ast.walk(tree))
    if len(nodes) > max_nodes:
        raise SandboxError(f"analysis program too large: {len(nodes)} AST nodes")
    for node in nodes:
        if isinstance(node, _DISALLOWED_NODES):
            raise SandboxError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SandboxError("private/dunder attribute access is disabled")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxError("dunder names are disabled")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DISALLOWED_CALLS:
                raise SandboxError(f"disallowed call: {node.func.id}")


def run_analysis_code(
    code: str,
    context: dict[str, Any],
    *,
    max_source_chars: int = 7000,
    max_line_events: int = 40_000,
    max_output_chars: int = 5000,
) -> str:
    """Execute model-written analysis code against read-only-ish ARC state.

    The sandbox is intentionally an analysis tool, not an environment-action tool.
    It has no filesystem, network, imports, eval/exec, or dunder access. A trace
    budget stops accidental infinite Python loops. The only exported answer is the
    variable ``result``.
    """
    if not isinstance(code, str) or not code.strip():
        raise SandboxError("empty analysis program")
    if len(code) > max_source_chars:
        raise SandboxError("analysis program exceeds source limit")
    tree = ast.parse(code, mode="exec")
    _validate(tree)

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": _safe_range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    env: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": math,
        "itertools": itertools,
        "collections": collections,
        "heapq": heapq,
        "deque": collections.deque,
        **context,
    }

    events = 0

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal events
        if event in {"line", "call"}:
            events += 1
            if events > max_line_events:
                raise SandboxError("analysis execution budget exceeded")
        return tracer

    old_trace = sys.gettrace()
    try:
        sys.settrace(tracer)
        exec(compile(tree, "<arcangel-analysis>", "exec"), env, env)
    finally:
        sys.settrace(old_trace)

    if "result" not in env:
        raise SandboxError("analysis code must assign a variable named result")
    text = repr(env["result"])
    if len(text) > max_output_chars:
        text = text[:max_output_chars] + "...<truncated>"
    return text
