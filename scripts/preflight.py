#!/usr/bin/env python
"""Cheap release checks that catch common ARC3 Kaggle submission failures."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = ROOT / "src" / "arc3lab" / "policy"
FORBIDDEN_POLICY_TERMS = (
    "environment_files",
    "baseline_actions",
    "human_actions",
)


def check_python() -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"syntax: {path.relative_to(ROOT)}: {exc}")
    return errors


def check_policy_leakage() -> list[str]:
    errors: list[str] = []
    for path in POLICY_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_POLICY_TERMS:
            if term in text:
                errors.append(f"policy leakage term {term!r}: {path.relative_to(ROOT)}")
    return errors


def check_notebooks() -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / "kaggle").glob("*.ipynb")):
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
            if nb.get("nbformat") != 4:
                errors.append(f"notebook format: {path.name}")
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                try:
                    compile(source, f"{path.name}:cell{i}", "exec")
                except SyntaxError as exc:
                    errors.append(f"notebook syntax: {path.name}:cell{i}: {exc}")
        except Exception as exc:
            errors.append(f"notebook parse: {path.name}: {exc}")
    return errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--require-model", action="store_true")
    args = p.parse_args()

    errors = check_python() + check_policy_leakage() + check_notebooks()
    if os.getenv("OPERATION_MODE") and os.getenv("OPERATION_MODE") not in {"offline", "competition"}:
        errors.append(f"unexpected OPERATION_MODE={os.getenv('OPERATION_MODE')}")
    if args.require_model:
        from arc3lab.model import discover_model_path

        if discover_model_path() is None:
            errors.append("no local model config discovered")

    if errors:
        print("PREFLIGHT FAIL")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)
    print("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
