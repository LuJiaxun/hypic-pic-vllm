# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find dataclass fields whose defaults break the class definition.

A dataclass may not declare a required field after a field with a default. Python raises TypeError
while the class decorator runs, which at import time means the package cannot be imported at all -
and ``compileall`` cannot see it, because the file parses happily.

    python3 tools/pic_check_dataclass_order.py vllm/v1

Exits non-zero and names each offending field, so this is caught offline rather than by a server that
will not start.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_DATACLASS_DECORATORS = {"dataclass", "dataclasses.dataclass"}


def _is_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id in _DATACLASS_DECORATORS:
            return True
        if isinstance(target, ast.Attribute) and ast.unparse(target) in _DATACLASS_DECORATORS:
            return True
    return False


def _has_default(node: ast.AnnAssign) -> bool:
    return node.value is not None


def check(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:  # pragma: no cover - a syntax error is a louder problem
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not _is_dataclass(node):
            continue
        seen_default: str | None = None
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            if not isinstance(statement.target, ast.Name):
                continue
            name = statement.target.id
            if _has_default(statement):
                seen_default = name
            elif seen_default is not None:
                problems.append(
                    f"{path}:{statement.lineno}: {node.name}.{name} has no default but follows "
                    f"{seen_default}, which has one; Python refuses to build the class"
                )
    return problems


def main(argv: list[str]) -> int:
    roots = [Path(item) for item in argv[1:]] or [Path("vllm/v1")]
    problems: list[str] = []
    checked = 0
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in paths:
            checked += 1
            problems.extend(check(path))
    if problems:
        print("\n".join(problems))
        print(f"\n{len(problems)} dataclass ordering problem(s) in {checked} files")
        return 1
    print(f"clean: {checked} files, every dataclass default follows its required fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
