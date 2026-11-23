# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find annotations that name a class defined later in the same module.

Python evaluates a function's annotations when the ``def`` statement runs. At module level that
happens at import time, so a module-level function annotated with a class defined below it raises
NameError and the whole package fails to import - a failure that ``compileall`` cannot see, because
the file parses happily.

    python3 tools/pic_check_forward_refs.py vllm/v1/pic

Exits non-zero and names each offending line, so this is caught offline rather than by a server that
will not start.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def check(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - a syntax error is a louder problem
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    # With PEP 563 every annotation stays a string and is never evaluated at import time, so the
    # module cannot fail this way however it is written.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(alias.name == "annotations" for alias in node.names):
                return []

    # Every top-level class and when it becomes visible.
    class_lines: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_lines.setdefault(node.name, node.lineno)

    problems: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        identifiers: set[str] = set()
        for annotated in _annotations(node):
            for inner in ast.walk(annotated):
                if isinstance(inner, ast.Name):
                    identifiers.add(inner.id)
        # Names introduced by the module's own earlier statements are fine, and so is anything the
        # module imported: only a class defined below this def is a forward reference.
        for name in sorted(identifiers):
            defined_at = class_lines.get(name)
            if defined_at is not None and defined_at > node.lineno:
                problems.append(
                    f"{path}:{node.lineno}: {node.name} annotates with {name}, "
                    f"defined later at line {defined_at}"
                )
    return problems


def _annotations(node: ast.FunctionDef | ast.AsyncFunctionDef):
    arguments = node.args
    for argument in (
        list(arguments.posonlyargs) + list(arguments.args) + list(arguments.kwonlyargs)
    ):
        if argument.annotation is not None:
            yield argument.annotation
    for argument in (arguments.vararg, arguments.kwarg):
        if argument is not None and argument.annotation is not None:
            yield argument.annotation
    if node.returns is not None:
        yield node.returns


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
        print(f"\n{len(problems)} forward reference(s) in {checked} files")
        return 1
    print(f"clean: {checked} files, no forward-referenced annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
