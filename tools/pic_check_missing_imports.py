# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find a module that uses a name it never imported.

``compileall`` proves a file parses; it says nothing about whether the names in it exist. A module
that calls ``torch.tensor`` without importing ``torch`` compiles, passes every syntax check, starts a
server, and then raises NameError on the first request that reaches that line - which is how three
separate import mistakes reached a running system in a row.

    python3 tools/pic_check_missing_imports.py vllm/v1

The check is deliberately conservative: a name is reported only when it is loaded somewhere in the
file and is neither imported, assigned at module level, defined as a class or function, nor a builtin.
Anything a module reaches through ``globals()`` or a star import is out of scope and stays quiet.
"""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path


def _module_names(tree: ast.Module) -> set[str]:
    """Every name the module binds at top level."""
    names: set[str] = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    # A star import can bind anything; refusing to guess keeps this quiet.
                    names.add("*")
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for inner in ast.walk(target):
                    if isinstance(inner, ast.Name):
                        names.add(inner.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            for inner in ast.walk(node.target):
                if isinstance(inner, ast.Name):
                    names.add(inner.id)
        elif isinstance(node, ast.withitem):
            if node.optional_vars is not None:
                for inner in ast.walk(node.optional_vars):
                    if isinstance(inner, ast.Name):
                        names.add(inner.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.MatchAs):
            if node.name:
                names.add(node.name)
        elif isinstance(node, ast.MatchStar):
            if node.name:
                names.add(node.name)
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                names.add(node.rest)
        elif isinstance(node, (ast.Lambda,)):
            names.update(argument.arg for argument in node.args.args)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.update(argument.arg for argument in node.args.args)
        if isinstance(node, (ast.Global,)):
            names.update(node.names)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            ):
                names.add(argument.arg)
            for argument in (node.args.vararg, node.args.kwarg):
                if argument is not None:
                    names.add(argument.arg)
    return names


def check(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:  # pragma: no cover - a syntax error is a louder problem
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    known = _module_names(tree)
    if "*" in known:
        return []

    problems: list[str] = []
    reported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in known and node.id not in reported:
                reported.add(node.id)
                problems.append(
                    f"{path}:{node.lineno}: {node.id} is used but never imported or defined"
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
        print(f"\n{len(problems)} unresolved name(s) in {checked} files")
        return 1
    print(f"clean: {checked} files, every loaded name is imported or defined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
