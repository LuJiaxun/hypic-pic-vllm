"""Static check for names that are read but never bound in their scope.

Catches the class of bug that bit the 11E-E cleanup twice: an edit deletes a
helper or a local initialiser while its uses remain, and `compileall` is happy
because the file still parses.

Rules:
  * module scope binds: top-level defs/classes/imports/assignments
  * function scope binds: parameters, its own assignments, nested def/class
    *names* (not their bodies), except/with targets, comprehension targets
  * a load is fine if it is bound in the current scope, an enclosing function
    scope, the module scope, or is a builtin
"""

from __future__ import annotations

import ast
import builtins
import pathlib
import sys

TARGETS = sys.argv[1:] or ["vllm/v1/pic", "vllm/v1/worker/gpu_model_runner.py"]
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
problems: list[str] = []


def binds(node: ast.AST) -> set[str]:
    """Names bound in this scope; nested function bodies are not descended."""
    names: set[str] = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            names.update(child.names)
        elif isinstance(child, ast.ExceptHandler):
            if child.name:
                names.add(child.name)
            names |= binds(child)
        elif isinstance(child, ast.arg):
            names.add(child.arg)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        else:
            names |= binds(child)
    return names


def loads(node: ast.AST) -> list[ast.Name]:
    """Loads in this scope, without descending into nested function bodies."""
    out: list[ast.Name] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            out.append(child)
        else:
            out.extend(loads(child))
    return out


def check(
    node: ast.AST,
    visible: set[str],
    enclosing: set[str],
    qualified: str,
    path: pathlib.Path,
) -> None:
    for name in loads(node):
        if name.id in visible or name.id in enclosing or name.id in BUILTINS:
            continue
        problems.append(f"{path}:{name.lineno}: {qualified}: unbound '{name.id}'")
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = enclosing | visible | binds(child)
            check(child, inner, enclosing, f"{qualified}.{child.name}", path)
        elif isinstance(child, ast.ClassDef):
            inner = enclosing | visible | binds(child)
            check(child, inner, enclosing, f"{qualified}.{child.name}", path)


paths: list[pathlib.Path] = []
for target in TARGETS:
    p = pathlib.Path(target)
    paths.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])

for path in paths:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    check(tree, binds(tree), set(), path.stem, path)

if problems:
    print("\n".join(problems))
    print(f"\n{len(problems)} problem(s)")
    raise SystemExit(1)
print(f"clean: {len(paths)} files, no unbound names")
