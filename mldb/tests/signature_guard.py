"""AST-based guard for frozen skeleton public signatures.

The skeleton is parsed as source only. It is never imported into the runtime
implementation package.
"""

from __future__ import annotations

import ast
from pathlib import Path


def compare_module_signatures(
    skeleton_file: Path,
    implementation_file: Path,
) -> tuple[str, ...]:
    """Return public-surface mismatches for one implemented skeleton module."""

    skeleton = _module_surface(skeleton_file)
    implementation = _module_surface(implementation_file)
    relative_name = implementation_file.name
    mismatches: list[str] = []

    _compare_named_surfaces(
        mismatches,
        relative_name,
        "function",
        skeleton["functions"],
        implementation["functions"],
    )
    _compare_named_surfaces(
        mismatches,
        relative_name,
        "class",
        skeleton["classes"],
        implementation["classes"],
    )
    _compare_named_surfaces(
        mismatches,
        relative_name,
        "binding",
        skeleton["bindings"],
        implementation["bindings"],
    )

    if skeleton["all"] != implementation["all"] and (
        skeleton["all"] is not None or implementation["all"] is not None
    ):
        mismatches.append(
            f"{relative_name}: __all__ differs: "
            f"skeleton={skeleton['all']!r}, implementation={implementation['all']!r}"
        )

    return tuple(mismatches)


def collect_signature_mismatches(
    skeleton_root: Path,
    implementation_root: Path,
) -> tuple[str, ...]:
    """Compare every implementation module that has a frozen skeleton counterpart.

    Missing implementation modules are intentionally skipped during incremental
    implementation. ``__init__.py`` files are package infrastructure and are skipped.
    An implementation-only module is allowed when it is private (an underscore-prefixed
    path component) or when it exposes no public functions, classes, bindings, or literal
    ``__all__`` surface.
    """

    mismatches: list[str] = []
    for implementation_file in sorted(implementation_root.rglob("*.py")):
        if implementation_file.name == "__init__.py":
            continue

        relative = implementation_file.relative_to(implementation_root)
        skeleton_file = skeleton_root / relative
        if skeleton_file.is_file():
            module_mismatches = compare_module_signatures(
                skeleton_file,
                implementation_file,
            )
            mismatches.extend(
                f"{relative.as_posix()}: {message.split(': ', 1)[1]}"
                for message in module_mismatches
            )
            continue

        if any(part.startswith("_") for part in relative.parts):
            continue

        surface = _module_surface(implementation_file)
        public_names = sorted(
            [
                *surface["functions"].keys(),
                *surface["classes"].keys(),
                *surface["bindings"].keys(),
            ]
        )
        if public_names or surface["all"] is not None:
            mismatches.append(
                f"{relative.as_posix()}: implementation-only module exposes public "
                f"surface: {public_names!r}, __all__={surface['all']!r}"
            )

    return tuple(mismatches)


def _compare_named_surfaces(
    mismatches: list[str],
    module_name: str,
    kind: str,
    skeleton: dict[str, object],
    implementation: dict[str, object],
) -> None:
    skeleton_names = set(skeleton)
    implementation_names = set(implementation)

    for name in sorted(skeleton_names - implementation_names):
        mismatches.append(f"{module_name}: missing public {kind} {name}")
    for name in sorted(implementation_names - skeleton_names):
        mismatches.append(f"{module_name}: unexpected public {kind} {name}")
    for name in sorted(skeleton_names & implementation_names):
        if skeleton[name] != implementation[name]:
            mismatches.append(f"{module_name}: public {kind} {name} signature differs")


def _module_surface(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions: dict[str, object] = {}
    classes: dict[str, object] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            functions[node.name] = _function_signature(node)
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            classes[node.name] = _class_surface(node)

    return {
        "functions": functions,
        "classes": classes,
        "bindings": _public_bindings(tree),
        "all": _literal_all(tree),
    }


def _class_surface(node: ast.ClassDef) -> tuple[object, ...]:
    is_dataclass = any(_is_dataclass_decorator(item) for item in node.decorator_list)
    is_enum = any(_decorator_name(base) == "Enum" for base in node.bases)
    methods = {
        item.name: _method_signature(item)
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _is_public(item.name)
    }
    fields = _dataclass_fields(node) if is_dataclass else ()
    enum_members = _enum_members(node) if is_enum else ()
    class_kind = "dataclass" if is_dataclass else "enum" if is_enum else "class"
    return (class_kind, fields, enum_members, methods)


def _public_bindings(tree: ast.Module) -> dict[str, None]:
    bindings: dict[str, None] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and _is_public(target.id):
                bindings[target.id] = None
    return bindings


def _enum_members(node: ast.ClassDef) -> tuple[str, ...]:
    members: list[str] = []
    for item in node.body:
        targets: list[ast.expr] = []
        if isinstance(item, ast.Assign):
            targets = list(item.targets)
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        for target in targets:
            if isinstance(target, ast.Name) and _is_public(target.id):
                members.append(target.id)
    return tuple(members)


def _function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[object, ...]:
    return (
        "async" if isinstance(node, ast.AsyncFunctionDef) else "sync",
        ast.dump(node.args, annotate_fields=True, include_attributes=False),
        ast.dump(node.returns, annotate_fields=True, include_attributes=False)
        if node.returns is not None
        else None,
    )


def _method_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[object, ...]:
    binding = "instance"
    for decorator in node.decorator_list:
        name = _decorator_name(decorator)
        if name in {"classmethod", "staticmethod"}:
            binding = name
            break
    return (*_function_signature(node), binding)


def _dataclass_fields(node: ast.ClassDef) -> tuple[str, ...]:
    fields: list[str] = []
    for item in node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        if not _is_public(item.target.id) or _is_classvar_annotation(item.annotation):
            continue
        fields.append(item.target.id)
    return tuple(fields)


def _is_classvar_annotation(annotation: ast.expr) -> bool:
    text = ast.unparse(annotation)
    return text == "ClassVar" or text.startswith("ClassVar[") or ".ClassVar[" in text


def _is_dataclass_decorator(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return _decorator_name(decorator) == "dataclass"


def _decorator_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def _literal_all(tree: ast.Module) -> tuple[str, ...] | None:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = node.value
        if value is None:
            return None
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError):
            return None
        if isinstance(literal, (tuple, list)) and all(isinstance(item, str) for item in literal):
            return tuple(literal)
        return None
    return None


def _is_public(name: str) -> bool:
    return not name.startswith("_")
