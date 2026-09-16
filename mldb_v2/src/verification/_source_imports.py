"""Private static discovery of repository-owned executable imports."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _ProjectSourceImportScan:
    required_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...]


@dataclass(frozen=True)
class _ResolvedModule:
    files: tuple[Path, ...]
    is_package: bool


def _is_type_checking_guard(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
        and node.attr == "TYPE_CHECKING"
    )


class _ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[ast.Import | ast.ImportFrom] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.nodes.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.nodes.append(node)

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)


def _path_target(root: Path, lexical: Path) -> _ResolvedModule | None:
    try:
        lexical.resolve().relative_to(root)
    except ValueError:
        return None
    package_init = lexical / "__init__.py"
    module_file = lexical.with_suffix(".py")
    if package_init.is_file():
        target: Path | None = package_init
        is_package = True
    elif module_file.is_file():
        target = module_file
        is_package = False
    elif lexical.is_dir():
        target = None
        is_package = True
    else:
        return None
    files: list[Path] = []
    prefix = root
    for part in lexical.resolve().relative_to(root).parts[:-1]:
        prefix = prefix / part
        init_file = prefix / "__init__.py"
        if init_file.is_file():
            files.append(init_file.resolve())
    if target is not None:
        files.append(target.resolve())
    return _ResolvedModule(tuple(files), is_package)


def _module_target(root: Path, module: str) -> _ResolvedModule | None:
    parts = tuple(part for part in module.split(".") if part)
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    lexical = root.joinpath(*parts)
    package_init = lexical / "__init__.py"
    module_file = lexical.with_suffix(".py")
    if package_init.is_file():
        target: Path | None = package_init
        is_package = True
    elif module_file.is_file():
        target = module_file
        is_package = False
    elif lexical.is_dir():
        target = None  # repository namespace package; it has no source file itself
        is_package = True
    else:
        return None

    files: list[Path] = []
    prefix = root
    for part in parts[:-1]:
        prefix = prefix / part
        init_file = prefix / "__init__.py"
        if init_file.is_file():
            files.append(init_file.resolve())
    if target is not None:
        files.append(target.resolve())
    return _ResolvedModule(tuple(files), is_package)


def _module_name_for_path(root: Path, path: Path) -> tuple[str, ...]:
    relative = path.resolve().relative_to(root)
    if relative.name == "__init__.py":
        return relative.parent.parts
    return relative.with_suffix("").parts


def _absolute_from_module(root: Path, source: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    module_parts = _module_name_for_path(root, source)
    package_parts = module_parts if source.name == "__init__.py" else module_parts[:-1]
    ascend = node.level - 1
    if ascend > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - ascend]
    if node.module:
        base_parts = (*base_parts, *node.module.split("."))
    return ".".join(base_parts) if base_parts else None


def _resolved_import_files(
    root: Path,
    source: Path,
    node: ast.Import | ast.ImportFrom,
) -> tuple[Path, ...]:
    discovered: list[Path] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            resolved = _module_target(root, alias.name)
            if resolved is not None:
                discovered.extend(resolved.files)
        return tuple(discovered)

    if node.level > 0:
        base_dir = source.parent
        for _ in range(node.level - 1):
            base_dir = base_dir.parent
        lexical = base_dir.joinpath(*(node.module.split(".") if node.module else ()))
        base = _path_target(root, lexical)
        if base is not None:
            discovered.extend(base.files)
            if base.is_package:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    child = _path_target(root, lexical / alias.name)
                    if child is not None:
                        discovered.extend(child.files)
        return tuple(discovered)

    base_name = node.module
    if base_name is None:
        return ()
    base = _module_target(root, base_name)
    if base is not None:
        discovered.extend(base.files)
        if base.is_package:
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = _module_target(root, f"{base_name}.{alias.name}")
                if child is not None:
                    discovered.extend(child.files)
    return tuple(discovered)


def _is_mldb_infrastructure(relative: str) -> bool:
    return relative == "mldb_v2/__init__.py" or relative.startswith("mldb_v2/src/")


def _relative_posix(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _scan_project_source_imports(
    repository_root: str | Path,
    companion: str | Path,
    *,
    allowed_source_root: str | Path,
) -> _ProjectSourceImportScan:
    root = Path(repository_root).resolve(strict=True)
    start = Path(companion).resolve(strict=True)
    allowed_root = Path(allowed_source_root).resolve(strict=False)
    if not allowed_root.is_relative_to(root):
        raise ValueError("allowed executable source root must stay inside the repository")
    if not start.is_file() or not start.is_relative_to(root):
        raise ValueError("executable companion must be a repository file")

    required: set[str] = set()
    forbidden: set[str] = set()
    visited: set[Path] = set()
    queue: list[Path] = [start]
    while queue:
        source = queue.pop(0)
        if source in visited:
            continue
        visited.add(source)
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeError, SyntaxError) as error:
            raise ValueError("project source import graph cannot be inspected") from error
        collector = _ImportCollector(); collector.visit(tree)
        for node in collector.nodes:
            for imported in _resolved_import_files(root, source, node):
                relative = _relative_posix(root, imported)
                if _is_mldb_infrastructure(relative) or imported == start:
                    continue
                if not imported.is_relative_to(allowed_root):
                    forbidden.add(relative)
                    continue
                required.add(relative)
                if imported not in visited:
                    queue.append(imported)
    return _ProjectSourceImportScan(tuple(sorted(required)), tuple(sorted(forbidden)))
