# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Core utilities for BailingMoeV3 monkey patches.

Provides helpers for runtime source-level patching of vllm / vllm-ascend
modules via ``inspect.getsource`` + textual replacement + ``exec``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

_PATCH_DIR = Path(__file__).resolve().parent.parent  # npu_patch/
_NEW_FILES_DIR = _PATCH_DIR / "Incremental_adapter_files"


def _import(dotted_path: str):
    """Import and return a module by its dotted path."""
    return importlib.import_module(dotted_path)


def _get_src(obj) -> str:
    """Return dedented source code of *obj*."""
    return textwrap.dedent(inspect.getsource(obj))


def _apply_replacements(
    src: str,
    replacements: list[tuple[str, str]],
    label: str,
    *,
    strict: bool = False,
) -> str:
    """Apply *(old, new)* textual replacements to *src*.

    Each replacement is applied at most once.  If *new* is already present
    the replacement is skipped (idempotency).  If *old* is not found and
    *strict* is False a warning is printed; if *strict* is True a
    ``RuntimeError`` is raised so the caller can abort (matching the
    fail-fast behaviour of ``git apply``).
    """
    for old, new in replacements:
        if new and new in src:
            continue
        if old not in src:
            msg = f"{label}: target not found: {old.strip()[:100]!r}"
            if strict:
                raise RuntimeError(msg)
            print(f"[WARNING] {msg}")
            continue
        src = src.replace(old, new, 1)
    return src


def _exec_in_module(src: str, module) -> dict:
    """Exec *src* in *module*'s namespace and return the namespace dict."""
    ns: dict = module.__dict__
    exec(compile(src, f"<patch:{module.__name__}>", "exec"), ns)
    return ns


def patch_function(
    module_path: str,
    func_name: str,
    replacements: list[tuple[str, str]],
) -> None:
    """Patch a module-level function by textual replacement.

    *replacements* is a list of ``(old_str, new_str)`` pairs applied to the
    function source.  The patched function replaces the original.
    """
    mod = _import(module_path)
    func = getattr(mod, func_name)
    src = _get_src(func)
    src = _apply_replacements(src, replacements, f"{module_path}.{func_name}")
    ns = _exec_in_module(src, mod)
    setattr(mod, func_name, ns[func_name])


def patch_method(
    module_path: str,
    cls_name: str,
    method_name: str,
    replacements: list[tuple[str, str]],
) -> None:
    """Patch a class method by textual replacement."""
    mod = _import(module_path)
    cls = getattr(mod, cls_name)
    method = getattr(cls, method_name)
    # Unwrap classmethod/staticmethod for source extraction
    raw = method
    is_classmethod = isinstance(method, classmethod)
    is_staticmethod = isinstance(method, staticmethod)
    if is_classmethod or is_staticmethod:
        raw = method.__func__
    src = _get_src(raw)
    src = _apply_replacements(
        src, replacements, f"{module_path}.{cls_name}.{method_name}"
    )
    ns = _exec_in_module(src, mod)
    new_obj = ns[method_name]
    if is_classmethod:
        new_obj = classmethod(new_obj)
    elif is_staticmethod:
        new_obj = staticmethod(new_obj)
    setattr(cls, method_name, new_obj)


def patch_method_by_content(
    module_path: str,
    cls_name: str,
    content_match: str,
    replacements: list[tuple[str, str]],
) -> None:
    """Find a method in *cls_name* containing *content_match* and patch it.

    Useful when the exact method name is unknown but the method body
    contains a recognisable string (e.g. a function call).
    """
    mod = _import(module_path)
    cls = getattr(mod, cls_name)
    for name, raw in vars(cls).items():
        if not callable(raw):
            continue
        is_cm = isinstance(raw, classmethod)
        is_sm = isinstance(raw, staticmethod)
        if is_cm or is_sm:
            raw = raw.__func__
        try:
            src = textwrap.dedent(inspect.getsource(raw))
        except (TypeError, OSError):
            continue
        if content_match not in src:
            continue
        src = _apply_replacements(src, replacements, f"{cls_name}.{name}")
        ns = _exec_in_module(src, mod)
        new_obj = ns[name]
        if is_cm:
            new_obj = classmethod(new_obj)
        elif is_sm:
            new_obj = staticmethod(new_obj)
        setattr(cls, name, new_obj)
        return
    print(f"[WARNING] {cls_name}: no method contains {content_match!r}")


def patch_property(
    module_path: str,
    cls_name: str,
    prop_name: str,
    new_fget=None,
    new_fset=None,
    new_fdel=None,
) -> None:
    """Replace a property on a class."""
    mod = _import(module_path)
    cls = getattr(mod, cls_name)
    old_prop = getattr(cls, prop_name)
    if not isinstance(old_prop, property):
        raise TypeError(f"{cls_name}.{prop_name} is not a property")
    fget = new_fget or old_prop.fget
    fset = new_fset or old_prop.fset
    fdel = new_fdel or old_prop.fdel
    setattr(cls, prop_name, property(fget, fset, fdel, old_prop.__doc__))


def add_module_attr(module_path: str, name: str, value: Any) -> None:
    """Add *name* = *value* to a module's namespace (skip if exists)."""
    mod = _import(module_path)
    if hasattr(mod, name):
        return
    setattr(mod, name, value)


def add_class_attr(module_path: str, cls_name: str, name: str, value: Any) -> None:
    """Add *name* = *value* to a class (skip if exists)."""
    mod = _import(module_path)
    cls = getattr(mod, cls_name)
    if hasattr(cls, name):
        return
    setattr(cls, name, value)


def add_dataclass_field(
    module_path: str,
    cls_name: str,
    field_name: str,
    type_hint: type = type(None),
    default: Any = None,
) -> None:
    """Add a field to a dataclass at runtime.

    Updates ``__annotations__`` and sets the default value on the class.
    """
    mod = _import(module_path)
    cls = getattr(mod, cls_name)
    if field_name in getattr(cls, "__annotations__", {}):
        return
    if not hasattr(cls, "__annotations__"):
        cls.__annotations__ = {}
    cls.__annotations__[field_name] = type_hint
    setattr(cls, field_name, default)


def replace_function_direct(module_path: str, func_name: str, new_func) -> None:
    """Directly replace a module-level function with *new_func*."""
    mod = _import(module_path)
    setattr(mod, func_name, new_func)


def replace_method_direct(
    module_path: str, cls_name: str, method_name: str, new_method
) -> None:
    """Directly replace a class method with *new_method*."""
    mod = _import(module_path)
    cls = getattr(mod, cls_name)
    setattr(cls, method_name, new_method)


def replace_classmethod_direct(
    module_path: str, cls_name: str, method_name: str, new_func
) -> None:
    """Directly replace a classmethod with *new_func* wrapped in classmethod."""
    mod = _import(module_path)
    cls = getattr(mod, cls_name)
    setattr(cls, method_name, classmethod(new_func))


def patch_source_file(
    module_path: str,
    replacements: list[tuple[str, str]],
    label: str | None = None,
) -> None:
    """Patch a module's source file in-place via textual replacement.

    Reads the installed ``.py`` source, applies *(old, new)* replacements,
    and writes it back.  Used for triton kernels and patch modules whose
    changes cannot be applied at runtime via ``inspect.getsource``.

    Uses ``importlib.util.find_spec`` to locate the file *without importing*
    the module (triton kernels compile at import time, patch modules apply
    patches at import time).
    """
    try:
        spec = importlib.util.find_spec(module_path)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is None or spec.origin is None:
        print(f"[WARNING] {module_path}: module not found, skip source patch")
        return
    file_path = Path(spec.origin)
    if not file_path.exists():
        print(f"[WARNING] {module_path}: file not found: {file_path}")
        return
    with open(file_path, "r") as f:
        src = f.read()
    tag = label or f"source:{module_path}"
    src = _apply_replacements(src, replacements, tag)
    with open(file_path, "w") as f:
        f.write(src)
    print(f"[INFO] Source patched: {file_path}")


def patch_triton_source(
    module_path: str,
    replacements: list[tuple[str, str]],
) -> None:
    """Patch a triton kernel source file in-place (alias for patch_source_file)."""
    patch_source_file(module_path, replacements, label=f"triton:{module_path}")


def apply_diff_file(file_path: str | Path, diff_path: str | Path) -> bool:
    """Apply a unified diff file to a source file.

    Parses the diff, extracts (old_block, new_block) pairs from each hunk,
    and applies them as textual replacements.  Returns True on success.
    """
    file_path = Path(file_path)
    diff_path = Path(diff_path)
    if not file_path.exists():
        print(f"[WARNING] Target file not found: {file_path}")
        return False
    if not diff_path.exists():
        print(f"[WARNING] Diff file not found: {diff_path}")
        return False

    with open(file_path, "r") as f:
        src = f.read()
    with open(diff_path, "r") as f:
        diff_content = f.read()

    replacements = _parse_unified_diff(diff_content)
    if not replacements:
        print(f"[WARNING] No hunks parsed from {diff_path}")
        return False

    src = _apply_replacements(src, replacements, f"diff:{file_path.name}", strict=True)
    with open(file_path, "w") as f:
        f.write(src)
    print(f"[INFO] Diff applied: {file_path} ({len(replacements)} hunks)")
    return True


def _get_target_path_from_diff(diff_path: str | Path) -> str | None:
    """Extract the target file path from a diff's ``+++ b/`` line."""
    with open(diff_path, "r") as f:
        for line in f:
            if line.startswith("+++ b/"):
                return line[6:].strip()
            if line.startswith("+++ "):
                return line[4:].strip()
    return None


def _resolve_installed_file(target_rel: str) -> Path | None:
    """Locate an installed source file from its repo-relative path.

    *target_rel* is e.g. ``vllm/model_executor/layers/kda.py`` or
    ``vllm_ascend/ops/mla.py``.  The top-level package (first path segment)
    is located with :func:`importlib.util.find_spec` *without importing it*
    (top-level packages are not imported by ``find_spec``), and the remaining
    path is joined relative to the package directory.  This avoids importing
    heavy parent packages such as ``vllm`` during patch application.
    """
    top_pkg = target_rel.split("/", 1)[0]
    try:
        spec = importlib.util.find_spec(top_pkg)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    pkg_dir = Path(spec.origin).parent
    rel_within = target_rel[len(top_pkg) + 1:]
    target = pkg_dir / rel_within
    return target if target.exists() else None


def apply_diffs_from_dir(diffs_dir: str | Path, label: str | None = None) -> int:
    """Apply every ``*.diff`` file in *diffs_dir* to its installed target.

    Each diff's target path is read from its ``+++ b/`` header and resolved
    to the installed file location via :func:`_resolve_installed_file` (no
    import of the target or its submodules).  Returns the number of diffs
    successfully applied.
    """
    diffs_dir = Path(diffs_dir)
    tag = label or diffs_dir.name
    if not diffs_dir.exists():
        print(f"[WARNING] Diffs directory not found: {diffs_dir}")
        return 0

    applied = skipped = failed = 0
    for diff_file in sorted(diffs_dir.glob("*.diff")):
        target_rel = _get_target_path_from_diff(diff_file)
        if target_rel is None:
            print(f"[WARNING] {tag}: cannot parse target from {diff_file.name}")
            failed += 1
            continue

        target_file = _resolve_installed_file(target_rel)
        if target_file is None:
            print(f"[SKIP] {tag}: not installed: {target_rel} ({diff_file.name})")
            skipped += 1
            continue

        try:
            if apply_diff_file(target_file, diff_file):
                applied += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] {tag}: {diff_file.name} -> {target_file}: {e}")
            failed += 1

    print(
        f"[INFO] {tag}: {applied} applied, {skipped} skipped, {failed} failed"
    )
    if failed:
        raise RuntimeError(f"{tag}: {failed} diff(s) failed to apply")
    return applied


def _parse_unified_diff(diff_content: str) -> list[tuple[str, str]]:
    """Parse a unified diff into (old_block, new_block) replacement pairs.

    Each hunk produces one replacement pair.  Context lines, removed lines
    and added lines are concatenated to form the old/new blocks.
    """
    lines = diff_content.split("\n")
    i = 0
    replacements: list[tuple[str, str]] = []

    # Skip header lines (diff --git, index, ---, +++)
    while i < len(lines) and not lines[i].startswith("@@"):
        i += 1

    while i < len(lines):
        if not lines[i].startswith("@@"):
            i += 1
            continue

        # Skip hunk header
        i += 1
        old_lines: list[str] = []
        new_lines: list[str] = []

        while i < len(lines) and not lines[i].startswith("@@"):
            line = lines[i]
            if line.startswith("\\"):
                # "\ No newline at end of file" - skip
                i += 1
                continue
            elif line.startswith(" "):
                old_lines.append(line[1:])
                new_lines.append(line[1:])
            elif line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            elif line == "":
                # Blank line in diff - treat as context (common in some diffs)
                old_lines.append("")
                new_lines.append("")
            else:
                # Unexpected line - could be end of diff
                break
            i += 1

        while old_lines and old_lines[-1] == "":
            old_lines.pop()
        while new_lines and new_lines[-1] == "":
            new_lines.pop()
        old_block = "\n".join(old_lines)
        new_block = "\n".join(new_lines)
        if old_block != new_block:
            replacements.append((old_block, new_block))

    return replacements


def copy_new_file(src_rel_path: str, target_pkg: str) -> Path:
    """Copy a new file from ``Incremental_adapter_files/`` to the installed package location.

    *src_rel_path* is relative to ``Incremental_adapter_files/`` (e.g.
    ``vllm/model_executor/models/bailing_moe_v3.py``).
    *target_pkg* is the top-level package name (``vllm`` or ``vllm_ascend``).
    """
    src_file = _NEW_FILES_DIR / src_rel_path
    if not src_file.exists():
        print(f"[WARNING] New file not found: {src_file}")
        return src_file
    # Locate the installed package directory
    spec = importlib.util.find_spec(target_pkg)
    if spec is None or spec.origin is None:
        print(f"[WARNING] Package {target_pkg} not found, cannot copy {src_rel_path}")
        return src_file
    pkg_dir = Path(spec.origin).parent
    # Compute the destination path inside the package
    # src_rel_path starts with target_pkg/, strip it
    rel_within_pkg = src_rel_path
    if rel_within_pkg.startswith(target_pkg + "/"):
        rel_within_pkg = rel_within_pkg[len(target_pkg) + 1:]
    dst_file = pkg_dir / rel_within_pkg
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    print(f"[INFO] Copied {src_rel_path} -> {dst_file}")
    return dst_file


def is_patched(module_path: str, marker: str) -> bool:
    """Check if a module has been patched by looking for *marker* attribute."""
    try:
        mod = _import(module_path)
        return getattr(mod, marker, False) is True
    except ImportError:
        return False


def mark_patched(module_path: str, marker: str) -> None:
    """Mark a module as patched by setting *marker* = True."""
    try:
        mod = _import(module_path)
        setattr(mod, marker, True)
    except ImportError:
        pass
