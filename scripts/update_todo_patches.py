#!/usr/bin/env python3
"""Pre-commit hook: regenerate the Patches section of TODO.md.

Scans src/ for:
  - Lines containing ``# TODO:`` or ``# TODO ...`` comments
  - Functions that raise ``NotImplementedError``

Rebuilds everything between ``## Patches`` and ``## Features`` in TODO.md,
preserving the Features section and the file header verbatim.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
TODO_PATH = REPO_ROOT / "TODO.md"

# ── Pattern definitions ──────────────────────────────────────────────

# Matches lines like:  # TODO: some description
#                      # TODO some description
TODO_RE = re.compile(r"#\s*TODO[:\s]+(.+)", re.IGNORECASE)

# Matches:  raise NotImplementedError(...)
NOT_IMPL_RE = re.compile(r"raise\s+NotImplementedError\(")

# ── Scanning helpers ─────────────────────────────────────────────────


def _rel(path: Path) -> str:
    """Return a POSIX-style path relative to the repo root."""
    return path.relative_to(REPO_ROOT).as_posix()


def _github_link(path: Path, line: int, end_line: int | None = None) -> str:
    rel = _rel(path)
    if end_line and end_line != line:
        return f"[{rel}:{line}-{end_line}]({rel}#L{line}-L{end_line})"
    return f"[{rel}:{line}]({rel}#L{line})"


def _find_enclosing_function(lines: list[str], line_idx: int) -> str | None:
    """Walk backwards to find the nearest ``def`` above *line_idx*."""
    for i in range(line_idx - 1, -1, -1):
        m = re.match(r"\s*def\s+(\w+)\s*\(", lines[i])
        if m:
            return m.group(1)
    return None


def _read_docstring_summary(lines: list[str], def_idx: int) -> str | None:
    """Read the first non-empty line of the docstring after a def."""
    for i in range(def_idx + 1, min(def_idx + 8, len(lines))):
        stripped = lines[i].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Single-line docstring
            doc = stripped.strip("\"'").strip()
            if doc:
                return doc
            # Multi-line: grab the next line
            for j in range(i + 1, min(i + 4, len(lines))):
                doc = lines[j].strip().strip("\"'").strip()
                if doc:
                    return doc
            return None
    return None


# ── Data structures ──────────────────────────────────────────────────


class PatchItem:
    """A single patch entry (TODO comment or NotImplementedError stub)."""

    def __init__(
        self,
        file: Path,
        line: int,
        description: str,
        *,
        end_line: int | None = None,
        func_name: str | None = None,
    ) -> None:
        self.file = file
        self.line = line
        self.end_line = end_line
        self.description = description
        self.func_name = func_name

    @property
    def module(self) -> str:
        """Top-level src module name (e.g. 'synchronization', 'audio')."""
        rel = self.file.relative_to(SRC_DIR)
        return rel.parts[0] if len(rel.parts) > 1 else "root"

    def to_markdown(self) -> str:
        link = _github_link(self.file, self.line, self.end_line)
        title = f"`{self.func_name}()`" if self.func_name else "TODO"
        return f"- **{title}** — {self.description}\n  {link}\n"


# ── Scanner ──────────────────────────────────────────────────────────


def scan_patches() -> list[PatchItem]:
    items: list[PatchItem] = []

    for py_file in sorted(SRC_DIR.rglob("*.py")):
        lines = py_file.read_text().splitlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # ── Multi-line TODO comments ────────────────────────
            m = TODO_RE.search(line)
            if m:
                desc_parts = [m.group(1).strip().rstrip(".,")]
                start_line = i + 1  # 1-indexed
                # Collect continuation TODO lines
                j = i + 1
                while j < len(lines):
                    m2 = TODO_RE.search(lines[j])
                    if m2:
                        desc_parts.append(m2.group(1).strip().rstrip(".,"))
                        j += 1
                    else:
                        break
                end_line = j  # 1-indexed (exclusive → last matched)
                func_name = _find_enclosing_function(lines, i)
                items.append(
                    PatchItem(
                        py_file,
                        start_line,
                        ". ".join(desc_parts),
                        end_line=end_line if end_line != start_line else None,
                        func_name=func_name,
                    )
                )
                i = j
                continue

            # ── NotImplementedError stubs ───────────────────────
            if NOT_IMPL_RE.search(line):
                func_name = _find_enclosing_function(lines, i)
                if func_name:
                    # Try to extract the def line number
                    for k in range(i - 1, -1, -1):
                        if re.match(r"\s*def\s+" + re.escape(func_name), lines[k]):
                            def_line = k
                            break
                    else:
                        def_line = i
                    doc = _read_docstring_summary(lines, def_line) or ""
                    desc = f"Raises `NotImplementedError`. {doc}".strip()
                else:
                    desc = "Raises `NotImplementedError`."
                items.append(
                    PatchItem(
                        py_file,
                        i + 1,
                        desc,
                        func_name=func_name,
                    )
                )
            i += 1

    return items


# ── Markdown generation ──────────────────────────────────────────────


def group_by_module(items: list[PatchItem]) -> dict[str, list[PatchItem]]:
    groups: dict[str, list[PatchItem]] = {}
    for item in items:
        groups.setdefault(item.module, []).append(item)
    return groups


def build_patches_section(items: list[PatchItem]) -> str:
    if not items:
        return "## Patches\n\n_No TODO comments or NotImplementedError stubs found._\n"

    parts = ["## Patches\n"]
    parts.append(
        "Code-level TODOs and stubs that need attention in existing modules.\n"
    )

    for module, group in group_by_module(items).items():
        heading = module.replace("_", " ").title()
        parts.append(f"### {heading}\n")
        for item in group:
            parts.append(item.to_markdown())

    return "\n".join(parts)


# ── File rewriting ───────────────────────────────────────────────────

FEATURES_MARKER = "## Features"
PATCHES_MARKER = "## Patches"


def rewrite_todo_md(patches_md: str) -> bool:
    """Rewrite TODO.md, returning True if the content changed."""
    if not TODO_PATH.exists():
        print("TODO.md not found — skipping patch update.", file=sys.stderr)
        return False

    content = TODO_PATH.read_text()

    # Find section boundaries
    patches_start = content.find(PATCHES_MARKER)
    features_start = content.find(FEATURES_MARKER)

    if patches_start == -1 or features_start == -1:
        print(
            "TODO.md missing ## Patches or ## Features marker — skipping.",
            file=sys.stderr,
        )
        return False

    header = content[:patches_start]
    features_section = content[features_start:]

    new_content = header + patches_md + "\n---\n\n" + features_section

    if new_content == content:
        return False

    TODO_PATH.write_text(new_content)
    return True


# ── Entry point ──────────────────────────────────────────────────────


def main() -> int:
    items = scan_patches()
    patches_md = build_patches_section(items)
    changed = rewrite_todo_md(patches_md)
    if changed:
        print(f"TODO.md updated ({len(items)} patch items found).")
        # Exit 1 so pre-commit re-stages the modified file
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
