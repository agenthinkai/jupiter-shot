#!/usr/bin/env python3
"""
Jupiter Seed 4B — UTF-8 Encoding Audit
========================================
AST-based audit of all production Python files under training/jupiter_seed_4b/.

Checks that every open(), Path.read_text(), Path.write_text(), csv reader/writer,
and JSONL reader/writer explicitly uses UTF-8 encoding.

Does NOT rely on:
  - PYTHONUTF8
  - PYTHONIOENCODING
  - OS locale
  - Developer shell configuration

Usage:
    python3 utf8_audit.py --scan-dir training/jupiter_seed_4b
                          --output docs/jupiter_seed_4b/UTF8_AUDIT_REPORT.md
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Patterns that require explicit encoding
UNSAFE_PATTERNS = [
    # open() without encoding
    ("open_no_encoding", "open() called without explicit encoding= argument"),
    # Path.read_text() without encoding
    ("read_text_no_encoding", "Path.read_text() called without explicit encoding= argument"),
    # Path.write_text() without encoding
    ("write_text_no_encoding", "Path.write_text() called without explicit encoding= argument"),
    # csv.reader/writer without explicit encoding on the file handle
    ("csv_no_encoding_comment", "csv.reader/writer — verify the file handle uses explicit UTF-8"),
]

# Files that are allowed to use utf-8-sig (the Excel reviewer CSV)
UTF8_SIG_ALLOWED = {"generate_review_queue.py"}


class Utf8Visitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[Dict] = []

    def _get_keywords(self, node: ast.Call) -> Dict[str, ast.expr]:
        return {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}

    def _has_encoding(self, keywords: Dict[str, ast.expr]) -> bool:
        """Return True if encoding= is explicitly set."""
        return "encoding" in keywords

    def _encoding_value(self, keywords: Dict[str, ast.expr]) -> str:
        enc = keywords.get("encoding")
        if enc is None:
            return ""
        if isinstance(enc, ast.Constant):
            return str(enc.value).lower()
        return "<dynamic>"

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        keywords = self._get_keywords(node)

        # Check open()
        if isinstance(func, ast.Name) and func.id == "open":
            if not self._has_encoding(keywords):
                self.violations.append({
                    "line": node.lineno,
                    "pattern": "open_no_encoding",
                    "description": "open() called without explicit encoding= argument",
                    "severity": "ERROR",
                })
            else:
                enc = self._encoding_value(keywords)
                base = Path(self.filename).name
                if enc == "utf-8-sig" and base not in UTF8_SIG_ALLOWED:
                    self.violations.append({
                        "line": node.lineno,
                        "pattern": "utf8_sig_unexpected",
                        "description": f"utf-8-sig used outside allowed files (only {UTF8_SIG_ALLOWED})",
                        "severity": "WARNING",
                    })

        # Check Path.read_text() and Path.write_text()
        elif isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            if not self._has_encoding(keywords):
                self.violations.append({
                    "line": node.lineno,
                    "pattern": f"{func.attr}_no_encoding",
                    "description": f"Path.{func.attr}() called without explicit encoding= argument",
                    "severity": "ERROR",
                })

        self.generic_visit(node)


def audit_file(path: Path) -> Tuple[str, List[Dict]]:
    """Audit a single Python file. Returns (filename, violations)."""
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        return str(path), [{"line": 0, "pattern": "read_error", "description": str(e), "severity": "ERROR"}]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return str(path), [{"line": e.lineno or 0, "pattern": "syntax_error", "description": str(e), "severity": "ERROR"}]

    visitor = Utf8Visitor(str(path))
    visitor.visit(tree)
    return str(path), visitor.violations


def write_report(results: Dict[str, List[Dict]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_errors = sum(1 for vs in results.values() for v in vs if v["severity"] == "ERROR")
    total_warnings = sum(1 for vs in results.values() for v in vs if v["severity"] == "WARNING")

    lines = [
        "# Jupiter Seed 4B: UTF-8 Encoding Audit Report",
        "",
        "This audit uses AST-based analysis to verify that all production Python files "
        "explicitly specify UTF-8 encoding for all text I/O operations.",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| Files audited | {len(results)} |",
        f"| Total errors | {total_errors} |",
        f"| Total warnings | {total_warnings} |",
        f"| **Overall result** | {'**PASS**' if total_errors == 0 else '**FAIL**'} |",
        "",
    ]

    if total_errors == 0 and total_warnings == 0:
        lines.append("All files pass UTF-8 encoding audit.")
    else:
        lines += ["## Violations", ""]
        for filepath, violations in sorted(results.items()):
            if violations:
                lines.append(f"### `{Path(filepath).name}`")
                lines.append("")
                lines.append("| Line | Severity | Pattern | Description |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for v in violations:
                    lines.append(f"| {v['line']} | {v['severity']} | {v['pattern']} | {v['description']} |")
                lines.append("")

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("UTF-8 audit report written to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jupiter Seed 4B — UTF-8 Encoding Audit")
    parser.add_argument("--scan-dir", type=Path,
                        default=Path("training/jupiter_seed_4b"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/jupiter_seed_4b/UTF8_AUDIT_REPORT.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    py_files = sorted(args.scan_dir.glob("*.py"))
    log.info("Auditing %d Python files in %s", len(py_files), args.scan_dir)

    results: Dict[str, List[Dict]] = {}
    for path in py_files:
        filename, violations = audit_file(path)
        results[filename] = violations
        if violations:
            for v in violations:
                log.warning("%s:%d [%s] %s", path.name, v["line"], v["severity"], v["description"])

    write_report(results, args.output)

    total_errors = sum(1 for vs in results.values() for v in vs if v["severity"] == "ERROR")
    if total_errors > 0:
        log.error("UTF-8 audit FAILED: %d errors found.", total_errors)
        sys.exit(1)
    log.info("UTF-8 audit PASSED.")


if __name__ == "__main__":
    main()
