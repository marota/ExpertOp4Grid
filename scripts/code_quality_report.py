#!/usr/bin/env python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
"""Generate a code-quality report for ExpertOp4Grid.

This script aggregates a handful of static checks into a single markdown report:

* module line-count inventory + the biggest offenders (radon raw)
* cyclomatic complexity hotspots (radon cc)
* maintainability index (radon mi)
* dead-code suspects (vulture)
* docstring coverage (interrogate)
* lint findings (ruff)
* TODO / FIXME markers, split between "bare" and "issue-referenced"
* hardcoded absolute paths (``/home/<user>/...``)
* duplicate module-level definitions in ``config*.py``
* coarse type-hint coverage metric

By default it always exits 0 and just prints the report. Pass ``--strict``
to make it exit non-zero when any of the *critical* regression checks
fails (bare TODO markers, duplicate config definitions, hardcoded home
paths).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
PKG_DIR: Path = REPO_ROOT / "alphaDeesp"
TEST_DIR: Path = REPO_ROOT / "alphaDeesp" / "tests"


@dataclass
class Finding:
    """One line in the rendered markdown report."""

    key: str
    value: str
    detail: str = ""


@dataclass
class Section:
    title: str
    lines: List[str] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)

    def render(self) -> str:
        out = [f"## {self.title}", ""]
        if self.findings:
            out.append("| Metric | Value |")
            out.append("|---|---|")
            for f in self.findings:
                out.append(f"| {f.key} | {f.value} |")
            out.append("")
        out.extend(self.lines)
        out.append("")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    """Run ``cmd`` and capture stdout/stderr. Never raises."""
    if shutil.which(cmd[0]) is None:
        return 127, "", f"tool not installed: {cmd[0]}"
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _iter_py_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_loc_inventory() -> Section:
    """Module line-count inventory via radon raw (falls back to a naive count)."""
    section = Section("1. Module line-count inventory")

    code, stdout, _ = _run(["radon", "raw", "-s", "-j", str(PKG_DIR)])
    sizes: Dict[str, int] = {}
    if code == 0 and stdout.strip():
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            data = {}
        for path, metrics in data.items():
            if isinstance(metrics, dict) and "loc" in metrics:
                sizes[path] = int(metrics["loc"])
    if not sizes:
        # Fallback: plain line count.
        for p in _iter_py_files(PKG_DIR):
            sizes[str(p)] = sum(1 for _ in p.read_text(errors="ignore").splitlines())

    total = sum(sizes.values())
    section.findings.append(Finding("Total package LOC", str(total)))
    section.findings.append(Finding("Tracked modules", str(len(sizes))))

    top = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:10]
    if top:
        section.lines.append("### Top 10 modules by LOC")
        section.lines.append("")
        section.lines.append("| Module | LOC |")
        section.lines.append("|---|---|")
        for path, loc in top:
            rel = Path(path).resolve().relative_to(REPO_ROOT)
            section.lines.append(f"| `{rel}` | {loc} |")
    return section


def section_complexity() -> Section:
    """Cyclomatic complexity hotspots via radon cc."""
    section = Section("2. Complexity hotspots (radon cc)")

    code_all, stdout_all, stderr_all = _run(
        ["radon", "cc", "-s", "-a", "-j", str(PKG_DIR)]
    )
    if code_all == 127:
        section.findings.append(Finding("radon cc", "not installed"))
        return section

    total_cc = 0
    total_blocks = 0
    worst: List[Tuple[int, str]] = []
    if stdout_all.strip():
        try:
            data = json.loads(stdout_all)
        except json.JSONDecodeError:
            data = {}
        for path, blocks in data.items():
            if not isinstance(blocks, list):
                continue
            rel = Path(path).resolve().relative_to(REPO_ROOT)
            for blk in blocks:
                cc_val = int(blk.get("complexity", 0))
                total_cc += cc_val
                total_blocks += 1
                rank = blk.get("rank", "?")
                if rank and rank >= "C":
                    name = blk.get("name", "?")
                    lineno = blk.get("lineno", "?")
                    worst.append(
                        (
                            cc_val,
                            f"{rel}:{lineno} {name} - {rank} ({cc_val})",
                        )
                    )

    if total_blocks:
        avg = total_cc / total_blocks
        section.findings.append(
            Finding("Average complexity", f"{avg:.2f} (over {total_blocks} blocks)")
        )
    section.findings.append(Finding("Hotspots (grade C or worse)", str(len(worst))))

    if worst:
        worst.sort(key=lambda kv: kv[0], reverse=True)
        section.lines.append("### Top 15 complexity offenders")
        section.lines.append("")
        section.lines.append("```")
        section.lines.extend(entry for _, entry in worst[:15])
        section.lines.append("```")
    return section


def section_maintainability() -> Section:
    """Maintainability index via radon mi."""
    section = Section("3. Maintainability index (radon mi)")
    code, stdout, stderr = _run(["radon", "mi", "-s", str(PKG_DIR)])
    if code != 0 and not stdout:
        section.findings.append(Finding("radon mi", "unavailable"))
        section.lines.append(f"```\n{stderr.strip() or 'unable to run radon mi'}\n```")
        return section

    grade_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
    worst: List[str] = []
    for line in stdout.splitlines():
        m = re.search(r"- ([A-C])\s+\(([0-9.]+)\)", line)
        if not m:
            continue
        grade = m.group(1)
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        if grade != "A":
            worst.append(line.strip())

    for grade in ("A", "B", "C"):
        section.findings.append(Finding(f"Modules with grade {grade}", str(grade_counts[grade])))
    if worst:
        section.lines.append("### Modules below grade A")
        section.lines.append("")
        section.lines.append("```")
        section.lines.extend(worst[:20])
        section.lines.append("```")
    return section


def section_dead_code() -> Section:
    """Dead code suspects via vulture."""
    section = Section("4. Dead-code suspects (vulture)")
    code, stdout, stderr = _run(
        [
            "vulture",
            "--min-confidence",
            "80",
            str(PKG_DIR),
        ]
    )
    if code == 127:
        section.findings.append(Finding("vulture", "not installed"))
        return section

    findings = [ln for ln in stdout.splitlines() if ln.strip()]
    section.findings.append(Finding("Suspect symbols (≥80% confidence)", str(len(findings))))
    if findings:
        section.lines.append("```")
        section.lines.extend(findings[:40])
        section.lines.append("```")
    return section


def section_docstring_coverage() -> Section:
    """Docstring coverage via interrogate."""
    section = Section("5. Docstring coverage (interrogate)")
    code, stdout, stderr = _run(
        ["interrogate", "-v", "--fail-under", "0", str(PKG_DIR)]
    )
    if code == 127:
        section.findings.append(Finding("interrogate", "not installed"))
        return section

    pct_match = re.search(r"actual:\s+([0-9.]+)%", stdout)
    if pct_match:
        section.findings.append(Finding("Coverage", f"{pct_match.group(1)}%"))
    else:
        tail = [ln for ln in stdout.splitlines() if "%" in ln][-5:]
        if tail:
            section.lines.append("```")
            section.lines.extend(tail)
            section.lines.append("```")
    return section


def section_lint() -> Section:
    """Ruff lint findings (warning-level, not blocking)."""
    section = Section("6. Lint findings (ruff)")
    code, stdout, stderr = _run(
        [
            "ruff",
            "check",
            "--output-format",
            "concise",
            "--exit-zero",
            str(PKG_DIR),
        ]
    )
    if code == 127:
        section.findings.append(Finding("ruff", "not installed"))
        return section

    findings = [ln for ln in stdout.splitlines() if ln.strip()]
    section.findings.append(Finding("Ruff findings", str(len(findings))))
    if findings:
        section.lines.append("```")
        section.lines.extend(findings[:40])
        section.lines.append("```")
    return section


_TODO_RE = re.compile(r"\b(?:TODO|FIXME|XXX)\b")
_ISSUE_RE = re.compile(r"#\d+")


def section_todo_inventory() -> Tuple[Section, int]:
    """Inventory of TODO/FIXME markers; split bare vs. issue-referenced."""
    section = Section("7. TODO / FIXME markers")
    bare: List[str] = []
    referenced: List[str] = []
    for p in _iter_py_files(PKG_DIR):
        text = p.read_text(errors="ignore")
        for idx, line in enumerate(text.splitlines(), 1):
            if not _TODO_RE.search(line):
                continue
            location = f"{p.relative_to(REPO_ROOT)}:{idx}"
            if _ISSUE_RE.search(line):
                referenced.append(location)
            else:
                bare.append(f"{location}  {line.strip()}")

    section.findings.append(Finding("Bare TODO/FIXME markers", str(len(bare))))
    section.findings.append(Finding("Issue-referenced markers", str(len(referenced))))

    if bare:
        section.lines.append("### Bare markers (regression — every TODO should reference an issue)")
        section.lines.append("")
        section.lines.append("```")
        section.lines.extend(bare)
        section.lines.append("```")
    return section, len(bare)


_HOME_PATH_RE = re.compile(r"""["']/home/[a-zA-Z][\w.-]*""")


def section_hardcoded_paths() -> Tuple[Section, int]:
    """Hardcoded absolute home paths committed into the package."""
    section = Section("8. Hardcoded absolute paths")
    findings: List[str] = []
    for p in _iter_py_files(PKG_DIR):
        for idx, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if _HOME_PATH_RE.search(line):
                findings.append(f"{p.relative_to(REPO_ROOT)}:{idx}  {line.strip()}")

    section.findings.append(Finding("Hardcoded `/home/<user>/…` literals", str(len(findings))))
    if findings:
        section.lines.append("```")
        section.lines.extend(findings)
        section.lines.append("```")
    return section, len(findings)


def section_duplicate_config_defs() -> Tuple[Section, int]:
    """Detect duplicate module-level definitions in every config*.py."""
    section = Section("9. Duplicate config definitions")
    duplicates: Dict[str, Dict[str, int]] = {}
    for path in sorted(PKG_DIR.glob("config*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        counts: Dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        counts[target.id] = counts.get(target.id, 0) + 1
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                counts[node.target.id] = counts.get(node.target.id, 0) + 1
        dups = {k: v for k, v in counts.items() if v > 1}
        if dups:
            duplicates[str(path.relative_to(REPO_ROOT))] = dups

    total = sum(sum(v.values()) - len(v) for v in duplicates.values())
    section.findings.append(Finding("Duplicate definitions (extra assignments)", str(total)))
    if duplicates:
        for fname, dups in duplicates.items():
            section.lines.append(f"### `{fname}`")
            section.lines.append("")
            for name, count in sorted(dups.items()):
                section.lines.append(f"- `{name}` defined {count} times")
            section.lines.append("")
    return section, total


def section_type_hint_coverage() -> Section:
    """Rough type-hint coverage by parsing every function with AST."""
    section = Section("10. Type-hint coverage")
    total = 0
    typed = 0
    for p in _iter_py_files(PKG_DIR):
        try:
            tree = ast.parse(p.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total += 1
                has_return = node.returns is not None
                positional_args = node.args.args + node.args.kwonlyargs
                non_self_args = [
                    a for a in positional_args if a.arg not in ("self", "cls")
                ]
                has_args = all(
                    a.annotation is not None for a in non_self_args
                ) if non_self_args else True
                if has_return and has_args:
                    typed += 1

    pct = (100.0 * typed / total) if total else 0.0
    section.findings.append(Finding("Fully typed functions", f"{typed} / {total}"))
    section.findings.append(Finding("Coverage", f"{pct:.1f}%"))
    return section


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_report() -> Tuple[str, int, Dict[str, int]]:
    """Assemble every section and return the full markdown + strict-exit code."""
    header = [
        "# ExpertOp4Grid — code quality report",
        "",
        "Generated by `scripts/code_quality_report.py`.",
        "",
    ]

    sections: List[Section] = []
    sections.append(section_loc_inventory())
    sections.append(section_complexity())
    sections.append(section_maintainability())
    sections.append(section_dead_code())
    sections.append(section_docstring_coverage())
    sections.append(section_lint())

    todo_section, bare_todos = section_todo_inventory()
    sections.append(todo_section)

    paths_section, hardcoded_paths = section_hardcoded_paths()
    sections.append(paths_section)

    dup_section, duplicate_defs = section_duplicate_config_defs()
    sections.append(dup_section)

    sections.append(section_type_hint_coverage())

    rendered = "\n".join(header + [s.render() for s in sections])

    critical = {
        "bare_todos": bare_todos,
        "hardcoded_paths": hardcoded_paths,
        "duplicate_config_defs": duplicate_defs,
    }
    strict_exit = 1 if any(v > 0 for v in critical.values()) else 0
    return rendered, strict_exit, critical


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the report to this file instead of stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when critical regressions are detected.",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Also append the report to $GITHUB_STEP_SUMMARY when set.",
    )
    args = parser.parse_args(argv)

    report, strict_code, critical = build_report()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if args.github_summary and summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(report)
            fh.write("\n")

    if args.strict and strict_code:
        print(
            "\nstrict mode: critical regressions detected: "
            + ", ".join(f"{k}={v}" for k, v in critical.items() if v),
            file=sys.stderr,
        )
        return strict_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
