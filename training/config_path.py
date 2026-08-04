"""
training/config_path.py
=======================
Shared configuration-path resolver for Jupiter Shot runners.

All three runners (dense, MoE, resume) and the pipeline validation steps
must use ``resolve_config_path()`` instead of constructing paths with
``f"{config_name}.yaml"`` directly.

Resolution contract (five supported input forms)
-------------------------------------------------
A. Absolute path with .yaml or .yml suffix
   → Use exactly as supplied. Never append another suffix.

B. Repository-relative path with .yaml or .yml suffix
   → Resolve relative to repository root. Never append another suffix.

C. Bare name without any suffix (legacy form)
   → Resolve under ``training/configs/`` and append ``.yaml`` exactly once.

D. Path without suffix (has directory components but no extension)
   → Resolve the path first, then append ``.yaml`` exactly once.

E. Missing file after resolution
   → Return a ``ConfigResolution`` with ``exists=False``.
   → Callers must raise ``SystemExit(3)`` (EXECUTION_ERROR) and print the
     full diagnostic block (see ``format_missing_error``).

Guarantees
----------
* ``.yaml`` is never appended twice.
* ``.yml`` is never changed to ``.yml.yaml``.
* ``Path.stem`` is never used as the sole repair (it discards directory info).
* No silent fallback to a similarly-named file.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Repository root — two levels up from this file (training/config_path.py)
# ---------------------------------------------------------------------------
_REPO_ROOT: pathlib.Path = pathlib.Path(__file__).parent.parent.resolve()
_CONFIGS_DIR: pathlib.Path = _REPO_ROOT / "training" / "configs"


@dataclass
class ConfigResolution:
    """Structured result returned by ``resolve_config_path``."""

    original_input: str
    """The raw string passed to the resolver."""

    resolved_path: pathlib.Path
    """The fully-resolved absolute path (may not exist)."""

    suffix_added: bool
    """True if ``.yaml`` was appended during resolution."""

    exists: bool
    """True if ``resolved_path`` exists on disk."""

    repository_root: pathlib.Path
    """Absolute path to the repository root used during resolution."""

    resolution_rule: str = field(default="")
    """Which rule (A/B/C/D) was applied."""


def resolve_config_path(
    config_input: str,
    repo_root: Optional[pathlib.Path] = None,
) -> ConfigResolution:
    """Resolve a ``--config`` argument to an absolute YAML path.

    Parameters
    ----------
    config_input:
        The raw value of ``--config`` from argparse.  May be an absolute
        path, a repository-relative path, or a bare config name.
    repo_root:
        Override the repository root (used in tests).  Defaults to the
        repository root detected from this file's location.

    Returns
    -------
    ConfigResolution
        Structured resolution result.  Callers must check ``.exists``
        before opening the file.
    """
    root: pathlib.Path = (repo_root or _REPO_ROOT).resolve()
    configs_dir: pathlib.Path = root / "training" / "configs"

    p = pathlib.Path(config_input)

    # ------------------------------------------------------------------
    # Rule A — absolute path with recognised suffix
    # ------------------------------------------------------------------
    if p.is_absolute() and p.suffix in (".yaml", ".yml"):
        resolved = p.resolve()
        return ConfigResolution(
            original_input=config_input,
            resolved_path=resolved,
            suffix_added=False,
            exists=resolved.exists(),
            repository_root=root,
            resolution_rule="A",
        )

    # ------------------------------------------------------------------
    # Rule B — relative path with recognised suffix
    # ------------------------------------------------------------------
    if not p.is_absolute() and p.suffix in (".yaml", ".yml"):
        resolved = (root / p).resolve()
        return ConfigResolution(
            original_input=config_input,
            resolved_path=resolved,
            suffix_added=False,
            exists=resolved.exists(),
            repository_root=root,
            resolution_rule="B",
        )

    # ------------------------------------------------------------------
    # Rule C — bare name without any suffix (no path separators or only
    #           a simple stem, e.g. "laptop_dense_run7")
    # ------------------------------------------------------------------
    if p.suffix == "" and len(p.parts) == 1:
        resolved = (configs_dir / f"{p.name}.yaml").resolve()
        return ConfigResolution(
            original_input=config_input,
            resolved_path=resolved,
            suffix_added=True,
            exists=resolved.exists(),
            repository_root=root,
            resolution_rule="C",
        )

    # ------------------------------------------------------------------
    # Rule D — path without suffix (has directory components but no ext)
    # ------------------------------------------------------------------
    if p.suffix == "":
        # Absolute or relative — resolve first, then append .yaml once
        base = p if p.is_absolute() else (root / p)
        resolved = pathlib.Path(str(base.resolve()) + ".yaml")
        return ConfigResolution(
            original_input=config_input,
            resolved_path=resolved,
            suffix_added=True,
            exists=resolved.exists(),
            repository_root=root,
            resolution_rule="D",
        )

    # ------------------------------------------------------------------
    # Fallback — unrecognised suffix (e.g. .json, .toml)
    # Treat as absolute/relative path and do not modify the suffix.
    # ------------------------------------------------------------------
    base = p if p.is_absolute() else (root / p)
    resolved = base.resolve()
    return ConfigResolution(
        original_input=config_input,
        resolved_path=resolved,
        suffix_added=False,
        exists=resolved.exists(),
        repository_root=root,
        resolution_rule="A" if p.is_absolute() else "B",
    )


def format_missing_error(res: ConfigResolution) -> str:
    """Return the mandatory EXECUTION_ERROR diagnostic block for a missing config.

    The block includes:
    1. Original input
    2. Resolved path
    3. Current working directory
    4. Repository root
    5. Whether a suffix was added
    6. Suggested valid alternatives (existing .yaml files in training/configs/)
    """
    import os

    configs_dir = res.repository_root / "training" / "configs"
    try:
        alternatives = sorted(
            p.name for p in configs_dir.glob("*.yaml") if p.is_file()
        )
    except OSError:
        alternatives = []

    lines = [
        "EXECUTION_ERROR: Configuration file not found.",
        f"  original_input   : {res.original_input!r}",
        f"  resolved_path    : {res.resolved_path}",
        f"  cwd              : {os.getcwd()}",
        f"  repository_root  : {res.repository_root}",
        f"  suffix_added     : {res.suffix_added}",
        f"  resolution_rule  : {res.resolution_rule}",
    ]
    if alternatives:
        lines.append("  valid alternatives (training/configs/):")
        for alt in alternatives[:10]:
            lines.append(f"    - {alt}")
        if len(alternatives) > 10:
            lines.append(f"    ... and {len(alternatives) - 10} more")
    else:
        lines.append("  valid alternatives : (none found in training/configs/)")
    return "\n".join(lines)
