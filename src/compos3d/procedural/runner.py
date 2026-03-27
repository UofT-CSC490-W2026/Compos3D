"""
Subprocess runner for Blender-side procedural scripts.

Scripts in procedural/scripts/ use bpy + infinigen and must be executed with
the project's Python environment (not inside the Blender GUI).  bpy 4.2.0 is
pip-installed in .venv, so all Blender scripting APIs are available from the
venv interpreter directly.

The runner sets PYTHONPATH so that both `llm_doc` (in procedural/llm_doc/)
and the `infinigen` package (from the git submodule at infinigen/) are
importable in every subprocess, with no pip-install required.

For generate_room.py: gin config files are resolved relative to a working
directory that contains `infinigen_examples/configs_indoor/` and
`infinigen_examples/configs_nature/`.  We locate the best available root at
import time, preferring the project's `infinigen/` submodule.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Compos3D project root  (src/compos3d/procedural/runner.py → up 4 levels)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# procedural/ directory — must be on PYTHONPATH for `llm_doc`
PROCEDURAL_DIR = PROJECT_ROOT / "procedural"

SCRIPTS_DIR = PROCEDURAL_DIR / "scripts"

# The infinigen git submodule root.  Adding this to PYTHONPATH makes
# `import infinigen` and `import infinigen_examples` work in subprocesses
# without any pip install.
INFINIGEN_SUBMODULE = PROJECT_ROOT / "infinigen"


def _find_gin_config_root() -> Path:
    """
    Return the directory that should be used as cwd when running generate_room.py.
    It must contain infinigen_examples/configs_indoor/singleroom.gin (and base.gin etc.).

    Search order:
      1. PROJECT_ROOT/infinigen/   (the Princeton git submodule)
      2. Any sibling directory that contains the gin config structure
    """
    candidates = [
        PROJECT_ROOT / "infinigen",
    ]
    for sibling in PROJECT_ROOT.parent.iterdir():
        if (
            sibling.is_dir()
            and (
                sibling / "infinigen_examples" / "configs_indoor" / "singleroom.gin"
            ).exists()
        ):
            candidates.append(sibling)

    for candidate in candidates:
        if (
            candidate / "infinigen_examples" / "configs_indoor" / "singleroom.gin"
        ).exists():
            return candidate

    raise FileNotFoundError(
        "Could not find infinigen_examples/configs_indoor/singleroom.gin in any "
        "candidate directory.  Expected it at: " + ", ".join(str(c) for c in candidates)
    )


# Resolve once at import time; cached for the process lifetime.
try:
    GIN_CONFIG_ROOT: Path = _find_gin_config_root()
except FileNotFoundError:
    GIN_CONFIG_ROOT = PROJECT_ROOT / "infinigen"  # best-effort default


def _build_env(extra_pythonpath: list[Path] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # PROCEDURAL_DIR → llm_doc importable
    # PROJECT_ROOT/src → local compos3d package importable in subprocesses
    # INFINIGEN_SUBMODULE → `import infinigen` / `import infinigen_examples`
    dirs = [str(PROCEDURAL_DIR), str(PROJECT_ROOT / "src"), str(INFINIGEN_SUBMODULE)]
    for p in extra_pythonpath or []:
        dirs.append(str(p))
    existing = env.get("PYTHONPATH", "")
    if existing:
        dirs.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(dirs)
    return env


def run_script(
    script: Path,
    args: list[str],
    *,
    extra_pythonpath: list[Path] | None = None,
    capture_output: bool = False,
    cwd: Path | None = None,
) -> dict:
    """
    Run a procedural script with the current Python interpreter.

    cwd: working directory for the subprocess.
         Use GIN_CONFIG_ROOT for scripts that rely on relative gin config paths.

    Returns a dict with exit_code, elapsed_seconds, cmd (and stdout/stderr if captured).
    Raises RuntimeError if exit_code != 0.
    """
    env = _build_env(extra_pythonpath)
    cmd = [sys.executable, str(script)] + [str(a) for a in args]
    t0 = time.time()

    run_kwargs: dict = {"env": env}
    if cwd is not None:
        run_kwargs["cwd"] = str(cwd)

    if capture_output:
        completed = subprocess.run(cmd, capture_output=True, text=True, **run_kwargs)
        result = {
            "exit_code": completed.returncode,
            "elapsed_seconds": round(time.time() - t0, 2),
            "cmd": " ".join(cmd),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    else:
        completed = subprocess.run(cmd, **run_kwargs)
        result = {
            "exit_code": completed.returncode,
            "elapsed_seconds": round(time.time() - t0, 2),
            "cmd": " ".join(cmd),
        }

    if result["exit_code"] != 0:
        stderr_snippet = result.get("stderr", "")[-500:] if result.get("stderr") else ""
        raise RuntimeError(
            f"Procedural script failed (exit {result['exit_code']}): {script.name}\n{stderr_snippet}"
        )
    return result
