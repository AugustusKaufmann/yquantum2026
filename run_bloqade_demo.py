"""
Thin entrypoint for the Bloqade-backed portfolio demo.

This wrapper exists so the repo has one obvious command to run for the
hackathon-required Bloqade workflow, while keeping the larger notebook-exported
script in its own file.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if sys.version_info < (3, 12) or sys.version_info >= (3, 14):
        raise SystemExit(
            "Bloqade in this repo is validated on Python 3.12.x. "
            f"Current interpreter: {sys.version.split()[0]}. "
            "Create a Python 3.12 virtual environment and run this command again."
        )

    script_path = Path(__file__).with_name("bloqade_qaoa_portfolio.py")
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
