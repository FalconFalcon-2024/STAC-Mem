"""Create an isolated environment, install STAC-Mem, and run the local demo."""

from __future__ import annotations

import argparse
import os
import subprocess
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--dev", action="store_true", help="Install test and lint tools")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    environment = args.venv.expanduser().resolve()
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print(f"Creating virtual environment: {environment}", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    if not args.skip_install:
        target = ".[dev]" if args.dev else "."
        run(str(python), "-m", "pip", "install", "-e", target)

    command = [str(python), "-m", "stacmem.standalone_demo"]
    if args.output:
        command.extend(["--output", str(args.output.expanduser().resolve())])
    run(*command)
    print("\nSTAC-Mem is ready. The demo passed all lifecycle and state checks.")
    print(f"Python: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
