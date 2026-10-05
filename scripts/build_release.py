"""Build and verify a curated STAC-Mem source archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    ".env.example",
    ".gitignore",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "LICENSE",
    "README.md",
    "README.zh-CN.md",
    "pyproject.toml",
    "quickstart.py",
)
PUBLIC_DIRS = (
    ".github",
    "configs",
    "docs",
    "examples",
    "scripts",
    "src",
    "tests",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
PRIVATE_FILES = {"tests/test_public_benchmark.py"}
BANNED_TERMS = (
    "ever" + "os",
    "connect_" + "ever" + "os",
    "mirror_to_" + "ever" + "os",
    "include_" + "native",
    "native_" + "episodes",
)
SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|secret|token)\s*=\s*['\"]([^'\"]+)['\"]"
)


def candidate_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES]
    for directory in PUBLIC_DIRS:
        base = ROOT / directory
        if base.is_dir():
            files.extend(path for path in base.rglob("*") if path.is_file())
    return sorted(
        {
            path.resolve()
            for path in files
            if not EXCLUDED_PARTS.intersection(path.parts)
            and path.relative_to(ROOT).as_posix() not in PRIVATE_FILES
            and path.suffix not in {".pyc", ".pyo"}
        },
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def validate(files: list[Path]) -> None:
    missing = [name for name in ROOT_FILES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required release files: {missing}")

    violations: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if relative in {".env", "config.json"}:
            violations.append(f"private configuration: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path.resolve() != Path(__file__).resolve():
            lowered = text.casefold()
            for term in BANNED_TERMS:
                if term in lowered:
                    violations.append(f"legacy identifier {term!r}: {relative}")
        for match in SECRET_PATTERN.finditer(text):
            value = match.group(1).strip()
            if len(value) >= 16 and value not in {"<redacted>"}:
                violations.append(f"possible literal secret: {relative}")
                break
    if violations:
        raise RuntimeError("Release validation failed:\n- " + "\n- ".join(violations))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    output = (args.output or ROOT / "artifacts" / f"stacmem-{version}-source.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {output}")

    files = candidate_files()
    validate(files)
    prefix = f"stacmem-{version}"
    manifest = {
        "name": project["name"],
        "version": version,
        "files": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "excluded": [
            ".env",
            ".venv",
            "artifacts",
            "data",
            "private research material",
            "runs",
            "runtime",
        ],
    }

    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{prefix}/{path.relative_to(ROOT).as_posix()}")
        archive.writestr(
            f"{prefix}/RELEASE_MANIFEST.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    with zipfile.ZipFile(output) as archive:
        for relative, expected in manifest["files"].items():
            actual = hashlib.sha256(archive.read(f"{prefix}/{relative}")).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Archive verification failed: {relative}")

    print(
        json.dumps(
            {
                "archive": str(output),
                "files": len(files),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
