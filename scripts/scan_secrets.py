"""Fail CI when tracked files contain common credential material."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_NAMES = {".env", "credentials.json", "secrets.json"}
_FORBIDDEN_SUFFIXES = {".key", ".p12", ".pem"}
_TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?im)^\s*(?:ZHIPU_API_KEY|API_KEY|SECRET_KEY|ACCESS_TOKEN)\s*=\s*"
        r"([A-Za-z0-9._~+/=-]{20,})\s*$"
    ),
)
_PLACEHOLDER_MARKERS = ("your_", "replace", "example", "placeholder", "changeme")


def tracked_files(root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item)


def scan(paths: tuple[Path, ...], root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    findings: list[str] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.name.lower() in _FORBIDDEN_NAMES or path.suffix.lower() in _FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden tracked credential file: {relative}")
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(content):
                candidate = match.group(1) if match.lastindex else match.group(0)
                if any(marker in candidate.casefold() for marker in _PLACEHOLDER_MARKERS):
                    continue
                line = content.count("\n", 0, match.start()) + 1
                findings.append(f"possible secret: {relative}:{line}")
    return tuple(sorted(set(findings)))


def main() -> int:
    findings = scan(tracked_files())
    if findings:
        print("\n".join(findings), file=sys.stderr)
        return 1
    print("tracked secret scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
