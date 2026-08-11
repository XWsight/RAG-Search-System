"""Run pip-audit with a strict, expiring vulnerability exception policy."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
DEFAULT_POLICY = PROJECT_ROOT / "security" / "dependency-exceptions.json"
_MAX_POLICY_BYTES = 64 * 1024
_MAX_EXCEPTIONS = 16
_VULNERABILITY_ID = re.compile(r"(?:PYSEC-\d{4}-\d+|CVE-\d{4}-\d+|GHSA-[a-z0-9-]+)")
_PACKAGE_NAME = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_VERSION = re.compile(r"[A-Za-z0-9]+(?:[A-Za-z0-9._+-]*[A-Za-z0-9])?")
_ROOT_KEYS = frozenset({"schema_version", "exceptions"})
_ENTRY_KEYS = frozenset(
    {
        "vulnerability_id",
        "aliases",
        "package",
        "pinned_version",
        "advisory",
        "owner",
        "reviewed_on",
        "expires_on",
        "rationale",
        "compensating_controls",
    }
)


class AuditPolicyError(ValueError):
    """The checked-in exception policy is invalid or expired."""


@dataclass(frozen=True, slots=True)
class DependencyException:
    vulnerability_id: str
    aliases: tuple[str, ...]
    package: str
    pinned_version: str
    advisory: str
    owner: str
    reviewed_on: date
    expires_on: date
    rationale: str
    compensating_controls: tuple[str, ...]

    @property
    def identifiers(self) -> frozenset[str]:
        return frozenset((self.vulnerability_id, *self.aliases))


@dataclass(frozen=True, slots=True)
class AuditFinding:
    package: str
    version: str
    vulnerability_id: str
    aliases: tuple[str, ...]
    fix_versions: tuple[str, ...]

    @property
    def identifiers(self) -> frozenset[str]:
        return frozenset((self.vulnerability_id, *self.aliases))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditPolicyError("dependency exception policy contains a duplicate key")
        result[key] = value
    return result


def _read_policy(path: Path) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise AuditPolicyError("dependency exception policy cannot be read") from error
    if not payload or len(payload) > _MAX_POLICY_BYTES:
        raise AuditPolicyError("dependency exception policy has an invalid size")
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuditPolicyError("dependency exception policy is not strict UTF-8 JSON") from error


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise AuditPolicyError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise AuditPolicyError(f"{field} must be an ISO date") from error


def _normalized_requirement_lines(path: Path) -> frozenset[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise AuditPolicyError("requirements file cannot be read as UTF-8") from error
    lines: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].strip().casefold().replace("_", "-")
        if line:
            lines.add(line)
    return frozenset(lines)


def load_policy(
    policy_path: Path,
    requirements_path: Path,
    *,
    today: date | None = None,
) -> tuple[DependencyException, ...]:
    """Validate the policy and return only explicit, currently active exceptions."""

    payload = _read_policy(policy_path)
    if not isinstance(payload, dict) or frozenset(payload) != _ROOT_KEYS:
        raise AuditPolicyError("dependency exception policy has unexpected root fields")
    if payload["schema_version"] != 1:
        raise AuditPolicyError("dependency exception policy schema is unsupported")
    entries = payload["exceptions"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= _MAX_EXCEPTIONS:
        raise AuditPolicyError("dependency exception policy must contain a bounded exception list")

    current_date = today or date.today()
    requirement_lines = _normalized_requirement_lines(requirements_path)
    seen_ids: set[str] = set()
    exceptions: list[DependencyException] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict) or frozenset(raw_entry) != _ENTRY_KEYS:
            raise AuditPolicyError("dependency exception has unexpected fields")

        vulnerability_id = raw_entry["vulnerability_id"]
        aliases = raw_entry["aliases"]
        package = raw_entry["package"]
        pinned_version = raw_entry["pinned_version"]
        advisory = raw_entry["advisory"]
        owner = raw_entry["owner"]
        rationale = raw_entry["rationale"]
        controls = raw_entry["compensating_controls"]
        if not isinstance(vulnerability_id, str) or _VULNERABILITY_ID.fullmatch(
            vulnerability_id
        ) is None:
            raise AuditPolicyError("dependency exception vulnerability_id is invalid")
        if vulnerability_id in seen_ids:
            raise AuditPolicyError("dependency exception vulnerability_id is duplicated")
        if (
            not isinstance(aliases, list)
            or not 1 <= len(aliases) <= 8
            or any(
                not isinstance(alias, str) or _VULNERABILITY_ID.fullmatch(alias) is None
                for alias in aliases
            )
            or vulnerability_id in aliases
            or len(set(aliases)) != len(aliases)
        ):
            raise AuditPolicyError("dependency exception aliases are invalid")
        if not isinstance(package, str) or _PACKAGE_NAME.fullmatch(package) is None:
            raise AuditPolicyError("dependency exception package is invalid")
        if not isinstance(pinned_version, str) or _VERSION.fullmatch(pinned_version) is None:
            raise AuditPolicyError("dependency exception pinned_version is invalid")
        if not isinstance(advisory, str) or not advisory.startswith("https://"):
            raise AuditPolicyError("dependency exception advisory must use HTTPS")
        if not isinstance(owner, str) or not 2 <= len(owner) <= 128:
            raise AuditPolicyError("dependency exception owner is invalid")
        if not isinstance(rationale, str) or not 40 <= len(rationale) <= 1000:
            raise AuditPolicyError("dependency exception rationale is invalid")
        if (
            not isinstance(controls, list)
            or not 1 <= len(controls) <= 12
            or any(not isinstance(control, str) or not 15 <= len(control) <= 500 for control in controls)
        ):
            raise AuditPolicyError("dependency exception compensating controls are invalid")

        reviewed_on = _parse_date(raw_entry["reviewed_on"], "reviewed_on")
        expires_on = _parse_date(raw_entry["expires_on"], "expires_on")
        if reviewed_on > current_date or reviewed_on >= expires_on:
            raise AuditPolicyError("dependency exception review window is invalid")
        if (expires_on - reviewed_on).days > 31:
            raise AuditPolicyError("dependency exception review window exceeds 31 days")
        if current_date >= expires_on:
            raise AuditPolicyError(
                f"dependency exception {vulnerability_id} expired on {expires_on.isoformat()}"
            )

        exact_pin = f"{package.casefold().replace('_', '-')}=={pinned_version.casefold()}"
        if exact_pin not in requirement_lines:
            raise AuditPolicyError(
                f"dependency exception {vulnerability_id} requires exact pin {exact_pin}"
            )
        seen_ids.add(vulnerability_id)
        exceptions.append(
            DependencyException(
                vulnerability_id=vulnerability_id,
                aliases=tuple(aliases),
                package=package,
                pinned_version=pinned_version,
                advisory=advisory,
                owner=owner,
                reviewed_on=reviewed_on,
                expires_on=expires_on,
                rationale=rationale,
                compensating_controls=tuple(controls),
            )
        )
    return tuple(exceptions)


def build_command(
    requirements_path: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pip_audit",
        "--progress-spinner",
        "off",
        "--strict",
        "--format",
        "json",
        "-r",
        str(requirements_path),
    )


def parse_report(payload: str) -> tuple[AuditFinding, ...]:
    try:
        report = json.loads(payload, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, AuditPolicyError) as error:
        raise AuditPolicyError("pip-audit returned invalid JSON") from error
    if not isinstance(report, dict) or not isinstance(report.get("dependencies"), list):
        raise AuditPolicyError("pip-audit report has an unsupported shape")

    findings: list[AuditFinding] = []
    for dependency in report["dependencies"]:
        if not isinstance(dependency, dict):
            raise AuditPolicyError("pip-audit dependency entry is invalid")
        package = dependency.get("name")
        version = dependency.get("version")
        vulnerabilities = dependency.get("vulns")
        if not isinstance(package, str) or not isinstance(version, str):
            raise AuditPolicyError("pip-audit dependency identity is invalid")
        if not isinstance(vulnerabilities, list):
            raise AuditPolicyError("pip-audit vulnerability list is invalid")
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict):
                raise AuditPolicyError("pip-audit vulnerability entry is invalid")
            vulnerability_id = vulnerability.get("id")
            aliases = vulnerability.get("aliases", [])
            fix_versions = vulnerability.get("fix_versions", [])
            if (
                not isinstance(vulnerability_id, str)
                or not isinstance(aliases, list)
                or any(not isinstance(alias, str) for alias in aliases)
                or not isinstance(fix_versions, list)
                or any(not isinstance(version, str) for version in fix_versions)
            ):
                raise AuditPolicyError("pip-audit vulnerability fields are invalid")
            findings.append(
                AuditFinding(
                    package=package.casefold().replace("_", "-"),
                    version=version,
                    vulnerability_id=vulnerability_id,
                    aliases=tuple(aliases),
                    fix_versions=tuple(fix_versions),
                )
            )
    return tuple(findings)


def evaluate_findings(
    exceptions: tuple[DependencyException, ...],
    findings: tuple[AuditFinding, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    used: set[str] = set()
    for finding in findings:
        matches = [
            exception
            for exception in exceptions
            if finding.identifiers & exception.identifiers
        ]
        if len(matches) != 1:
            errors.append(
                f"unapproved vulnerability: {finding.package}=={finding.version} "
                f"{finding.vulnerability_id}"
            )
            continue
        exception = matches[0]
        expected_package = exception.package.casefold().replace("_", "-")
        if finding.package != expected_package or finding.version != exception.pinned_version:
            errors.append(
                f"exception scope mismatch: {finding.package}=={finding.version} "
                f"{finding.vulnerability_id}"
            )
            continue
        if finding.fix_versions:
            errors.append(
                f"patched version available for {finding.vulnerability_id}: "
                f"{', '.join(finding.fix_versions)}"
            )
            continue
        used.add(exception.vulnerability_id)

    for exception in exceptions:
        if exception.vulnerability_id not in used:
            errors.append(f"stale dependency exception: {exception.vulnerability_id}")
    return tuple(errors)


def run_audit(
    requirements_path: Path,
    exceptions: tuple[DependencyException, ...],
) -> int:
    completed = subprocess.run(
        build_command(requirements_path),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode or 2
    try:
        findings = parse_report(completed.stdout)
    except AuditPolicyError as error:
        print(f"dependency audit report rejected: {error}", file=sys.stderr)
        return 2
    errors = evaluate_findings(exceptions, findings)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"dependency audit passed: {len(findings)} accepted finding(s), "
        "no unapproved vulnerabilities"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        exceptions = load_policy(arguments.policy, arguments.requirements)
    except AuditPolicyError as error:
        print(f"dependency audit policy rejected: {error}", file=sys.stderr)
        return 2

    for exception in exceptions:
        print(
            "active dependency exception: "
            f"{exception.vulnerability_id} {exception.package}=={exception.pinned_version} "
            f"expires {exception.expires_on.isoformat()}"
        )
    if arguments.validate_only:
        return 0
    return run_audit(arguments.requirements, exceptions)


if __name__ == "__main__":
    raise SystemExit(main())
