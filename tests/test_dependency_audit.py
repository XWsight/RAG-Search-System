from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.audit_dependencies import (
    AuditFinding,
    AuditPolicyError,
    build_command,
    evaluate_findings,
    load_policy,
)


class DependencyAuditPolicyTests(unittest.TestCase):
    @staticmethod
    def _write_fixture(directory: str, *, expires_on: str = "2026-09-01") -> tuple[Path, Path]:
        root = Path(directory)
        requirements = root / "requirements.txt"
        requirements.write_text("chromadb==1.5.9\npypdf==6.15.0\n", encoding="utf-8")
        policy = root / "policy.json"
        policy.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "exceptions": [
                        {
                            "vulnerability_id": "PYSEC-2026-311",
                            "aliases": ["CVE-2026-45829", "GHSA-f4j7-r4q5-qw2c"],
                            "package": "chromadb",
                            "pinned_version": "1.5.9",
                            "advisory": "https://github.com/advisories/example",
                            "owner": "XWsight",
                            "reviewed_on": "2026-08-11",
                            "expires_on": expires_on,
                            "rationale": "The vulnerable network endpoint is not part of this deployment boundary.",
                            "compensating_controls": [
                                "The dependency is embedded and its server endpoint is not exposed."
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return policy, requirements

    def test_valid_policy_builds_unfiltered_json_audit_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy, requirements = self._write_fixture(directory)
            exceptions = load_policy(
                policy,
                requirements,
                today=date(2026, 8, 11),
            )
            command = build_command(requirements)
            self.assertEqual(len(exceptions), 1)
            self.assertIn("--strict", command)
            self.assertIn("json", command)

    def test_policy_fails_closed_on_expiry_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy, requirements = self._write_fixture(directory)
            with self.assertRaisesRegex(AuditPolicyError, "expired"):
                load_policy(policy, requirements, today=date(2026, 9, 1))

    def test_policy_requires_the_exact_dependency_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy, requirements = self._write_fixture(directory)
            requirements.write_text("chromadb==1.5.8\n", encoding="utf-8")
            with self.assertRaisesRegex(AuditPolicyError, "exact pin"):
                load_policy(policy, requirements, today=date(2026, 8, 11))

    def test_policy_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy, requirements = self._write_fixture(directory)
            policy.write_text(
                '{"schema_version":1,"schema_version":1,"exceptions":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditPolicyError, "duplicate key"):
                load_policy(policy, requirements, today=date(2026, 8, 11))

    def test_report_allows_only_the_exact_unfixed_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy, requirements = self._write_fixture(directory)
            exceptions = load_policy(policy, requirements, today=date(2026, 8, 11))
            finding = AuditFinding(
                package="chromadb",
                version="1.5.9",
                vulnerability_id="PYSEC-2026-311",
                aliases=("CVE-2026-45829",),
                fix_versions=(),
            )
            self.assertEqual(evaluate_findings(exceptions, (finding,)), ())

            fixed = AuditFinding(
                package="chromadb",
                version="1.5.9",
                vulnerability_id="PYSEC-2026-311",
                aliases=(),
                fix_versions=("1.5.10",),
            )
            self.assertIn("patched version available", evaluate_findings(exceptions, (fixed,))[0])

    def test_report_rejects_new_findings_and_stale_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy, requirements = self._write_fixture(directory)
            exceptions = load_policy(policy, requirements, today=date(2026, 8, 11))
            finding = AuditFinding(
                package="example",
                version="1.0",
                vulnerability_id="PYSEC-2026-999",
                aliases=(),
                fix_versions=(),
            )
            errors = evaluate_findings(exceptions, (finding,))
            self.assertTrue(any("unapproved vulnerability" in error for error in errors))
            self.assertTrue(any("stale dependency exception" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
