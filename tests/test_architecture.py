from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_NEUTRAL_MODULES = (
    "rag_system/domain.py",
    "rag_system/ports.py",
    "rag_system/grounding.py",
    "rag_system/answer_protocol.py",
    "rag_system/provider_errors.py",
    "rag_system/json_contract.py",
)
FORBIDDEN_FRAMEWORK_PREFIXES = (
    "chromadb",
    "fastapi",
    "gradio",
    "langchain",
    "pydantic",
    "requests",
)


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_domain_and_protocol_modules_do_not_import_frameworks(self) -> None:
        violations: list[str] = []
        for relative in FRAMEWORK_NEUTRAL_MODULES:
            for imported in _imports(ROOT / relative):
                if imported.startswith(FORBIDDEN_FRAMEWORK_PREFIXES):
                    violations.append(f"{relative}: {imported}")
        self.assertEqual(violations, [])

    def test_application_and_http_layers_do_not_import_concrete_providers(self) -> None:
        violations = [
            f"{relative}: {imported}"
            for relative in ("rag_system/service.py", "rag_system/api.py")
            for imported in _imports(ROOT / relative)
            if imported == "rag_system.providers"
        ]
        self.assertEqual(violations, [])


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return tuple(result)


if __name__ == "__main__":
    unittest.main()
