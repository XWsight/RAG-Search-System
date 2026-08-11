from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_NEUTRAL_MODULES = (
    "rag_system/application.py",
    "rag_system/application_ports.py",
    "rag_system/domain.py",
    "rag_system/ports.py",
    "rag_system/grounding.py",
    "rag_system/answer_protocol.py",
    "rag_system/provider_errors.py",
    "rag_system/json_contract.py",
    "rag_system/submission.py",
    "rag_system/coordination.py",
    "rag_system/assets.py",
    "rag_system/indexing.py",
    "rag_system/health.py",
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

    def test_http_boundary_depends_on_application_contract_not_platform_implementation(self) -> None:
        forbidden = {
            "rag_system.catalog",
            "rag_system.file_store",
            "rag_system.idempotency",
            "rag_system.jobs",
            "rag_system.loaders",
            "rag_system.platform",
            "rag_system.providers",
            "rag_system.retrieval",
            "rag_system.service",
        }
        imports = set(_imports(ROOT / "rag_system/api.py"))
        self.assertEqual(sorted(imports & forbidden), [])
        self.assertIn("rag_system.application", imports)

    def test_platform_depends_on_runtime_ports_not_concrete_adapters(self) -> None:
        forbidden = {
            "ChromaIndexRepository",
            "IdempotencyStore",
            "JobManager",
            "KnowledgeBaseCatalog",
            "RagService",
            "TenantFileStore",
        }
        imported = set(_imported_symbols(ROOT / "rag_system/platform.py"))
        self.assertEqual(sorted(imported & forbidden), [])

    def test_production_modules_have_no_import_cycles(self) -> None:
        modules = {
            path.stem: path
            for path in (ROOT / "rag_system").glob("*.py")
            if path.name != "__init__.py"
        }
        graph = {
            name: {
                imported.removeprefix("rag_system.").split(".", maxsplit=1)[0]
                for imported in _imports(path)
                if imported.startswith("rag_system.")
                and imported.removeprefix("rag_system.").split(".", maxsplit=1)[0]
                in modules
            }
            for name, path in modules.items()
        }

        cycles: list[str] = []
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(module: str) -> None:
            if module in visiting:
                start = visiting.index(module)
                cycles.append(" -> ".join((*visiting[start:], module)))
                return
            if module in visited:
                return
            visiting.append(module)
            for dependency in sorted(graph[module]):
                visit(dependency)
            visiting.pop()
            visited.add(module)

        for module in sorted(graph):
            visit(module)
        self.assertEqual(cycles, [])


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return tuple(result)


def _imported_symbols(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(item.asname or item.name.split(".", maxsplit=1)[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.extend(item.asname or item.name for item in node.names)
    return tuple(result)


if __name__ == "__main__":
    unittest.main()
