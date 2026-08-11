from __future__ import annotations

import hashlib
import io
import logging
import unittest

from fastapi.testclient import TestClient

from rag_system.api import create_app
from rag_system.application import IdempotencyInProgressError, KnowledgeBaseSubmission
from rag_system.catalog import DocumentManifest, KnowledgeBaseRecord, KnowledgeBaseStatus
from rag_system.config import Settings
from rag_system.domain import AnswerClaim, AnswerResult, Citation, Route, RouteDecision
from rag_system.idempotency import IdempotencyConflictError
from rag_system.jobs import JobId, JobSnapshot, JobStatus
from rag_system.metrics import create_operational_metrics
from rag_system.observability import JsonEventLogger
from rag_system.rate_limit import TokenBucketRateLimiter
from rag_system.tenancy import ApiKeyAuthenticator, Principal, TenantId


READER_KEY = "reader-key-0123456789abcdef"
WRITER_KEY = "writer-key-0123456789abcdef"
OPERATOR_KEY = "operator-key-0123456789abcdef"
ALL_ROLES_KEY = "service-key-0123456789abcdef"
KNOWLEDGE_BASE_ID = "kb_0123456789abcdef0123456789abcdef"
JOB_ID = "job_0123456789abcdef"


def _principal(subject: str, roles: set[str]) -> Principal:
    return Principal(subject, TenantId("tenant-one"), frozenset(roles))


def _record(*, status: KnowledgeBaseStatus = KnowledgeBaseStatus.READY) -> KnowledgeBaseRecord:
    document = DocumentManifest(
        display_name="guide.md",
        relative_path="doc_0123456789abcdef/guide.md",
        size_bytes=12,
        sha256="a" * 64,
    )
    return KnowledgeBaseRecord(
        resource_id=KNOWLEDGE_BASE_ID,
        tenant_id=TenantId("tenant-one"),
        display_name="Engineering guide",
        status=status,
        internal_index_id="index_0123456789abcdef" if status is KnowledgeBaseStatus.READY else None,
        documents=(document,),
        document_count=1,
        total_bytes=12,
        chunk_count=3 if status is KnowledgeBaseStatus.READY else 0,
        error_code=None,
        created_at=100.0,
        updated_at=101.0,
        version=2,
    )


class FakePlatform:
    def __init__(self) -> None:
        self.settings = Settings(
            max_file_bytes=8,
            max_total_bytes=12,
            max_documents=2,
            max_question_characters=50,
        )
        self.metrics = create_operational_metrics("api_test")
        self.record = _record()
        self.closed = False
        self.created_documents = ()
        self.created_name = ""
        self.idempotency_key = ""
        self.submission_replayed = False
        self.create_error: Exception | None = None
        self.last_answer_request = None
        self.raise_on_list: Exception | None = None

    def close(self) -> None:
        self.closed = True

    def create_knowledge_base(
        self,
        principal: Principal,
        *,
        display_name: str,
        documents,
        idempotency_key: str,
    ) -> KnowledgeBaseSubmission:
        del principal
        if self.create_error is not None:
            raise self.create_error
        self.created_name = display_name
        self.created_documents = tuple(documents)
        self.idempotency_key = idempotency_key
        pending = _record(status=KnowledgeBaseStatus.PENDING)
        return KnowledgeBaseSubmission(
            pending,
            JobId(JOB_ID),
            replayed=self.submission_replayed,
        )

    def list_knowledge_bases(self, principal: Principal, *, limit: int, offset: int):
        del principal, limit, offset
        if self.raise_on_list is not None:
            raise self.raise_on_list
        return (self.record,)

    def get_knowledge_base(self, principal: Principal, resource_id: str):
        del principal
        if resource_id != KNOWLEDGE_BASE_ID:
            raise RuntimeError("hidden storage lookup detail")
        return self.record

    def delete_knowledge_base(self, principal: Principal, resource_id: str) -> bool:
        del principal
        return resource_id == KNOWLEDGE_BASE_ID

    def get_job(self, principal: Principal, job_id: str) -> JobSnapshot:
        del principal
        return self._job(job_id)

    def cancel_job(self, principal: Principal, job_id: str) -> JobSnapshot:
        del principal
        snapshot = self._job(job_id)
        return JobSnapshot(
            job_id=snapshot.job_id,
            status=JobStatus.CANCELLED,
            created_at=snapshot.created_at,
            updated_at=102.0,
            finished_at=102.0,
        )

    def answer(self, principal: Principal, resource_id: str, request):
        del principal, resource_id
        self.last_answer_request = request
        return AnswerResult(
            answer="RAG retrieves evidence before generation.",
            decision=RouteDecision(Route.LOCAL, 0.91, "internal routing explanation"),
            claims=(
                AnswerClaim(
                    text="RAG retrieves evidence before generation.",
                    citation_ids=("L1",),
                ),
            ),
            citations=(
                Citation(
                    citation_id="L1",
                    source_name="guide.md",
                    excerpt="Grounded evidence.",
                    score=0.88,
                ),
            ),
            trace_id="internal-trace-must-not-leak",
            latency_ms=12.5,
            diagnostics={"internal_secret": "must-not-leak"},
        )

    def clear_session(self, principal: Principal, resource_id: str, session_id: str) -> bool:
        del principal, resource_id, session_id
        return True

    @staticmethod
    def _job(job_id: str) -> JobSnapshot:
        return JobSnapshot(
            job_id=JobId(job_id),
            status=JobStatus.SUCCEEDED,
            created_at=100.0,
            updated_at=101.0,
            started_at=100.1,
            finished_at=101.0,
            result={"knowledge_base_id": KNOWLEDGE_BASE_ID, "chunk_count": 3},
            error_message="internal worker detail must not leak",
        )


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.platform = FakePlatform()
        self.authenticator = ApiKeyAuthenticator.from_mapping(
            {
                READER_KEY: _principal("reader-user", {"reader"}),
                WRITER_KEY: _principal("writer-user", {"writer"}),
                OPERATOR_KEY: _principal("operator-user", {"operator"}),
                ALL_ROLES_KEY: _principal("service-user", {"reader", "writer", "operator"}),
            }
        )
        self.log_stream = io.StringIO()
        sink = logging.Logger(f"test-api-{id(self)}")
        sink.handlers.clear()
        sink.propagate = False
        sink.addHandler(logging.StreamHandler(self.log_stream))
        self.logger = JsonEventLogger(sink)
        self.app = create_app(
            platform=self.platform,
            authenticator=self.authenticator,
            rate_limiter=TokenBucketRateLimiter(rate_per_second=100, capacity=100),
            logger=self.logger,
        )
        self.client_context = TestClient(self.app, raise_server_exceptions=False)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    @staticmethod
    def _headers(key: str = ALL_ROLES_KEY) -> dict[str, str]:
        return {"X-API-Key": key}

    def test_liveness_readiness_and_correlation_headers(self) -> None:
        response = self.client.get(
            "/health/live",
            headers={"X-Trace-ID": "client-trace-1", "X-Request-ID": "client-request-1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(response.headers["x-trace-id"], "client-trace-1")
        self.assertEqual(response.headers["x-request-id"], "client-request-1")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("access-control-allow-origin", response.headers)

        ready = self.client.get("/health/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json(), {"status": "ready"})

    def test_openapi_describes_knowledge_base_documents_as_uploadable_files(self) -> None:
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        request_schema = (
            schema["paths"]["/v1/knowledge-bases"]["post"]["requestBody"]["content"]
            ["multipart/form-data"]["schema"]
        )
        component_name = request_schema["$ref"].rsplit("/", maxsplit=1)[-1]
        files = schema["components"]["schemas"][component_name]["properties"]["files"]
        self.assertEqual(files["type"], "array")
        self.assertEqual(files["items"], {"type": "string", "format": "binary"})

    def test_product_web_app_is_packaged_same_origin_and_hardened(self) -> None:
        root = self.client.get("/", follow_redirects=False)
        self.assertEqual(root.status_code, 307)
        self.assertEqual(root.headers["location"], "/app")

        page = self.client.get("/app")
        self.assertEqual(page.status_code, 200)
        self.assertIn("RAG Studio", page.text)
        self.assertIn("外部服务授权", page.text)
        self.assertIn("资料详情", page.text)
        self.assertIn("/app/assets/app.js", page.text)
        self.assertNotIn(ALL_ROLES_KEY, page.text)
        self.assertIn("default-src 'self'", page.headers["content-security-policy"])
        self.assertEqual(page.headers["x-frame-options"], "DENY")
        self.assertEqual(page.headers["referrer-policy"], "no-referrer")

        script = self.client.get("/app/assets/app.js")
        stylesheet = self.client.get("/app/assets/styles.css")
        self.assertEqual(script.status_code, 200)
        self.assertIn("/v1/knowledge-bases", script.text)
        self.assertIn("sessionStorage", script.text)
        self.assertNotIn("localStorage", script.text)
        self.assertIn("requestExternalConsent", script.text)
        self.assertIn("navigator.clipboard.writeText", script.text)
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn(".app-shell", stylesheet.text)

    def test_unready_is_safe_503_envelope(self) -> None:
        app = create_app(
            platform=self.platform,
            authenticator=self.authenticator,
            rate_limiter=TokenBucketRateLimiter(rate_per_second=100, capacity=100),
            logger=self.logger,
            readiness=False,
            close_on_shutdown=False,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "not_ready")
        self.assertIn("trace_id", response.json()["error"])

    def test_lifespan_uses_the_injected_runtime_shutdown(self) -> None:
        shutdown_calls: list[str] = []
        app = create_app(
            platform=self.platform,
            authenticator=self.authenticator,
            rate_limiter=TokenBucketRateLimiter(rate_per_second=100, capacity=100),
            logger=self.logger,
            shutdown=lambda: shutdown_calls.append("closed"),
        )
        with TestClient(app, raise_server_exceptions=False):
            pass
        self.assertEqual(shutdown_calls, ["closed"])

    def test_authentication_accepts_api_key_or_bearer_and_rejects_ambiguous_credentials(self) -> None:
        missing = self.client.get("/v1/knowledge-bases")
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.json()["error"]["code"], "authentication_failed")
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")

        bearer = self.client.get(
            "/v1/knowledge-bases",
            headers={"Authorization": f"Bearer {READER_KEY}"},
        )
        self.assertEqual(bearer.status_code, 200)

        ambiguous = self.client.get(
            "/v1/knowledge-bases",
            headers={"Authorization": f"Bearer {READER_KEY}", "X-API-Key": READER_KEY},
        )
        self.assertEqual(ambiguous.status_code, 401)

    def test_roles_are_enforced_per_operation(self) -> None:
        reader_list = self.client.get(
            "/v1/knowledge-bases", headers=self._headers(READER_KEY)
        )
        self.assertEqual(reader_list.status_code, 200)

        reader_create = self.client.post(
            "/v1/knowledge-bases",
            headers={**self._headers(READER_KEY), "Idempotency-Key": "create-1"},
            data={"name": "docs"},
            files=[("files", ("a.md", b"text", "text/markdown"))],
        )
        self.assertEqual(reader_create.status_code, 403)

        writer_list = self.client.get(
            "/v1/knowledge-bases", headers=self._headers(WRITER_KEY)
        )
        self.assertEqual(writer_list.status_code, 403)

        reader_metrics = self.client.get("/metrics", headers=self._headers(READER_KEY))
        self.assertEqual(reader_metrics.status_code, 403)

    def test_create_enforces_upload_bounds_and_passes_bounded_bytes(self) -> None:
        short_key = self.client.post(
            "/v1/knowledge-bases",
            headers={**self._headers(), "Idempotency-Key": "short"},
            data={"name": "docs"},
            files=[("files", ("a.md", b"text", "text/markdown"))],
        )
        self.assertEqual(short_key.status_code, 422)
        self.assertEqual(short_key.json()["error"]["code"], "invalid_request")

        response = self.client.post(
            "/v1/knowledge-bases",
            headers={**self._headers(), "Idempotency-Key": "create-1"},
            data={"name": "  Product docs  "},
            files=[("files", ("a.md", b"12345678", "text/markdown"))],
        )
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["job_id"], JOB_ID)
        self.assertFalse(response.json()["replayed"])
        self.assertEqual(self.platform.created_name, "Product docs")
        self.assertEqual(self.platform.idempotency_key, "create-1")
        self.assertEqual(self.platform.created_documents[0].display_name, "a.md")
        self.assertEqual(self.platform.created_documents[0].source, b"12345678")

        oversized = self.client.post(
            "/v1/knowledge-bases",
            headers={**self._headers(), "Idempotency-Key": "create-2"},
            data={"name": "docs"},
            files=[("files", ("a.md", b"123456789", "text/markdown"))],
        )
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(oversized.json()["error"]["code"], "upload_limit_exceeded")

        too_many = self.client.post(
            "/v1/knowledge-bases",
            headers={**self._headers(), "Idempotency-Key": "create-3"},
            data={"name": "docs"},
            files=[
                ("files", ("a.md", b"1", "text/markdown")),
                ("files", ("b.md", b"2", "text/markdown")),
                ("files", ("c.md", b"3", "text/markdown")),
            ],
        )
        self.assertEqual(too_many.status_code, 413)

        raw_oversize = self.client.post(
            "/v1/knowledge-bases",
            headers={
                **self._headers(),
                "Idempotency-Key": "create-4",
                "Content-Length": "1000000",
            },
        )
        self.assertEqual(raw_oversize.status_code, 413)
        self.assertEqual(raw_oversize.json()["error"]["code"], "upload_limit_exceeded")

    def test_create_reports_replay_and_safe_idempotency_conflicts(self) -> None:
        self.platform.submission_replayed = True
        replay = self.client.post(
            "/v1/knowledge-bases",
            headers={**self._headers(), "Idempotency-Key": "create-replay"},
            data={"name": "docs"},
            files=[("files", ("a.md", b"text", "text/markdown"))],
        )
        self.assertEqual(replay.status_code, 202)
        self.assertTrue(replay.json()["replayed"])

        self.platform.create_error = IdempotencyConflictError()
        conflict = self.client.post(
            "/v1/knowledge-bases",
            headers={**self._headers(), "Idempotency-Key": "conflicting-key"},
            data={"name": "docs"},
            files=[("files", ("a.md", b"text", "text/markdown"))],
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["error"]["code"], "idempotency_conflict")

        self.platform.create_error = IdempotencyInProgressError("private detail")
        pending = self.client.post(
            "/v1/knowledge-bases",
            headers={**self._headers(), "Idempotency-Key": "pending-key"},
            data={"name": "docs"},
            files=[("files", ("a.md", b"text", "text/markdown"))],
        )
        self.assertEqual(pending.status_code, 409)
        self.assertEqual(pending.json()["error"]["code"], "idempotency_in_progress")
        self.assertNotIn("private detail", pending.text)

    def test_knowledge_base_responses_do_not_expose_tenant_paths_or_internal_index(self) -> None:
        response = self.client.get(
            f"/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}", headers=self._headers()
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        serialized = response.text
        self.assertEqual(payload["id"], KNOWLEDGE_BASE_ID)
        self.assertNotIn("tenant_id", payload)
        self.assertNotIn("internal_index_id", payload)
        self.assertNotIn("relative_path", serialized)
        self.assertNotIn("index_0123456789abcdef", serialized)

    def test_jobs_can_be_read_and_cancelled_without_worker_messages(self) -> None:
        response = self.client.get(f"/v1/jobs/{JOB_ID}", headers=self._headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "succeeded")
        self.assertNotIn("error_message", response.json())
        self.assertNotIn("internal worker detail", response.text)

        cancelled = self.client.delete(f"/v1/jobs/{JOB_ID}", headers=self._headers())
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["status"], "cancelled")

    def test_answer_is_strict_and_does_not_echo_question_or_internal_diagnostics(self) -> None:
        question = "What is retrieval augmented generation?"
        response = self.client.post(
            "/v1/answers",
            headers={**self._headers(), "X-Trace-ID": "public-trace"},
            json={
                "knowledge_base_id": KNOWLEDGE_BASE_ID,
                "question": question,
                "session_id": "session-1",
                "allow_cloud": True,
                "allow_web": False,
                "deep_research": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["decision"]["route"], "local")
        self.assertEqual(payload["trace_id"], "public-trace")
        self.assertNotIn(question, response.text)
        self.assertNotIn("internal routing explanation", response.text)
        self.assertNotIn("internal-trace-must-not-leak", response.text)
        self.assertNotIn("internal_secret", response.text)
        self.assertEqual(
            payload["claims"],
            [
                {
                    "text": "RAG retrieves evidence before generation.",
                    "citation_ids": ["L1"],
                }
            ],
        )
        self.assertTrue(self.platform.last_answer_request.allow_cloud)

        unknown = self.client.post(
            "/v1/answers",
            headers=self._headers(),
            json={
                "knowledge_base_id": KNOWLEDGE_BASE_ID,
                "question": "secret-question-value",
                "session_id": "session-1",
                "unknown": "secret-extra-value",
            },
        )
        self.assertEqual(unknown.status_code, 422)
        self.assertNotIn("secret-question-value", unknown.text)
        self.assertNotIn("secret-extra-value", unknown.text)

    def test_session_and_knowledge_base_deletion(self) -> None:
        session = self.client.delete(
            f"/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/sessions/session-1",
            headers=self._headers(),
        )
        self.assertEqual(session.status_code, 200)
        self.assertEqual(session.json(), {"deleted": True})

        knowledge_base = self.client.delete(
            f"/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}", headers=self._headers()
        )
        self.assertEqual(knowledge_base.status_code, 200)
        self.assertEqual(knowledge_base.json(), {"deleted": True})

    def test_rate_limit_returns_retry_after_without_tenant_identifier(self) -> None:
        limiter = TokenBucketRateLimiter(rate_per_second=0.01, capacity=1)
        app = create_app(
            platform=self.platform,
            authenticator=self.authenticator,
            rate_limiter=limiter,
            logger=self.logger,
            close_on_shutdown=False,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            first = client.get("/v1/knowledge-bases", headers=self._headers(READER_KEY))
            second = client.get("/v1/knowledge-bases", headers=self._headers(READER_KEY))
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertGreaterEqual(int(second.headers["retry-after"]), 1)
        self.assertNotIn("tenant-one", second.text)
        self.assertEqual(second.json()["error"]["code"], "rate_limit_exceeded")

    def test_metrics_require_operator_and_use_prometheus_content_type(self) -> None:
        response = self.client.get("/metrics", headers=self._headers(OPERATOR_KEY))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/plain; version=0.0.4")
        )
        self.assertIn("api_test_requests_total", response.text)

    def test_unknown_exceptions_and_unknown_routes_use_safe_envelopes(self) -> None:
        secret = "database-password-and-query-text"
        self.platform.raise_on_list = RuntimeError(secret)
        response = self.client.get("/v1/knowledge-bases", headers=self._headers())
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "internal_error")
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, self.log_stream.getvalue())
        self.assertEqual(response.headers["x-trace-id"], response.json()["error"]["trace_id"])
        self.assertEqual(
            response.headers["x-request-id"], response.json()["error"]["request_id"]
        )

        missing = self.client.get("/does-not-exist")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "resource_unavailable")

    def test_invalid_client_correlation_ids_are_not_reflected(self) -> None:
        malicious = "bad\r\nX-Injected: yes"
        response = self.client.get("/health/live", headers={"X-Trace-ID": malicious})
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.headers["x-trace-id"], malicious)
        self.assertRegex(response.headers["x-trace-id"], r"^trace_[0-9a-f]{32}$")

    def test_openapi_declares_both_authentication_schemes(self) -> None:
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        document = response.json()
        schemes = document["components"]["securitySchemes"]
        self.assertEqual(schemes["ApiKeyAuth"]["in"], "header")
        self.assertEqual(schemes["ApiKeyAuth"]["name"], "X-API-Key")
        self.assertEqual(schemes["BearerAuth"]["scheme"], "bearer")
        operation = document["paths"]["/v1/answers"]["post"]
        self.assertIn({"ApiKeyAuth": []}, operation["security"])
        self.assertIn({"BearerAuth": []}, operation["security"])
        answer_schemas = [
            schema
            for name, schema in document["components"]["schemas"].items()
            if name.endswith("ConfiguredAnswerPayload")
        ]
        self.assertEqual(len(answer_schemas), 1)
        self.assertEqual(answer_schemas[0]["properties"]["question"]["maxLength"], 50)
        list_operation = response.json()["paths"]["/v1/knowledge-bases"]["get"]
        offset = next(item for item in list_operation["parameters"] if item["name"] == "offset")
        self.assertEqual(offset["schema"]["maximum"], 10_000)
        create_operation = document["paths"]["/v1/knowledge-bases"]["post"]
        idempotency_header = next(
            item
            for item in create_operation["parameters"]
            if item["name"] == "Idempotency-Key"
        )
        self.assertEqual(idempotency_header["schema"]["pattern"], r"^[!-~]{8,128}$")
        for status_code in ("401", "403", "409", "413", "422", "429", "500", "503"):
            schema = create_operation["responses"][status_code]["content"][
                "application/json"
            ]["schema"]
            self.assertEqual(schema["$ref"], "#/components/schemas/ErrorEnvelope")

    def test_logging_uses_hashed_tenant_only(self) -> None:
        response = self.client.get("/v1/knowledge-bases", headers=self._headers(READER_KEY))
        self.assertEqual(response.status_code, 200)
        logs = self.log_stream.getvalue()
        self.assertNotIn("tenant-one", logs)
        expected_hash = hashlib.sha256(b"tenant-one").hexdigest()[:16]
        self.assertIn(expected_hash, logs)


if __name__ == "__main__":
    unittest.main()
