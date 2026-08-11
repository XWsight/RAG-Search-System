"""Application-level question answering orchestration."""

from __future__ import annotations

import time
from collections.abc import Sequence
from uuid import uuid4

from rag_system.config import Settings
from rag_system.domain import (
    AnswerClaim,
    AnswerRequest,
    AnswerResult,
    Citation,
    IndexRef,
    Route,
    RouteDecision,
    SearchHit,
    WebSearchResult,
)
from rag_system.grounding import (
    GroundingContractError,
    render_grounded_answer,
    validate_grounded_answer,
)
from rag_system.index_manager import IndexManager
from rag_system.ingestion import IngestionResult
from rag_system.memory import ConversationMemory
from rag_system.ports import ChatModel, QueryPlanner, Retriever, WebSearchProvider
from rag_system.providers import ProviderError
from rag_system.research import fuse_query_hits, normalize_query_plan
from rag_system.retrieval import RoutingPolicy
from rag_system.text import truncate_text
from rag_system.web import rank_web_results


class RagService:
    """Coordinate retrieval, privacy-aware routing, generation, and citations."""

    def __init__(
        self,
        settings: Settings,
        index_manager: IndexManager,
        chat_model: ChatModel,
        web_search: WebSearchProvider,
        *,
        memory: ConversationMemory | None = None,
        query_planner: QueryPlanner | None = None,
        timer=time.perf_counter,
    ) -> None:
        self.settings = settings.validate()
        self.index_manager = index_manager
        self.chat_model = chat_model
        self.web_search = web_search
        self.query_planner = query_planner
        self.routing = RoutingPolicy(settings)
        self.memory = memory or ConversationMemory(
            max_sessions=settings.max_sessions,
            ttl_seconds=settings.session_ttl_seconds,
            max_rounds=settings.memory_max_rounds,
            max_characters=settings.memory_max_characters,
            summary_max_characters=settings.retrieval_history_characters,
        )
        self.timer = timer

    def create_index(
        self,
        paths: Sequence[str] | None = None,
        *,
        namespace: str = "",
    ) -> IndexRef:
        return self.index_manager.build(paths, namespace=namespace)

    def prepare_index(
        self,
        paths: Sequence[str] | None = None,
        *,
        namespace: str = "",
    ) -> IngestionResult:
        return self.index_manager.prepare(paths, namespace=namespace)

    def create_prepared_index(self, ingestion: IngestionResult) -> IndexRef:
        return self.index_manager.build_prepared(ingestion)

    def clear_session(self, session_id: str) -> bool:
        return self.memory.clear(session_id)

    def close(self) -> None:
        """Release provider connection pools exactly once per adapter."""

        closed: set[int] = set()
        for provider in (self.chat_model, self.web_search, self.query_planner):
            if provider is None or id(provider) in closed:
                continue
            closed.add(id(provider))
            close_provider = getattr(provider, "close", None)
            if callable(close_provider):
                close_provider()

    def answer(self, index_id: str, request: AnswerRequest) -> AnswerResult:
        started = self.timer()
        trace_id = uuid4().hex[:16]
        question = (request.question or "").strip()
        if not question:
            raise ValueError("请输入问题。")
        if len(question) > self.settings.max_question_characters:
            raise ValueError(f"问题不能超过 {self.settings.max_question_characters} 个字符。")

        history_turns = len(self.memory.history(request.session_id))
        retrieval_query = self._retrieval_query(request.session_id, question)
        query_plan, planning_error = self._query_plan(question, request)
        retrieval_queries = (retrieval_query, *query_plan[1:])
        with self.index_manager.lease(index_id) as retriever:
            hits = self._retrieve_queries(
                retriever,
                retrieval_queries,
                deep_research=request.deep_research,
            )
        decision = self.routing.decide(hits, allow_web=request.allow_web)
        if (
            request.deep_research
            and request.allow_web
            and self.web_search.available
            and decision.route is Route.LOCAL
        ):
            decision = RouteDecision(
                Route.HYBRID,
                decision.confidence,
                "研究模式将用网络来源补充本地证据。",
            )
        selected_hits = hits if decision.route in {Route.LOCAL, Route.HYBRID} else ()
        citations, evidence = self._local_evidence(selected_hits)
        web_error = ""
        web_domain_count = 0
        web_query_count = 0

        if decision.route in {Route.WEB, Route.HYBRID}:
            if not request.allow_web or not self.web_search.available:
                decision = RouteDecision(Route.REFUSED, decision.confidence, "联网搜索未开启或不可用。")
            else:
                web_results_list: list[WebSearchResult] = []
                web_queries = (
                    query_plan[: self.settings.research_max_web_queries]
                    if request.deep_research
                    else (question[:70],)
                )
                for web_query in web_queries:
                    try:
                        web_results_list.extend(
                            self.web_search.search(web_query[:70], count=5)
                        )
                        web_query_count += 1
                    except (ProviderError, ValueError) as error:
                        web_error = type(error).__name__
                web_results = tuple(web_results_list)
                web_citations, web_evidence, web_domain_count = self._web_evidence(
                    question, web_results
                )
                citations = (*citations, *web_citations)
                evidence = (*evidence, *web_evidence)
                if not web_results and not selected_hits:
                    decision = RouteDecision(Route.REFUSED, decision.confidence, "没有找到足够的可靠证据。")
                elif not web_results and selected_hits:
                    decision = RouteDecision(Route.LOCAL, decision.confidence, "网络来源不可用，保留本地证据。")

        if not evidence:
            result = self._result(
                answer="现有资料不足以回答这个问题。",
                decision=decision,
                citations=(),
                hits=hits,
                trace_id=trace_id,
                started=started,
                diagnostics={
                    "web_error": web_error,
                    "evidence_count": 0,
                    "history_turns": history_turns,
                    "planned_query_count": len(query_plan),
                    "planning_error": planning_error,
                    "web_query_count": web_query_count,
                },
            )
            return self._remember_result(request, question, result)

        claims: tuple[AnswerClaim, ...] = ()
        grounding_claim_count = 0
        grounding_citation_count = 0
        if not request.allow_cloud or not self.chat_model.available:
            retrieval_decision = RouteDecision(
                Route.RETRIEVAL_ONLY,
                decision.confidence,
                "云端生成未开启，仅展示检索到的证据。",
            )
            answer = self._retrieval_only_answer(citations)
        else:
            try:
                generated_answer = self.chat_model.answer(question, evidence)
                allowed_ids = tuple(citation.citation_id for citation in citations)
                grounding_audit = validate_grounded_answer(generated_answer, allowed_ids)
                answer = render_grounded_answer(generated_answer)
                claims = generated_answer.claims
                grounding_claim_count = grounding_audit.claim_count
                grounding_citation_count = grounding_audit.citation_count
            except (ProviderError, GroundingContractError) as error:
                result = self._result(
                    answer="生成服务暂时不可用。你仍可以查看下方检索证据。",
                    decision=RouteDecision(Route.ERROR, decision.confidence, "生成服务调用失败。"),
                    citations=citations,
                    hits=hits,
                    trace_id=trace_id,
                    started=started,
                    diagnostics={
                        "provider_error": type(error).__name__,
                        "evidence_count": len(evidence),
                        "history_turns": history_turns,
                        "planned_query_count": len(query_plan),
                        "planning_error": planning_error,
                        "web_query_count": web_query_count,
                    },
                )
                return self._remember_result(request, question, result)
            retrieval_decision = decision

        result = self._result(
            answer=answer,
            decision=retrieval_decision,
            claims=claims,
            citations=citations,
            hits=hits,
            trace_id=trace_id,
            started=started,
            diagnostics={
                "evidence_count": len(evidence),
                "grounded_claim_count": grounding_claim_count,
                "grounding_citation_count": grounding_citation_count,
                "citation_completeness": 1.0 if claims else 0.0,
                "web_error": web_error,
                "web_domain_count": web_domain_count,
                "history_turns": history_turns,
                "planned_query_count": len(query_plan),
                "planning_error": planning_error,
                "web_query_count": web_query_count,
            },
        )
        return self._remember_result(request, question, result)

    def _query_plan(
        self,
        question: str,
        request: AnswerRequest,
    ) -> tuple[tuple[str, ...], str]:
        planned: Sequence[str] = ()
        planning_error = ""
        if request.deep_research and request.allow_cloud:
            if self.query_planner is None or not self.query_planner.available:
                planning_error = "planner_unavailable"
            else:
                try:
                    planned = self.query_planner.plan_queries(
                        question,
                        max_queries=self.settings.research_max_queries,
                    )
                except (ProviderError, ValueError) as error:
                    planning_error = type(error).__name__
        return (
            normalize_query_plan(
                question,
                planned,
                max_queries=self.settings.research_max_queries,
            ),
            planning_error,
        )

    def _retrieve_queries(
        self,
        retriever: Retriever,
        queries: Sequence[str],
        *,
        deep_research: bool,
    ) -> tuple[SearchHit, ...]:
        if not deep_research or len(queries) == 1:
            return tuple(
                retriever.search(
                    queries[0],
                    top_k=self.settings.final_evidence_count,
                )
            )
        rankings = {
            f"query-{index}": tuple(
                retriever.search(query, top_k=self.settings.final_evidence_count)
            )
            for index, query in enumerate(queries, start=1)
        }
        return fuse_query_hits(rankings, top_k=self.settings.final_evidence_count)

    def _retrieval_query(self, session_id: str, question: str) -> str:
        """Add recent user questions without trusting prior generated answers."""

        turns = self.memory.history(session_id)
        if not turns:
            return question
        prior_questions = [turn.question for turn in turns[-3:]]
        history = "\n".join(f"此前问题：{item}" for item in prior_questions)
        history = truncate_text(history, self.settings.retrieval_history_characters)
        return f"当前问题：{question}\n{history}"

    def _remember_result(
        self,
        request: AnswerRequest,
        question: str,
        result: AnswerResult,
    ) -> AnswerResult:
        self.memory.add_turn(request.session_id, question, result.answer)
        return result

    def _local_evidence(
        self, hits: Sequence[SearchHit]
    ) -> tuple[tuple[Citation, ...], tuple[tuple[str, str], ...]]:
        citations: list[Citation] = []
        evidence: list[tuple[str, str]] = []
        for index, hit in enumerate(hits, start=1):
            citation_id = f"L{index}"
            heading = f" / {hit.chunk.heading}" if hit.chunk.heading else ""
            evidence_text = f"来源：{hit.chunk.source_name}{heading}\n{hit.chunk.text}"
            citations.append(
                Citation(
                    citation_id=citation_id,
                    source_name=hit.chunk.source_name,
                    excerpt=truncate_text(hit.chunk.text, 900),
                    score=hit.score,
                )
            )
            evidence.append((citation_id, evidence_text))
        return tuple(citations), tuple(evidence)

    def _web_evidence(
        self, question: str, results: Sequence[WebSearchResult]
    ) -> tuple[tuple[Citation, ...], tuple[tuple[str, str], ...], int]:
        ranked = rank_web_results(question, results, limit=5, per_domain=2)
        citations: list[Citation] = []
        evidence: list[tuple[str, str]] = []
        for index, ranked_item in enumerate(ranked, start=1):
            item = ranked_item.result
            citation_id = f"W{index}"
            content = item.content or item.title
            citations.append(
                Citation(
                    citation_id=citation_id,
                    source_name=item.title,
                    excerpt=truncate_text(content, 900),
                    url=item.url,
                    score=ranked_item.score,
                )
            )
            evidence.append((citation_id, f"标题：{item.title}\n内容：{content}\n链接：{item.url}"))
        domain_count = len({item.domain for item in ranked if item.domain != "unknown"})
        return tuple(citations), tuple(evidence), domain_count

    @staticmethod
    def _retrieval_only_answer(citations: Sequence[Citation]) -> str:
        lines = ["云端生成未开启，以下是检索到的相关证据："]
        lines.extend(
            f"- [{citation.citation_id}] {citation.source_name}：{citation.excerpt}"
            for citation in citations
        )
        return "\n\n".join(lines)

    def _result(
        self,
        *,
        answer: str,
        decision: RouteDecision,
        citations: Sequence[Citation],
        hits: Sequence[SearchHit],
        trace_id: str,
        started: float,
        diagnostics: dict[str, float | int | str],
        claims: Sequence[AnswerClaim] = (),
    ) -> AnswerResult:
        latency_ms = max(0.0, (self.timer() - started) * 1_000)
        return AnswerResult(
            answer=answer,
            decision=decision,
            claims=tuple(claims),
            citations=tuple(citations),
            hits=tuple(hits),
            trace_id=trace_id,
            latency_ms=latency_ms,
            diagnostics=diagnostics,
        )
