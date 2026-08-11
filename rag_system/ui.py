"""Gradio interface kept separate from retrieval and provider logic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from rag_system.config import Settings
from rag_system.domain import AnswerRequest, AnswerResult, Route
from rag_system.security import markdown_text, safe_external_url
from rag_system.service import RagService


_ROUTE_LABELS = {
    Route.LOCAL: "本地知识库",
    Route.WEB: "网络来源",
    Route.HYBRID: "本地 + 网络",
    Route.RETRIEVAL_ONLY: "仅检索模式",
    Route.REFUSED: "证据不足",
    Route.ERROR: "服务降级",
}


def create_demo(service: RagService, settings: Settings):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("缺少 Gradio，请先安装项目依赖。") from error

    def build_index(
        files: Sequence[str] | str | None,
        session_id: str,
    ) -> tuple[str, str, dict[str, Any], list[dict[str, Any]], str, str, dict[str, Any]]:
        paths = _file_paths(files)
        try:
            index_ref = service.create_index(paths or None)
        except Exception:
            return (
                "",
                "索引失败：请检查文档格式、大小和内容后重试。",
                {"status": "error"},
                [],
                "",
                "",
                {},
            )
        service.clear_session(session_id)
        source_label = "示例知识库" if not paths else f"{len(paths)} 个上传文档"
        status = (
            f"索引已就绪：{source_label}，共 {index_ref.document_count} 个文档、"
            f"{index_ref.chunk_count} 个片段。"
        )
        return (
            index_ref.index_id,
            status,
            {
                "index_id": index_ref.index_id,
                "documents": index_ref.document_count,
                "chunks": index_ref.chunk_count,
            },
            [],
            "",
            "",
            {},
        )

    def ask(
        index_id: str,
        session_id: str,
        question: str,
        history: list[dict[str, Any]] | None,
        allow_cloud: bool,
        allow_web: bool,
        deep_research: bool,
    ) -> tuple[str, list[dict[str, Any]], str, str, dict[str, Any], str]:
        chat_history = list(history or [])
        clean_question = (question or "").strip()
        if not clean_question:
            return index_id, chat_history, "", "请输入问题。", {}, ""

        try:
            active_index = index_id
            if not active_index:
                active_index = service.create_index().index_id
            result = service.answer(
                active_index,
                AnswerRequest(
                    question=clean_question,
                    session_id=session_id,
                    allow_cloud=bool(allow_cloud),
                    allow_web=bool(allow_web),
                    deep_research=bool(deep_research),
                ),
            )
        except Exception:
            message = "处理失败：请检查输入或稍后重试。"
            chat_history.extend(
                [
                    {"role": "user", "content": clean_question},
                    {"role": "assistant", "content": message},
                ]
            )
            return index_id, chat_history, "", "处理失败", {"status": "error"}, ""

        chat_history.extend(
            [
                {"role": "user", "content": clean_question},
                {"role": "assistant", "content": result.answer},
            ]
        )
        route = (
            f"{_ROUTE_LABELS[result.decision.route]} · "
            f"置信度 {result.decision.confidence:.0%} · {result.decision.reason}"
        )
        diagnostics = {
            "trace_id": result.trace_id,
            "route": result.decision.route.value,
            "confidence": round(result.decision.confidence, 4),
            "latency_ms": round(result.latency_ms, 1),
            **result.diagnostics,
        }
        return active_index, chat_history, _sources_markdown(result), route, diagnostics, ""

    def clear_chat(session_id: str) -> tuple[list[dict[str, Any]], str, str, dict[str, Any]]:
        service.clear_session(session_id)
        return [], "", "", {}

    with gr.Blocks(title="RAG Studio") as demo:
        index_state = gr.State(value="", time_to_live=settings.session_ttl_seconds)
        session_state = gr.State(
            value=lambda: uuid4().hex,
            time_to_live=settings.session_ttl_seconds,
            delete_callback=service.clear_session,
        )

        gr.Markdown(
            """
# RAG Studio

多文档知识检索、关键词与向量融合、置信度路由和可核验引用。
默认不会发送文档内容或问题到外部服务；需要时请主动开启对应选项。
"""
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                files = gr.File(
                    label="知识文档",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".pdf"],
                    type="filepath",
                    height=180,
                )
                build_button = gr.Button("建立知识索引", variant="primary")
                index_status = gr.Markdown("尚未建立索引；直接提问将使用示例知识库。")
                allow_cloud = gr.Checkbox(
                    label="允许将检索证据发送到云端生成回答",
                    value=settings.allow_cloud_default,
                )
                allow_web = gr.Checkbox(
                    label="允许将问题发送到联网搜索",
                    value=settings.allow_web_default,
                )
                deep_research = gr.Checkbox(
                    label="研究模式：多查询检索并在允许时补充多个网络来源",
                    value=False,
                )
                gr.Markdown(
                    "研究模式有严格预算，不会无限循环；它会增加模型与搜索调用次数。"
                )
                gr.Markdown("开启上述选项代表你同意将相应内容发送给配置的第三方服务。")
                index_details = gr.JSON(label="索引信息", value={})

            with gr.Column(scale=2, min_width=500):
                chatbot = gr.Chatbot(
                    label="问答",
                    height=480,
                    placeholder="建立索引后开始提问",
                    sanitize_html=True,
                    render_markdown=True,
                    buttons=["copy"],
                )
                with gr.Row():
                    question = gr.Textbox(
                        label="问题",
                        placeholder="例如：混合检索为什么比单独向量检索更可靠？",
                        lines=2,
                        max_lines=5,
                        scale=5,
                    )
                    ask_button = gr.Button("提问", variant="primary", scale=1)
                clear_button = gr.Button("清空对话")

        with gr.Row():
            route_output = gr.Markdown(label="路由决策")
            diagnostics_output = gr.JSON(label="运行诊断", value={})
        sources_output = gr.Markdown(label="来源与证据")

        build_button.click(
            fn=build_index,
            inputs=[files, session_state],
            outputs=[
                index_state,
                index_status,
                index_details,
                chatbot,
                sources_output,
                route_output,
                diagnostics_output,
            ],
        )
        ask_inputs = [
            index_state,
            session_state,
            question,
            chatbot,
            allow_cloud,
            allow_web,
            deep_research,
        ]
        ask_outputs = [index_state, chatbot, sources_output, route_output, diagnostics_output, question]
        ask_button.click(fn=ask, inputs=ask_inputs, outputs=ask_outputs)
        question.submit(fn=ask, inputs=ask_inputs, outputs=ask_outputs)
        clear_button.click(
            fn=clear_chat,
            inputs=[session_state],
            outputs=[chatbot, sources_output, route_output, diagnostics_output],
        )

    return demo.queue(max_size=32, default_concurrency_limit=2)


def _file_paths(files: Sequence[str] | str | None) -> list[str]:
    if files is None:
        return []
    if isinstance(files, str):
        return [files]
    return [str(path) for path in files if path]


def _sources_markdown(result: AnswerResult) -> str:
    if not result.citations:
        return "没有可展示的来源。"
    sections: list[str] = []
    for citation in result.citations:
        title = markdown_text(citation.source_name, max_characters=300)
        excerpt = markdown_text(citation.excerpt, max_characters=1_200)
        score = f" · 相关度 {citation.score:.0%}" if citation.score is not None else ""
        section = f"### [{citation.citation_id}] {title}{score}\n\n> {excerpt.replace(chr(10), chr(10) + '> ')}"
        url = safe_external_url(citation.url)
        if url:
            section += f"\n\n[打开来源]({url})"
        sections.append(section)
    return "\n\n---\n\n".join(sections)
