from pathlib import Path
from uuid import uuid4

import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_search_demo import call_model, search_web


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)


def build_store(file_path):
    if file_path:
        text = Path(file_path).read_text(encoding="utf-8")
        source_name = Path(file_path).name
    else:
        text = Path("data/knowledge.txt").read_text(encoding="utf-8")
        source_name = "data/knowledge.txt"

    if not text.strip():
        raise ValueError("上传的文档为空，请选择包含文本内容的文件。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_text(text)

    store = Chroma(
        collection_name=f"upload_{uuid4().hex}",
        embedding_function=embeddings,
    )

    store.add_texts(
        texts=chunks,
        ids=[f"chunk-{index}" for index in range(len(chunks))],
    )

    return store, source_name


def answer_question(file_path, question):
    if not question.strip():
        return "请输入问题", "", ""

    store, source_name = build_store(file_path)

    local_results = store.similarity_search_with_score(
        question,
        k=2,
    )

    best_score = local_results[0][1]

    if best_score <= 1.0:
        context = "\n\n".join(
            document.page_content
            for document, _ in local_results
        )

        answer = call_model(question, context)
        route = f"本地知识库：{source_name}"

        sources = "\n\n".join(
            f"{index}. 距离 {score:.3f}\n{document.page_content}"
            for index, (document, score)
            in enumerate(local_results, start=1)
        )

    else:
        web_results = search_web(question)

        context = "\n\n".join(
            f"[来源 {index}] {item.get('title')}\n"
            f"{item.get('content')}\n"
            f"链接：{item.get('link')}"
            for index, item
            in enumerate(web_results, start=1)
        )

        answer = call_model(question, context)
        route = "本地资料不足，已使用网络搜索"

        sources = "\n\n".join(
            f"{index}. {item.get('title')}\n{item.get('link')}"
            for index, item
            in enumerate(web_results, start=1)
        )

    return answer, sources, route


file_input = gr.File(
    label="上传知识文档（可选）",
    file_types=[".txt", ".md"],
    type="filepath",
)

question_input = gr.Textbox(
    label="请输入问题",
    placeholder="例如：这个文档主要讲了什么？",
)

demo = gr.Interface(
    fn=answer_question,
    inputs=[file_input, question_input],
    outputs=[
        gr.Markdown(label="回答"),
        gr.Markdown(label="来源"),
        gr.Textbox(label="检索路径"),
    ],
    title="RAG Search System",
    description="上传文档后提问；本地资料不足时自动联网搜索",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
