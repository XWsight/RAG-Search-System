"""V1.0 Day 1：最简 RAG —— 本地文档检索问答

流程：
1. 读取 data/ 下的 md/txt 文档
2. 切成小块（chunk）
3. 用本地 Embedding 模型把每块转成向量
4. 提问 -> 转成向量 -> 找最相似的几块
5. 有 LLM API Key 就用大模型生成答案，否则直接展示检索到的原文

运行：
    python src/rag_demo.py
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
TOP_K = 3


def load_documents():
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"读取到 {len(docs)} 个文档")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"切分成 {len(chunks)} 个片段")
    return chunks


def build_index(chunks, embeddings):
    """把每个片段变成向量，堆成矩阵。"""
    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    return np.array(vectors)


def retrieve(query, embeddings, matrix, chunks, top_k=TOP_K):
    """余弦相似度检索最相似的 top_k 个片段。"""
    qv = np.array(embeddings.embed_query(query))
    scores = matrix @ qv / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(qv) + 1e-9)
    idx = scores.argsort()[::-1][:top_k]
    return [(chunks[i].page_content, float(scores[i])) for i in idx]


def generate_answer(query, hits):
    """有 API Key 时用大模型生成答案，否则返回 None。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    context = "\n\n".join(f"[片段{i + 1}]\n{text}" for i, (text, _) in enumerate(hits))
    prompt = (
        "请只根据下面的资料回答用户问题；如果资料里没有答案，请直接说"
        f"「资料中没有找到」。\n\n资料：\n{context}\n\n问题：{query}"
    )
    return llm.invoke(prompt).content


def main():
    print(f"Embedding 模型：{EMBEDDING_MODEL}（首次运行会自动下载）")
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("正在生成向量索引……")
    matrix = build_index(chunks, embeddings)
    print("索引完成。开始提问吧（输入 exit 退出）")

    while True:
        query = input("\n问题：").strip()
        if query.lower() in {"exit", "quit"}:
            break
        hits = retrieve(query, embeddings, matrix, chunks)
        answer = generate_answer(query, hits)
        if answer:
            print("\n答案：", answer)
        else:
            print("\n（未配置 LLM API Key，先展示检索到的原文）")
        for i, (text, score) in enumerate(hits, 1):
            print(f"\n[{i}] 相关度 {score:.3f}\n{text[:200]}")


if __name__ == "__main__":
    main()
