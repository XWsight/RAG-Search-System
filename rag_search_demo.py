import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
MODEL = os.getenv("ZHIPU_MODEL", "glm-5.2")
CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"


def build_vector_store():
    text = Path("data/knowledge.txt").read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    store = Chroma(
        collection_name="rag_search_v1",
        embedding_function=embeddings,
        persist_directory="./rag_chroma_db",
    )

    old_data = store.get()
    if old_data["ids"]:
        store.delete(ids=old_data["ids"])

    store.add_texts(
        texts=chunks,
        ids=[f"chunk-{index}" for index in range(len(chunks))],
    )

    return store


def call_model(question, context):
    if not API_KEY:
        return "未找到 ZHIPU_API_KEY，请先在 .env 中配置智谱 API Key。"

    prompt = f"""
请根据下面提供的资料回答问题。
如果资料没有答案，请说“资料中没有找到答案”。
不要凭空补充信息。

资料：
{context}

问题：
{question}
"""

    response = requests.post(
        CHAT_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )

    if not response.ok:
        return f"模型调用失败：{response.text[:300]}"

    return response.json()["choices"][0]["message"]["content"]


def search_web(question):
    if not API_KEY:
        return []

    response = requests.post(
        SEARCH_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "search_query": question,
            "search_engine": "search_std",
            "search_intent": False,
            "count": 3,
            "content_size": "medium",
        },
        timeout=60,
    )

    if not response.ok:
        return []

    return response.json().get("search_result", [])


def main():
    question = input("请输入问题：")

    store = build_vector_store()
    local_results = store.similarity_search_with_score(question, k=2)

    best_score = local_results[0][1]
    print(f"\n本地最佳距离：{best_score:.3f}")

    if best_score <= 1.0:
        print("路由：使用本地知识库")
        context = "\n\n".join(
            document.page_content
            for document, _ in local_results
        )
        answer = call_model(question, context)

    else:
        print("路由：本地资料不足，使用网络搜索")
        web_results = search_web(question)

        context = "\n\n".join(
            f"[来源 {index}] {item.get('title')}\n"
            f"{item.get('content')}\n"
            f"链接：{item.get('link')}"
            for index, item in enumerate(web_results, start=1)
        )

        answer = call_model(question, context)

        print("\n搜索来源：")
        for index, item in enumerate(web_results, start=1):
            print(f"[来源 {index}] {item.get('link')}")

    print(f"\n回答：\n{answer}")


if __name__ == "__main__":
    main()
