# RAG Search System

一个支持本地文档检索与联网搜索回退的 RAG（检索增强生成）问答系统。

用户可以上传 `.txt` 或 `.md` 文档并提问。系统首先使用本地 Embedding 模型和 ChromaDB 检索相关片段；当本地资料相关度不足时，自动调用智谱 Web Search API，并使用 GLM 生成带来源的回答。

## 功能

- 上传 TXT / Markdown 知识文档
- 递归文本切分与重叠片段
- 中文 Embedding 向量化
- ChromaDB 语义检索
- 本地知识库优先的问答
- 低相关度时自动联网搜索
- 展示本地片段或网页来源
- Gradio Web 界面

## 技术栈

- Python 3.11
- LangChain Text Splitters
- `BAAI/bge-small-zh-v1.5`
- ChromaDB
- 智谱 GLM API
- 智谱 Web Search API
- Gradio

## 工作流程

```text
上传文档 -> 文本切分 -> Embedding -> ChromaDB
                                      |
用户问题 -> 向量检索 -> 判断相关度 ---+
                         |             |
                    相关度足够      相关度不足
                         |             |
                    本地资料回答    联网搜索
                         |             |
                         +----> GLM 生成回答 -> 展示答案和来源
```

## 安装

```powershell
git clone https://github.com/XWsight/RAG-Search-System.git
cd RAG-Search-System

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

首次运行会下载约 100 MB 的中文 Embedding 模型。

## 配置

复制环境变量示例：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
ZHIPU_API_KEY=你的智谱APIKey
ZHIPU_MODEL=glm-5.2
```

`.env` 已加入 `.gitignore`，请勿把真实密钥提交到仓库。

## 运行

```powershell
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = "127.0.0.1,localhost"
python app.py
```

打开浏览器访问：

```text
http://127.0.0.1:7860
```

不上传文件时，系统使用 `data/knowledge.txt` 作为示例知识库。

## 项目结构

```text
RAG-Search-System/
|-- app.py                 # Gradio Web 应用
|-- rag_search_demo.py     # 本地检索、路由、联网搜索与模型调用
|-- data/
|   |-- knowledge.txt      # 示例知识文档
|-- src/
|   |-- rag_demo.py        # 早期最简 RAG 原型
|-- .env.example           # 环境变量示例
|-- .gitignore
|-- requirements.txt
|-- README.md
```

## 当前版本

V1.0：完成本地文档 RAG、自动联网搜索、来源展示和 Gradio 界面。

后续计划包括多文档持久化知识库、更可靠的路由阈值、检索评估与重排序。
