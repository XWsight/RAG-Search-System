# RAG-Search-System

带搜索功能的 RAG 问答系统（V1.0，Agent 雏形）。

## 目标

- 能回答基于本地文档的问题
- 本地找不到答案时，自动调用搜索引擎查找并总结
- 回答带引用来源
- 可演示的界面 + 完整 GitHub 仓库

## 当前进度

- [x] Day 1：项目骨架 + 环境
- [x] 最简 RAG：读本地文档 → 检索问答（未配 LLM Key 时展示检索原文）
- [ ] 接入搜索 API，实现智能路由
- [ ] Streamlit 界面 + 引用标注
- [ ] 文档整理 + 上传

## 快速开始

```powershell
# 1. 创建虚拟环境（项目根目录下）
python -m venv .venv

# 2. 激活并安装依赖
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. 运行最简 RAG（首次运行会自动下载中文 Embedding 模型，约 100MB）
python src/rag_demo.py
```

> 提示：如果 Windows 报「文件名或扩展名太长」（WinError 206），把虚拟环境建到短路径，
> 例如 `python -m venv C:\Users\你的用户名\Documents\Codex\rag-venv`，再用它的 python.exe 运行。

## 配置 LLM（可选）

复制 `.env.example` 为 `.env`，填入 OpenAI 兼容的 API Key（DeepSeek / OpenAI 均可）。
不配置也能运行——此时会直接展示检索到的原文，方便你理解 RAG 的「检索」环节。
