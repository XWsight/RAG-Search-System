# RAG Studio

[![quality](https://github.com/XWsight/RAG-Search-System/actions/workflows/quality.yml/badge.svg?branch=rag-studio)](https://github.com/XWsight/RAG-Search-System/actions/workflows/quality.yml?query=branch%3Arag-studio)

面向中文知识库的隐私优先 RAG 系统：支持安全的多格式文档导入、向量与 BM25 混合检索、RRF 融合、可选重排序、证据路由、联网补充、带引用回答和有预算上限的研究模式。

仓库提供三个入口：面向普通用户的同源 Web 工作台、适合本地实验的 Gradio 工作台，以及具备租户隔离、鉴权、后台索引任务、持久化目录、限流、指标和安全错误协议的 FastAPI 服务。`main` 保留原始 V1.0；当前增强版位于 `rag-studio` 分支。

Web 工作台覆盖知识库创建、异步索引进度、知识库切换与删除、多轮问答、引用证据、检索路径，以及云端生成和联网搜索的请求级授权。访问密钥只保存在当前浏览器标签会话中，不会写入仓库或持久化到浏览器长期存储。

> 当前生产形态是 **durable single-node**，不是多副本高可用集群。SQLite、Chroma 和上传原文必须位于同一个持久卷，API 只能运行一个 Uvicorn worker。能力边界详见[部署说明](docs/deployment.md)。

## 核心能力

| 领域 | 已实现能力 |
| --- | --- |
| 文档摄取 | TXT、Markdown、HTML、DOCX、PDF；大小、页数、字符数、压缩包膨胀和路径安全边界 |
| 检索 | 中文 Embedding、Chroma 余弦检索、BM25、RRF 融合、来源多样化、可选 CrossEncoder 重排序 |
| 回答 | 本地 / 混合 / 网络 / 拒答路由；请求级云端与联网授权；原子结论—证据结构化契约 |
| 研究模式 | 有界查询拆解、多查询检索融合、多次网络补充；无无限 ReAct 循环 |
| 会话 | TTL、LRU、轮数与字符数上限；租户、知识库和浏览器会话三重隔离 |
| 数据生命周期 | 原子上传、SHA-256 清单校验、持久索引复用、残缺集合重建、耐久取消意图、显式删除 |
| 产品界面 | 同源 Web 工作台、拖放上传、任务进度、知识库管理、多轮问答、引用证据与隐私开关 |
| 服务边界 | API Key / Bearer、reader / writer / operator、租户隔离、持久幂等、后台任务、限流 |
| 可运维性 | liveness/readiness、隐私安全 JSON 事件、Prometheus 指标、Docker Compose、备份恢复手册 |
| 质量 | 检索 Recall/MRR/nDCG、回答事实/拒答/原子性/归因、逐题诊断、冻结质量门禁、Python 3.11/3.12 CI |

## 架构

```mermaid
flowchart LR
    C["Web app / Gradio / REST client"] --> A["API boundary\nauth · roles · rate limit"]
    A --> P["Application platform\ntenant catalog · jobs · idempotency"]
    C --> S["RAG service"]
    P --> S
    P --> F["Tenant file store"]
    P --> Q["SQLite catalog"]
    S --> I["Ingestion\nsecure loaders · adaptive chunks"]
    S --> R["Hybrid retrieval\ndense · BM25 · RRF · rerank"]
    R --> V["Persistent Chroma"]
    S --> M["Bounded conversation memory"]
    S --> Z["Chat / web providers\nexplicit outbound consent"]
```

索引身份包含租户命名空间、文档内容、切分参数和 Embedding 模型。同一知识库在进程重启后可重新挂载已有向量；不同租户上传完全相同的文件也不会共享索引身份。

更完整的数据流、模块边界和非目标见[架构文档](docs/architecture.md)。

## 本地工作台

要求 Python 3.11 或 3.12。Windows PowerShell：

```powershell
git clone https://github.com/XWsight/RAG-Search-System.git
cd RAG-Search-System
git switch rag-studio

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

如果需要云端生成或联网搜索，再在 `.env` 中配置：

```dotenv
ZHIPU_API_KEY=你的智谱密钥
ZHIPU_MODEL=glm-5.2
```

不配置密钥也能使用本地检索模式。`RAG_ALLOW_CLOUD_DEFAULT` 与 `RAG_ALLOW_WEB_DEFAULT` 只控制 Gradio 界面的初始勾选状态；REST API 始终要求每次请求显式授权。关闭时，问题与文档证据不会被发送到相应外部服务。

运行：

```powershell
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = "127.0.0.1,localhost"
python app.py
```

访问 `http://127.0.0.1:7860`。首次使用默认 Embedding 模型时需要下载模型文件。

## 持久化 API

在 `.env` 中使用高熵 API Key 配置租户主体，并启用持久化：

```dotenv
RAG_PERSIST_DATA=true
RAG_STORAGE_ROOT=.rag_data
RAG_API_KEYS_JSON={"替换为至少16字符的随机密钥":{"subject":"local-admin","tenant_id":"local","roles":["reader","writer","operator"]}}
```

启动单 worker 服务：

```powershell
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000 --workers 1
```

主要端点：

| 方法 | 路径 | 角色 | 用途 |
| --- | --- | --- | --- |
| `GET` | `/health/live`, `/health/ready` | 公开 | 存活与就绪检查 |
| `POST` | `/v1/knowledge-bases` | writer | 上传并异步建立知识库，需要 `Idempotency-Key` |
| `GET` | `/v1/knowledge-bases` | reader | 列出本租户知识库 |
| `GET/DELETE` | `/v1/knowledge-bases/{id}` | reader / writer | 查询或完整删除资源 |
| `GET/DELETE` | `/v1/jobs/{id}` | reader / writer | 查询或取消后台任务 |
| `POST` | `/v1/answers` | reader | 检索、路由并回答 |
| `DELETE` | `/v1/knowledge-bases/{id}/sessions/{session}` | writer | 清除一段对话记忆 |
| `GET` | `/metrics` | operator | Prometheus 文本指标 |

每个受保护请求只能提供一种认证方式：`X-API-Key` 或 `Authorization: Bearer`。`Idempotency-Key` 必须是 8–128 个可打印 ASCII 字符；在 24 小时有效窗口内，相同键与相同请求会返回原资源，同一键绑定不同请求会安全地返回冲突。窗口过期后，该键会被视为新的创建请求。

索引取消采用两层状态：平台会先把可取消知识库的 Catalog 状态从 `PENDING`/`INDEXING` 耐久写为 `CANCELLING`，成功后才向进程内 worker 发出取消信号。意图写入失败时 API 返回 `503`、worker 不会收到信号，调用方应使用同一 job ID 重试。排队 job 可能立即变为 `cancelled`；运行中 job 先变为 `cancelling`，只有 worker 实际退出后才释放任务容量。知识库通常收敛为 `FAILED`/`index_cancelled`；若终态写入中断，启动恢复或同一创建请求的幂等重放会完成收敛。旧 worker 不会恢复执行，但其快照在有界保留期内仍可查询；必要时重放会轮换并绑定一个新的可轮询 job。

如果知识库已经耐久提交为 `READY`，取消请求已经太迟：Catalog 保持 `READY`，job 可能短暂显示 `cancelling`，但已完成的任务最终记为 `succeeded`。Catalog 当前使用 schema v4：上传先进入 `PREPARING`，绑定一次性不可变清单，全部文件落盘核验后才进入 `PENDING`。启动时会事务化迁移 schema v2/v3；升级前仍须停写备份，旧程序不能直接读取迁移后的 v4 Catalog。

`ready` 验证文档存储根、Catalog、向量目录、任务执行器和耐久 job 快照库；它不加载可选模型、不执行 Chroma 查询，也不探测外部供应商。调用方仍需处理单次检索、模型下载或上游服务失败。

正式 Web 工作台位于 `http://127.0.0.1:8000/app`，根路径会自动跳转到该页面。OpenAPI 位于 `http://127.0.0.1:8000/docs`；生产环境可设置 `RAG_API_DOCS_ENABLED=false`。

## Docker 部署

```bash
cp .env.example .env
# 编辑 .env 后：
docker compose config --quiet
docker compose build --pull
docker compose up -d
curl --fail http://127.0.0.1:8000/health/ready
```

容器健康后，浏览器访问 `http://127.0.0.1:8000/app`。首次进入时粘贴 `.env` 中配置的租户 API Key；密钥仅在当前标签会话中保存。

Compose 默认仅发布到 `127.0.0.1`，使用非 root 用户、只读容器根文件系统、最小权限、资源上限、日志轮转和 `/data` 持久卷。公网访问必须由可信反向代理终止 TLS。备份、恢复、升级、回滚、密钥轮换和删除要求见[部署说明](docs/deployment.md)与[运维手册](docs/operations.md)。

## 验证与评测

```powershell
python -m unittest discover -s tests -v
python -m compileall -q rag_system tests scripts
python scripts\benchmark_sparse.py evals\retrieval_cases.jsonl `
  evals\corpus\rag.md evals\corpus\retrieval.md `
  evals\corpus\safety.md evals\corpus\storage.md `
  --top-k 5 `
  --quality-gate evals\gates\bm25-smoke.json `
  --json-output reports\bm25-smoke.json `
  --markdown-output reports\bm25-smoke.md
```

当前 ground truth 摘要 `74fe19194ca06876` 的 18 题开发集上，依赖无关 BM25 冒烟基线为 Recall@5 `1.0000`、MRR@5 `0.9583`、nDCG@5 `0.9692`、路由准确率 `0.8333`。这个小型、仓库内开发集只用于回归，**不能外推为真实业务效果**，引用指标也不适用于该检索集。

报告同时列出每题的缺失相关来源、期望/实际路由、首个相关排名、置信度和延迟，并汇总 P50/P95/P99。冻结门禁会校验数据集摘要、top-k 和最低指标；依赖无关的 BM25 门禁由本地检查与 GitHub CI 自动执行。延迟受硬件和缓存影响，当前只报告、不作为跨机器硬门槛。

正式检索回归套件使用严格 manifest 管理 50 个语义家族、200 个问题、10 篇来源和 12 个类别。开发/验证/test 分段分别为 80/60/60 题，相关来源文档禁止跨分段复用；本地、拒答和联网路由分别为 160/20/20 题。校验器会拒绝规范化重复问题、缺失或越界来源、错误路由标签、来源泄漏和覆盖不足：

```powershell
python scripts\validate_retrieval_suite.py evals\retrieval_suite.json `
  --contract evals\gates\retrieval-suite.json
python scripts\benchmark_sparse.py evals\retrieval_suite.json `
  --top-k 5 `
  --quality-gate evals\gates\bm25-foundation.json `
  --json-output reports\bm25-foundation.json `
  --markdown-output reports\bm25-foundation.md
```

当前 suite/corpus bundle 摘要为 `128a3bedb786c8b4`，题目 ground truth 摘要为 `5dba420a8979a05d`。全语料 BM25 下限为 Recall@5 `0.9844`、MRR@5 `0.9568`、nDCG@5 `0.9592`、路由准确率 `0.8000`。路由分数暴露出词法置信度会把越界问题误判为本地可回答，属于下一轮需要改善的已知基线，不能被描述为目标成绩。`--split development|validation|test` 可以只运行一个来源隔离分段；同义问法用于鲁棒性覆盖，因此 200 题应同时报告为 50 个独立语义家族，不能包装成 200 个独立知识点。

真实混合检索需要安装运行依赖后执行：

```powershell
python scripts\benchmark_retrieval.py evals\retrieval_cases.jsonl `
  evals\corpus\rag.md evals\corpus\retrieval.md `
  evals\corpus\safety.md evals\corpus\storage.md `
  --top-k 5 `
  --quality-gate evals\gates\hybrid-development.json
```

在当前本地模型标识与默认配置下，当前 18 题 Hybrid 开发基线的 Recall@5、MRR@5、nDCG@5 和路由准确率均为 `1.0000`。该门禁需要下载并运行 Embedding 模型，因此是发布前手动门禁，不在默认 CI 中执行。该集合专门加入了语义改写、本地越界和必须联网的对抗题，仍然只是开发回归证据，不是生产质量或真实业务泛化证明。

200 题套件的首次真实 Hybrid 全语料基线为 Recall@5 `0.9844`、MRR@5 `0.9536`、nDCG@5 `0.9537`、路由准确率 `0.8850`，冻结在 `evals/gates/hybrid-foundation.json`。相比纯 BM25，路由改善但排序没有改善，证明当前融合仍有优化空间；该门禁只手动运行，不因结果较大就替代 18 题秒级回归。

云端生成不再把引用关系藏在自由文本中。模型必须返回由原子 `claims`、每条结论的 `citation_ids` 和明确的 `insufficient` 状态组成的 JSON；供应商适配器与应用服务会分别校验同一证据契约，API 同时返回可直接审计的 `claims` 映射和兼容展示用文本。该机制保证引用 ID 存在且每条生成结论都有归因，但**不等于语义蕴含或事实正确性已经被证明**。

结构化回答另有独立的真实模型基准，不复用检索成绩：

```powershell
python scripts\benchmark_answers.py evals\answer_cases.jsonl `
  --dotenv .env `
  --quality-gate evals\gates\answer-live.json `
  --json-output reports\answers-live.json `
  --markdown-output reports\answers-live.md
```

2026-08-11 的一次 4 题/8 原子事实开发运行中，结构契约、拒答、事实召回、原子结论和归因五项均为 `1.0000`。这是会受模型版本与随机性影响的手动发布冒烟结果；门禁最低值为 `0.75`，不在默认 CI 中执行，也不能外推为生产准确率。

评测协议、数据泄漏防范与阈值校准见[评测文档](docs/evaluation.md)，数据标注与 split 治理见[评测数据说明](evals/README.md)。仓库中的 `evals/sample_dataset.jsonl` 是评测格式夹具，不是系统性能证明。

## 项目结构

```text
rag_system/
  application.py      # 与 HTTP/CLI/Agent 无关的应用用例端口和稳定错误
  api.py              # REST 路由、鉴权、角色、限流和安全响应策略
  api_contract.py     # 版本化 HTTP schema 与领域到 wire 的投影
  api_errors.py       # 应用失败到公共 HTTP 错误的集中映射
  web_ui.py           # 同源产品界面的安全挂载与响应头
  web_ui/             # 知识库管理、任务进度、问答和引用展示前端
  platform.py         # 应用门面：多租户用例、幂等与恢复编排
  submission.py       # 上传物化、大小策略、请求摘要和文档 ID
  coordination.py     # 有界分片锁与双向 resource-job 登记
  assets.py           # 文档清单、存储结果、路径和内容一致性边界
  indexing.py         # 耐久索引状态机、取消语义和失败补偿
  catalog.py          # SQLite 知识库目录与状态机
  idempotency.py      # 持久化幂等 reservation
  file_store.py       # 租户隔离、原子且有界的上传存储
  ingestion.py        # 安全文档加载与确定性切分
  retrieval.py        # Chroma、混合检索、路由和索引持久化
  benchmark.py        # 检索指标、逐题诊断与延迟分位数
  benchmark_suite.py  # 评测家族、覆盖矩阵、来源隔离和泄漏校验
  quality_gate.py     # 绑定数据集的严格回归门禁
  answer_benchmark.py # 结构化回答事实、拒答、原子性与归因评测
  answer_quality_gate.py # 绑定回答数据集的手动发布门禁
  answer_protocol.py  # 可替换的提示、证据边界与严格输出解码协议
  provider_errors.py  # 应用层与外部服务适配器共享的稳定失败契约
  json_contract.py    # 拒绝重复键和非有限值的严格 JSON 边界
  service.py          # 问答、联网、研究模式、引用和会话编排
  grounding.py        # 原子结论、证据归因校验与稳定渲染
  metrics.py          # 有界 Prometheus 指标
  observability.py    # 不记录问题/文档正文的结构化事件
evals/                # 标注检索集、离线评测夹具与冻结质量门禁
scripts/              # 基准、校准和质量检查入口
tests/                # 单元、隔离、并发、故障与 API 契约测试
```

## 安全与边界

- 上传内容、网络摘要和检索片段全部视为不可信数据；系统提示明确禁止执行证据中的指令。
- API 不回显供应商响应体、内部路径、租户 ID、内部索引 ID、问题正文或文档正文到日志。
- 网络搜索会发送问题，云端生成会发送问题与选中证据；两条出站路径分别授权。
- 会话、可执行任务队列和限流状态仍为进程内状态；job 快照/ID 已有租户隔离的 SQLite 有界归档。重启不会继续执行旧 worker，而会把其中断快照安全终态化，先核验或回滚 `PREPARING`，再按 Catalog 重新提交 `PENDING`/`INDEXING`，并把耐久 `CANCELLING` 收敛为 `FAILED`/`index_cancelled`。这仍不是分布式任务系统。
- “删除”是应用层逻辑删除与文件删除，不等于 SSD、快照和离线备份的介质级不可恢复擦除。

威胁模型、残余风险和安全报告流程见[安全设计](docs/security.md)与[安全策略](SECURITY.md)。

> 依赖边界：当前 Chroma 上游尚无针对 `PYSEC-2026-311` 的修复版本。本项目仅支持嵌入式
> `PersistentClient`，不得运行或暴露 Chroma Server；精确、会到期的风险例外及补偿控制见
> [依赖与供应链](docs/security.md#依赖与供应链)。

## 当前非目标

- 多副本写入、跨可用区容灾和零停机滚动升级
- 任意代码执行、自动训练或无边界自主工具循环
- 未经人工评审的自动科研结论或自动投稿
- 用代码行数、测试数量或开发集得分代替真实负载与领域验收

下一阶段的规模化路径是把目录、向量服务、任务队列、会话和限流依次迁移到外置共享基础设施，并先补齐迁移工具、故障演练和多租户负载测试。

## 贡献与许可

版本变化见 [CHANGELOG.md](CHANGELOG.md)，贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。本仓库目前尚未选择开源许可证；在许可证明确前，请不要假设获得复制、修改或再分发授权。
