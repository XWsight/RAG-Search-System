# 评测与阈值校准

本项目把“评测代码能计算指标”“小型开发集 smoke test”和“真实混合检索表现”分开报告。任何结果都必须同时说明数据集、代码/配置、模型、top-k 和运行日期；不得把夹具输出包装成生产质量结论。

数据维护、双人标注、family 语义和 split 使用规则见 [`evals/README.md`](../evals/README.md)。

## 三类证据

| 层级 | 输入 | 实际测量对象 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- | --- |
| 指标夹具 | [`evals/sample_dataset.jsonl`](../evals/sample_dataset.jsonl) | 预先写入的 `retrieved_ids`、`predicted_route`、`answer` 和引用 | JSONL 严格校验、指标公式、引用 ID 审计和报告渲染工作正常 | 当前检索器、Embedding、路由或模型回答的质量 |
| BM25 smoke baseline | [`evals/retrieval_cases.jsonl`](../evals/retrieval_cases.jsonl) + 4 个 corpus 文档 | 真实 DocumentIngestor 切分、依赖无关的 BM25 检索和当前 RoutingPolicy | 小型开发语料上的确定性回归基线 | 混合检索、真实业务、生成质量、规模/并发表现 |
| 216-case retrieval suite | [`evals/retrieval_suite.json`](../evals/retrieval_suite.json) + 10 篇来源 | 54 个语义家族的改写鲁棒性、通用负例、来源隔离 split、困难度、路由与完整 BM25 回归 | 更广覆盖下的确定性检索下限和逐题失败资产 | 216 个独立事实、真实客户分布、混合检索或生产 SLA |
| 真实本地检索基准 | 同一 ground truth + 当前 Chroma/HuggingFace/BM25/RRF/可选 reranker | 实际本地检索结果和路由 | 指定模型与配置在该数据集上的检索/路由结果 | 云生成事实性、真实流量泛化或生产 SLA |

`retrieval_cases.jsonl` 是严格 ground truth，只允许问题、相关来源、期望路由和 `allow_web`；loader 会拒绝混入预测字段。这样可避免把手写预测误当作系统输出。

## 指标定义

- **Recall@k**：每个可回答样例在前 k 个唯一来源中找回的相关来源比例，再做宏平均。
- **MRR@k**：第一个相关来源排名倒数，在可回答样例上宏平均。
- **nDCG@k**：使用 1–3 级相关性计算折损累计增益，在可回答样例上宏平均。
- **路由准确率**：全部样例中 `local`、`web` 或 `refused` 与人工期望一致的比例。
- **引用有效率**：旧格式评测夹具中，回答里出现的引用 ID 属于允许集合的比例。
- **引用覆盖率**：旧格式评测夹具中，被引用标记覆盖的事实句比例；它是格式启发式，不验证引用是否蕴含句子。生产生成路径已经改用结构化 claim 契约，不再依赖该正则作为信任边界。
- **逐题诊断**：记录相关来源缺失、期望/实际路由、首个相关来源排名、置信度和耗时，避免只看宏平均值掩盖失败样例。
- **延迟分位数**：同一进程顺序执行每题检索与路由，并报告平均值、P50、P95、P99 和最大值。它适合同机 before/after，不代表并发吞吐或生产 SLA。

真实检索基准按 `source_name` 去重后评分，而不是按 chunk ID。多个相关 chunk 命中同一文档只算一个来源。

## 1. 指标夹具

运行：

```powershell
python scripts/evaluate.py evals/sample_dataset.jsonl --top-k 5 `
  --json-output reports/sample-fixture.json `
  --markdown-output reports/sample-fixture.md
```

该文件只有 6 个手工样例，且已经包含“预测”和回答，因此输出只能用于验证评测器。它不会加载文档、创建 Chroma、调用 Embedding、搜索网页或调用模型。报告它时必须称为 **sample fixture**，不能称为项目 Recall、路由准确率或回答质量。

## 2. 18-case BM25 smoke baseline

无需 Chroma、模型下载或云端调用：

```powershell
python scripts/benchmark_sparse.py evals/retrieval_cases.jsonl `
  evals/corpus/rag.md `
  evals/corpus/retrieval.md `
  evals/corpus/safety.md `
  evals/corpus/storage.md `
  --top-k 5 `
  --quality-gate evals/gates/bm25-smoke.json `
  --json-output reports/bm25-smoke.json `
  --markdown-output reports/bm25-smoke.md
```

在 2026-08-11、ground truth 摘要 `74fe19194ca06876`、仓库当前默认检索/路由配置下，本地实测为：

| 指标 | 结果 |
| --- | ---: |
| Recall@5 | 1.000000000000 |
| MRR@5 | 0.958333333333 |
| nDCG@5 | 0.969244146131 |
| 路由准确率 | 0.833333333333 |

该开发集只有 18 个样例，其中 12 个有相关来源，语料仅 4 篇、主题与代码高度贴近。新增样例覆盖语义改写、本地越界和必须联网的问题。结果适合发现明显回归，**不可外推**到真实业务、不同语言、长文档、同名来源、噪声语料或更大知识库。BM25 缺少语义模型，仍会误拒部分可回答改写题，因此不应把它的路由结果冒充 Hybrid 表现。

这个 benchmark 没有引用样例；Markdown 报告把引用有效率/覆盖率显示为 `N/A`。机器可读 JSON 为保持指标字段始终是数值，仍用 `1.0` 表示“没有待评引用时的约定值”，不能解释为生成引用达到 100%。

### 冻结质量门禁

[`evals/gates/bm25-smoke.json`](../evals/gates/bm25-smoke.json) 固定以下契约：

- ground truth 摘要必须为 `74fe19194ca06876`，防止题目或人工标签悄悄变化后继续沿用旧基线；摘要只依赖问题、相关来源、期望路由和联网许可，不包含系统预测，因此不同检索器可以公平比较；
- `top_k` 必须为 5，防止通过扩大候选数量伪造提升；
- Recall@5 不低于 `1.0`、MRR@5 不低于 `0.958`、nDCG@5 不低于 `0.969`、路由准确率不低于 `0.833`；
- 门禁 JSON 使用严格 schema：未知字段、重复键、NaN、无穷大、越界值和错误类型都会失败；
- 指标回归时脚本仍先写出逐题 JSON/Markdown 报告，然后以退出码 `3` 结束，便于 CI 保存诊断。

延迟门槛字段已经受 schema 支持，但当前仓库门禁保持为空。GitHub runner、开发机、模型冷启动和缓存会造成明显波动；在没有固定硬件与预热协议前，用跨机器毫秒阈值阻止提交会制造噪声。延迟数字目前必须和运行环境一起报告。

基准命令默认不读取项目 `.env`，避免 API Key 或本地运行参数无意间污染可复现结果。如确实要复现某个部署配置，显式添加 `--dotenv path/to/evaluation.env`，并在报告旁记录该配置的脱敏摘要。

## 3. 216-case retrieval foundation suite

[`evals/retrieval_suite.json`](../evals/retrieval_suite.json) 不是把 18 条题目机械复制。它包含 54 个语义家族，每个家族有 4 种人工编写问法，共 216 个 case；10 篇来源只属于 development、validation 或 test 中的一个分段，loader 会拒绝同一来源跨分段出现。新增 16 题覆盖普通世界知识、无关理工/生活问题和不支持的外部动作，只进入 development/validation，没有改写首次冻结的 test。覆盖矩阵为：

| 维度 | 分布 |
| --- | --- |
| split | development 88 / validation 68 / test 60 |
| route | local 160 / refused 36 / web 20 |
| difficulty | easy 64 / medium 92 / hard 60 |
| semantics | 54 families / 12 categories / 10 source documents |

先验证数据契约，再运行全语料基线：

```powershell
python scripts/validate_retrieval_suite.py evals/retrieval_suite.json `
  --contract evals/gates/retrieval-suite.json `
  --json-output reports/retrieval-suite.json `
  --markdown-output reports/retrieval-suite.md

python scripts/benchmark_sparse.py evals/retrieval_suite.json `
  --top-k 5 `
  --quality-gate evals/gates/bm25-foundation.json `
  --json-output reports/bm25-foundation.json `
  --markdown-output reports/bm25-foundation.md
```

manifest 使用严格 JSON、精确字段和显式最低覆盖要求。校验会拒绝：重复 family/case、忽略空白和标点后相同的问题、错误的 `allow_web`/route/relevance 组合、不存在或逃逸 corpus 根的路径、符号链接来源、来源跨 split 泄漏，以及低于声明的题量、家族、类别、route、difficulty 或 split 覆盖。独立冻结契约还绑定规范化 manifest、全部 corpus 正文 SHA-256 和精确覆盖矩阵；当前 bundle 摘要为 `e75fb276b6a2a227`，因此只改语料正文也会让 CI 失败。

2026-08-11、摘要 `2f40b11e574096d0` 的 10 文档全语料 BM25 运行结果为：

| 指标 | 结果 |
| --- | ---: |
| Recall@5 | 0.984375000000 |
| MRR@5 | 0.956770833333 |
| nDCG@5 | 0.959200740890 |
| 路由准确率 | 0.944444444444 |

160 个本地问题中有 4 个未完整召回，主要来自跨来源问题和更大语料中的干扰项；查询能力意图层把实时、外部动作和受限请求从证据阈值中分离后，BM25 的路由达到 `0.9444`，仍有 12 个普通无关问题被稀疏分数误判为本地可回答。冻结门禁锁住这个下限是为了防止继续退化，不能把它描述为 Hybrid 或生产质量。

可用 `--split development|validation|test` 单独运行来源隔离分段。分段只索引该 split 的文档，因此适合开发和误差分析；全套运行同时加入其余文档作为干扰项，结果不能与分段数字直接混为一谈。216 个 case 中包含同一家族的语义改写，汇报时必须同时写明“216 questions / 54 semantic families”，不得宣称 216 个独立事实。

当输入是 governed suite 时，BM25 与 Hybrid 两个 runner 都会生成同一份诊断契约：总体结果、路由混淆矩阵，以及 split、category、difficulty、expected route 四类切片。逐题预测同时记录查询意图/规则 ID、首名/次名分数、分差、dense/sparse 一致性、词法原始分数/饱和支撑度和最终置信度；这些字段不包含问题或文档正文，可用于聚合和失败定位。报告 schema `4` 是增加字段后的版本，读取方不得假设未知字段不存在。

## 4. 真实 hybrid benchmark

先安装运行依赖；首次运行可能下载 Embedding 模型。该命令只执行本地索引和检索，不调用智谱 Chat 或 Web Search：

```powershell
python scripts/benchmark_retrieval.py evals/retrieval_cases.jsonl `
  evals/corpus/rag.md `
  evals/corpus/retrieval.md `
  evals/corpus/safety.md `
  evals/corpus/storage.md `
  --top-k 5 `
  --quality-gate evals/gates/hybrid-development.json `
  --json-output reports/hybrid-run.json `
  --markdown-output reports/hybrid-run.md
```

运行前记录以下配置，否则结果不可复现：

- Git commit、Python/操作系统、`requirements.txt` 和模型缓存版本；
- `EMBEDDING_MODEL`、`RAG_RERANKER_MODEL` 与 weight；
- chunk size/overlap、dense/sparse/fused candidates、final evidence count；
- `RAG_LOCAL_CONFIDENCE`、`RAG_HYBRID_CONFIDENCE_RATIO`、`RAG_ROUTING_LEXICAL_SATURATION` 和 top-k；
- ground truth 文件摘要与 corpus 内容摘要。

当前 18 题 Hybrid 开发基线的 Recall@5、MRR@5、nDCG@5 和路由准确率均为 `1.0`，并由 [`hybrid-development.json`](../evals/gates/hybrid-development.json) 冻结。它需要实际加载 Embedding 模型，是发布前手动门禁；默认 CI 只运行不依赖模型下载的 BM25 门禁。该结果来自本地 `BAAI/bge-small-zh-v1.5`、当前依赖与默认配置，不代表独立 blind test、生成质量或生产 SLA。不得引用 BM25 数字作为 Hybrid 成绩，也不得把这 18 题的满分外推为真实业务准确率。

同一环境运行 216 题全语料 Hybrid 得到 Recall@5 `0.984375`、MRR@5 `0.953646`、nDCG@5 `0.953695`、路由准确率 `0.990741`，由 [`hybrid-foundation.json`](../evals/gates/hybrid-foundation.json) 冻结为手动下限。查询能力意图先处理实时、未授权动作和受限请求，只有普通知识问题进入 `0.59` 证据阈值；排序指标仍说明稠密候选和 RRF 并非全面增益。

同一默认模型和配置的来源隔离运行中，development 88 题和 validation 68 题的 Recall@5、路由准确率均为 `1.0`。配置冻结后的首次 test 得到 Recall@5 `1.0`、MRR `0.989583`、nDCG `0.992311`、路由 `0.916667`，其中医疗诊断和密钥提取请求暴露能力边界缺失。修复后该公开 test 已被消费，只能作为回归集；后续可信泛化结论需要新的、未参与开发且最好由独立标注者维护的外部盲测集。

## 路由阈值校准

用真实 hybrid run JSON 校准，而不是 sample fixture 或手工预测：

```powershell
python scripts/calibrate_threshold.py `
  evals/retrieval_cases.jsonl `
  reports/hybrid-run.json `
  --false-positive-cost 2 `
  --false-negative-cost 1 `
  --output reports/threshold.md
```

校准器枚举相邻置信度的中点，先最小化 `2 × FP + 1 × FN` 的平均加权错误，再依次偏好更高 F1、更高 precision 和离最近样例更远的稳定间隔。这里 FP 表示本地证据不足却选择本地回答，默认代价更高。查询能力意图分层并加入普通 hard negatives 后，本地证据阈值冻结为 `0.59`；它只适用于当前 Embedding、融合公式和语料分布，不是跨项目常数。

不要在同一小集合上选阈值后又把该集合的最优结果当作无偏测试成绩。正确流程是：

1. 用训练集开发检索、重排和提示；
2. 用独立 validation 集只选择超参数与路由阈值；
3. 冻结代码、依赖、模型和阈值；
4. 在一次性 blind test 上生成最终报告；
5. 后续改动创建新版本，不能反复窥视 test 后调参。

## 防止数据泄漏

- 按**来源文档、客户/租户、时间窗口或主题簇**切分，而不是把同一文档的相邻 chunk 随机分到 train/test。
- 去重原文和近重复文档；模板文本、答案提示、文件名及明显实体也可能泄漏标签。
- ground truth 不包含系统预测。生成 run JSON 后保持只读，并记录数据集摘要。
- 标注者先写相关性和期望路由，再查看待评系统输出；有争议样例保留双人标注和裁决记录。
- 不把供应商返回、私有文档或真实 API Key 提交到仓库。若使用生产抽样，先脱敏并获得授权。
- 选择 Embedding/reranker、切分、RRF 权重、top-k 或 prompt 都算调参，必须只看开发/验证集。
- 做时间敏感 Web 评测时固定查询时间、地区、供应商与原始响应快照；否则结果无法复现。

## 生成与端到端质量

正式回答套件 [`answer_suite.json`](../evals/answer_suite.json) 在原有回答 ground truth 上增加 category、split、difficulty 和 risk tags 治理。当前包含 50 个独立问题、70 个原子事实、35 个可回答样例和 15 个拒答样例；覆盖冲突证据、错误领域近似证据、间接提示注入、隐私边界、生命周期、可靠性和资源耗尽等 13 个类别。冻结契约摘要为 `89e99234c8b10102`：

```powershell
python scripts/validate_answer_suite.py evals/answer_suite.json `
  --contract evals/gates/answer-suite.json
```

该命令只验证数据，不调用模型，所以进入默认 CI。真实生成运行必须显式提供 `--dotenv`；可用 `--split development|validation|test` 保持开发、选型与最终复核分离。完整 50 题云端运行有费用且受供应商版本和随机性影响，不能作为每次提交的确定性门禁。

使用 governed suite 运行时，JSON/Markdown 报告会同时给出总体指标和四类切片：split、category、difficulty、risk tag。每个切片包含 case/fact/claim 数、严格通过数、五项指标和失败 case ID；没有标注事实的纯拒答切片把事实召回、原子性和归因显示为 `N/A`，不能把“无适用分母”包装成满分。切片用于定位退化，不得只挑表现最好的切片对外报告。

生产生成路径使用结构化 `claims + citation_ids + insufficient` 契约。独立的 [`answer_cases.jsonl`](../evals/answer_cases.jsonl) 只包含问题、证据和人工标注的原子事实，不包含模型预测。运行时直接把这些证据交给当前 Chat provider，再分别计算：

- **结构契约成功率**：供应商输出通过严格 JSON、状态和当前证据注册表校验的样例比例；
- **拒答准确率**：有事实时回答、无事实时明确 `insufficient` 的样例比例；
- **事实召回率**：被一个正确归因的原子 claim 覆盖的必需事实比例；
- **原子结论率**：只匹配一个人工事实、没有把多个事实塞进同一 claim 的比例；
- **归因精确率**：原子 claim 的所有引用都属于该事实人工支持来源的比例。

```powershell
python scripts/benchmark_answers.py evals/answer_cases.jsonl `
  --dotenv .env `
  --quality-gate evals/gates/answer-live.json `
  --json-output reports/answers-live.json `
  --markdown-output reports/answers-live.md
```

2026-08-11、数据集摘要 `cb234975cbaf3a67` 的一次真实智谱运行中，4 个样例、8 个原子事实的五项指标均为 `1.0`。此前协议/标注迭代曾测得事实召回 `0.375` 和一次 `ProviderProtocolError`，因此当前满分不能解释为任务简单或模型天然可靠。手动门禁将五项最低值设为 `0.75`，允许一个小样例的非确定性波动；它不在默认 CI 中运行，门禁通过也不代表开放域事实正确。

该方法采用人工关键词组识别仓库内的有限原子事实，优点是可解释、无 judge 漂移，缺点是不能可靠识别任意同义改写。它是开发冒烟而不是语义蕴含证明。仍未实现以下可信结论所需的基准：

- 开放域语义等价、跨语言事实正确性和由独立 NLI/人工标注验证的引用蕴含；
- 间接提示注入成功率、隐私外发和有害输出；
- 多轮记忆、研究模式、多查询规划和 Web 来源质量；
- 并发吞吐、端到端生成延迟、成本、索引时间和峰值内存；当前仅实现顺序检索/路由的 P50/P95/P99；
- 供应商故障、磁盘满、进程终止和恢复一致性。

若引入 LLM-as-judge，必须固定 judge 模型/版本/prompt，加入人工校准和顺序盲化，并把 judge 结果与硬指标分开。高风险结论应抽样人工核验，不能把一个模型评价另一个模型当作客观真值。

## 变更门槛

涉及 loader、splitter、Embedding、BM25、RRF、reranker、路由或 prompt 的变更至少应：

1. 通过全部单元测试与静态检查；
2. 运行 BM25 smoke，确认非目标模块无意外回归；
3. 在冻结的 validation 集运行真实 hybrid before/after；
4. 修改生成协议、prompt 或 provider 时运行真实结构化回答门禁；
5. 报告每项指标、误差样例和配置差异，而不是只挑最好的数字；
6. 若修改阈值，在独立 validation 上重新校准；
7. 若触及生产路径，补充端到端、恢复或安全测试证据，并明确尚未运行的测试。

测试与提交要求见 [`CONTRIBUTING.md`](../CONTRIBUTING.md)。
