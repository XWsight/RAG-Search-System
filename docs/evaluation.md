# 评测与阈值校准

本项目把“评测代码能计算指标”“小型开发集 smoke test”和“真实混合检索表现”分开报告。任何结果都必须同时说明数据集、代码/配置、模型、top-k 和运行日期；不得把夹具输出包装成生产质量结论。

## 三类证据

| 层级 | 输入 | 实际测量对象 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- | --- |
| 指标夹具 | [`evals/sample_dataset.jsonl`](../evals/sample_dataset.jsonl) | 预先写入的 `retrieved_ids`、`predicted_route`、`answer` 和引用 | JSONL 严格校验、指标公式、引用 ID 审计和报告渲染工作正常 | 当前检索器、Embedding、路由或模型回答的质量 |
| BM25 smoke baseline | [`evals/retrieval_cases.jsonl`](../evals/retrieval_cases.jsonl) + 4 个 corpus 文档 | 真实 DocumentIngestor 切分、依赖无关的 BM25 检索和当前 RoutingPolicy | 小型开发语料上的确定性回归基线 | 混合检索、真实业务、生成质量、规模/并发表现 |
| 真实本地检索基准 | 同一 ground truth + 当前 Chroma/HuggingFace/BM25/RRF/可选 reranker | 实际本地检索结果和路由 | 指定模型与配置在该数据集上的检索/路由结果 | 云生成事实性、真实流量泛化或生产 SLA |

`retrieval_cases.jsonl` 是严格 ground truth，只允许问题、相关来源、期望路由和 `allow_web`；loader 会拒绝混入预测字段。这样可避免把手写预测误当作系统输出。

## 指标定义

- **Recall@k**：每个可回答样例在前 k 个唯一来源中找回的相关来源比例，再做宏平均。
- **MRR@k**：第一个相关来源排名倒数，在可回答样例上宏平均。
- **nDCG@k**：使用 1–3 级相关性计算折损累计增益，在可回答样例上宏平均。
- **路由准确率**：全部样例中 `local`、`web` 或 `refused` 与人工期望一致的比例。
- **引用有效率**：回答里出现的引用 ID 属于允许集合的比例。
- **引用覆盖率**：被引用标记覆盖的事实句比例；它是格式启发式，不验证引用是否蕴含句子。
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

## 2. 12-case BM25 smoke baseline

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

在 2026-08-11、数据集摘要 `2723586171c4445d`、仓库当前默认检索/路由配置下，本地实测为：

| 指标 | 结果 |
| --- | ---: |
| Recall@5 | 1.000000000000 |
| MRR@5 | 0.950000000000 |
| nDCG@5 | 0.963092975357 |
| 路由准确率 | 0.750000000000 |

该开发集只有 12 个样例，其中 10 个有相关来源，语料仅 4 篇、主题与代码高度贴近。结果适合发现明显回归，**不可外推**到真实业务、不同语言、长文档、同名来源、噪声语料或更大知识库。路由准确率 0.75 也表明当前默认阈值在该集合上仍有误路由，不应只展示检索指标而隐藏路由表现。

这个 benchmark 没有引用样例；Markdown 报告把引用有效率/覆盖率显示为 `N/A`。机器可读 JSON 为保持指标字段始终是数值，仍用 `1.0` 表示“没有待评引用时的约定值”，不能解释为生成引用达到 100%。

### 冻结质量门禁

[`evals/gates/bm25-smoke.json`](../evals/gates/bm25-smoke.json) 固定以下契约：

- 数据集摘要必须为 `2723586171c4445d`，防止数据悄悄变化后继续沿用旧基线；
- `top_k` 必须为 5，防止通过扩大候选数量伪造提升；
- Recall@5 不低于 `1.0`、MRR@5 不低于 `0.95`、nDCG@5 不低于 `0.963`、路由准确率不低于 `0.75`；
- 门禁 JSON 使用严格 schema：未知字段、重复键、NaN、无穷大、越界值和错误类型都会失败；
- 指标回归时脚本仍先写出逐题 JSON/Markdown 报告，然后以退出码 `3` 结束，便于 CI 保存诊断。

延迟门槛字段已经受 schema 支持，但当前仓库门禁保持为空。GitHub runner、开发机、模型冷启动和缓存会造成明显波动；在没有固定硬件与预热协议前，用跨机器毫秒阈值阻止提交会制造噪声。延迟数字目前必须和运行环境一起报告。

基准命令默认不读取项目 `.env`，避免 API Key 或本地运行参数无意间污染可复现结果。如确实要复现某个部署配置，显式添加 `--dotenv path/to/evaluation.env`，并在报告旁记录该配置的脱敏摘要。

## 3. 真实 hybrid benchmark

先安装运行依赖；首次运行可能下载 Embedding 模型。该命令只执行本地索引和检索，不调用智谱 Chat 或 Web Search：

```powershell
python scripts/benchmark_retrieval.py evals/retrieval_cases.jsonl `
  evals/corpus/rag.md `
  evals/corpus/retrieval.md `
  evals/corpus/safety.md `
  evals/corpus/storage.md `
  --top-k 5 `
  --json-output reports/hybrid-run.json `
  --markdown-output reports/hybrid-run.md
```

运行前记录以下配置，否则结果不可复现：

- Git commit、Python/操作系统、`requirements.txt` 和模型缓存版本；
- `EMBEDDING_MODEL`、`RAG_RERANKER_MODEL` 与 weight；
- chunk size/overlap、dense/sparse/fused candidates、final evidence count；
- `RAG_LOCAL_CONFIDENCE` 和 top-k；
- ground truth 文件摘要与 corpus 内容摘要。

仓库当前不附带 hybrid 质量门禁或已经执行的 hybrid 结果。冻结独立 validation 集后，可以按 BM25 gate 的严格格式创建独立门禁，并用 `--quality-gate path/to/hybrid-quality-gate.json` 启用。不得引用 BM25 数字作为 hybrid 成绩。建议把报告保存到忽略跟踪的 `reports/`，在变更说明中提供 before/after、运行环境和报告摘要。

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

校准器枚举候选阈值，先最小化 `2 × FP + 1 × FN` 的平均加权错误，再依次偏好更高 F1、更高 precision 和更保守的阈值。这里 FP 表示本地证据不足却选择本地回答，默认代价更高。

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

当前离线工具覆盖检索、路由、引用 ID 有效性和启发式引用覆盖；尚未实现以下可信结论所需的基准：

- 回答正确性、完整性、拒答质量和引用蕴含；
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
4. 报告每项指标、误差样例和配置差异，而不是只挑最好的数字；
5. 若修改阈值，在独立 validation 上重新校准；
6. 若触及生产路径，补充端到端、恢复或安全测试证据，并明确尚未运行的测试。

测试与提交要求见 [`CONTRIBUTING.md`](../CONTRIBUTING.md)。
