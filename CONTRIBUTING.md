# 贡献指南

感谢改进 `rag-studio`。贡献应保持租户隔离、隐私默认关闭、资源有界和可验证结果，不以增加框架或代码量作为目标。

仓库当前未包含开源许可证。公开可见不等于授予额外使用、复制或分发权；许可证选择由维护者另行决定，本文件不会替代许可证。

## 开发环境

支持 Python 3.11 和 3.12。建议使用独立虚拟环境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

不要提交 `.env`、真实 API Key、模型缓存、`.rag_data`、评测报告或客户数据。单元测试不应依赖真实云凭据；通过协议、fake provider、临时目录和可控时钟隔离外部状态。

## 开始修改

1. 从最新目标分支创建短生命周期分支，例如 `fix/tenant-delete`、`feat/pdf-boundary` 或 `test/router-calibration`。
2. 先阅读[系统架构](docs/architecture.md)、[威胁模型](docs/security.md)和[评测指南](docs/evaluation.md)。涉及运行方式时同时阅读[部署指南](docs/deployment.md)与[运维手册](docs/operations.md)。
3. 保持改动单一目的；不要在功能提交中混入无关格式化、依赖升级或大规模重命名。
4. 对用户可见行为、持久格式、环境变量和安全边界同步更新测试与文档。

## 代码约定

- 目标版本为 Python 3.11；行长 100，Ruff 规则以 [`pyproject.toml`](pyproject.toml) 为准。
- 业务依赖通过 [`ports.py`](rag_system/ports.py) 或构造参数注入，避免在领域层绑定 HTTP、FastAPI 或具体供应商。
- 所有外部输入必须有类型、长度、数量和字符集边界；异常跨 API 边界前转换成稳定 code 与安全消息。
- 禁止把 question、document text、evidence、HTTP headers、API Key 或任意上游响应加入日志/指标。
- 租户资源查询必须在存储层携带 tenant 条件；不能先按全局 ID 读取再在内存中判断所有权。
- 新的内存缓存、队列、metric label 或历史记录必须有 TTL/容量/基数上限。
- 文件写入与删除必须使用验证后的精确路径，拒绝符号链接、重解析点和路径穿越。
- 外部调用必须有超时、有限重试、严格响应解析和隐私开关。不要根据模型/文档内容执行代码或任意工具。
- `domain`、`ports`、`grounding` 与回答协议必须保持框架无关；应用和 HTTP 层不得导入具体 Provider。扩展新模型时复用 `AnswerProtocol` 与 `provider_errors`，不要复制 prompt、引用校验或重试状态机。
- 不捕获宽泛异常后静默成功；如需降级，应保留非敏感诊断 code 并测试该路径。

## 测试

Windows 上运行完整本地检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check.ps1
```

它依次执行 compileall、Ruff、带分支覆盖率的 unittest、覆盖率门槛和 `git diff --check`。也可以单独运行：

```powershell
python -m unittest discover -s tests -v
python -m ruff check .
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
git diff --check
```

最低测试要求：

- 修复缺陷时先加入能重现问题的回归测试。
- 新模块包含正常路径、边界值、无效类型/大小、并发或幂等路径，以及安全失败行为。
- 文件系统测试使用临时目录；不得删除开发者仓库、home 或广泛匹配路径。
- API 测试使用注入的应用端口、authenticator 和 provider，不依赖具体 `RagPlatform`，也不调用真实模型或搜索服务。
- `api.py` 只能依赖 `application.py` 暴露的用例端口，不得直接导入 platform、catalog、file store、job manager 或模型供应商；`test_architecture.py` 会检查该方向以及生产模块导入环。
- 平台工作流必须依赖 `application_ports.py` 的稳定端口，具体 SQLite、文件系统、任务执行器和向量实现只能在 `bootstrap.py` 组合根装配。CI 会对这条 26 模块的架构主干执行严格 mypy 检查；新增地基模块必须加入 `[tool.mypy].files`，不得用宽泛 ignore 绕过。
- 时间相关组件注入 fake clock，任务并发测试必须有有界等待，避免 sleep 驱动的脆弱断言。
- SQLite/持久索引变更覆盖重启、部分写入、重复请求、外租户访问和删除恢复。

CI 在 Python 3.11/3.12 上执行单元测试、Ruff、分支覆盖率门槛和冻结的 BM25 检索质量门禁，并对固定版本的直接运行依赖执行 `pip-audit`。本地通过不是合并保证，CI 通过也不等于完成性能、渗透或恢复测试。

## 检索和模型变更

修改 loader、splitter、Embedding、BM25、RRF、reranker、路由阈值、研究规划或 prompt 时，按[评测指南](docs/evaluation.md)运行相应基准。

提交说明至少包含：

- before/after 的数据集摘要、配置与所有指标；
- 逐题退化和改善，而不只是平均值；
- 是否运行真实 hybrid、云端、人工、延迟或安全测试；
- 未运行项目和原因。

`evals/sample_dataset.jsonl` 只是指标夹具。不得用它声明实际系统性能。18-case BM25/Hybrid、200-question/50-family 检索套件、50-case/70-fact 回答套件和 4-case/8-fact 云端回答结果都只是仓库内回归基线，也不得外推为真实客户分布或生产质量。

若有意更新 [`evals/gates/bm25-smoke.json`](evals/gates/bm25-smoke.json) 或 [`evals/gates/bm25-foundation.json`](evals/gates/bm25-foundation.json)，提交必须解释数据集摘要变化、语义家族与 split 覆盖、逐题差异和门槛调整理由。不得为了让退化代码通过而单独降低门槛；同一来源不得跨 development/validation/test 泄漏。

真实 Hybrid 对完整套件的手动下限由 [`evals/gates/hybrid-foundation.json`](evals/gates/hybrid-foundation.json) 约束。它不进入默认 CI，修改 Embedding、融合、重排或路由时仍必须报告 development/validation 的 before/after，并说明 test 是否保持冻结。

修改回答套件时必须同时更新 `evals/gates/answer-suite.json`，解释 case、fact、split、category、risk-tag 覆盖和摘要变化；禁止只为提高当前模型得分而删除失败样例。修改 Chat provider、生成 prompt、claim schema 或证据渲染时，还必须先在 governed suite 的 development/validation 分段报告 before/after，并运行 [`answer-live.json`](evals/gates/answer-live.json) 手动冒烟门禁、保存脱敏报告。该门禁调用外部模型、有成本且非确定，因此不进入默认 CI；降低门槛必须附失败样例、人工复核和明确理由。

## 数据库、API 与配置兼容性

- Catalog/Idempotency schema 改动必须有明确版本检查、迁移与回滚方案；不能假设删除数据库即可升级。
- API 响应字段、状态 code、幂等语义和角色要求属于兼容性边界。破坏性变更必须版本化并更新调用示例。
- 新环境变量必须加入 `.env.example`、验证逻辑和部署文档，给出安全默认值。
- 持久布局变更必须同步备份、恢复和删除流程，并说明旧数据如何迁移。

## 提交与变更说明

推荐使用清晰的命令式提交前缀：

```text
feat: add bounded parser worker
fix: preserve tenant scope during recovery
test: cover partial Chroma rebuild
docs: document routing calibration
chore: update audited dependency pins
```

每个提交应可独立审查。Pull request 描述应包含：问题、设计选择、风险、验证命令与结果、数据/兼容性影响、回滚方法和后续工作。截图只能补充 UI 变更，不能替代自动化测试。

## Pull request 检查表

- [ ] 改动范围单一，未包含密钥、私有数据、缓存或生成报告。
- [ ] 新行为有正常、失败和边界测试。
- [ ] `scripts/check.ps1` 或等价命令通过，并粘贴准确结果。
- [ ] 租户隔离、隐私外发、日志、删除和资源上限已复核。
- [ ] 检索变更附冻结数据集上的 before/after，未把 fixture 当实测。
- [ ] API、schema、配置或持久布局变化已说明兼容/迁移/回滚。
- [ ] 文档描述的是已实现行为，未声称未执行的负载、恢复或安全测试。

## 安全问题

不要在公开 issue 或 pull request 中披露可利用细节、真实凭据或敏感样本。请按 [`SECURITY.md`](SECURITY.md) 的私密渠道报告；普通缺陷和功能建议可使用公开 issue，并提供最小、脱敏的复现。
