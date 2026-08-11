# Changelog

本项目采用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的组织方式，版本号遵循
[Semantic Versioning](https://semver.org/lang/zh-CN/)。尚未发布的行为变化应先记录在
`Unreleased`，再随经过验证的版本一起冻结。

## [Unreleased]

> 当前增强版仍处于开发阶段，尚未创建正式 Release 或版本 Tag。

### Added

- 同源产品 Web 工作台：知识库上传与管理、异步索引进度、多轮问答、引用展示，以及云端和联网的显式隐私开关。
- Web 工作台增强：知识库搜索和移动端切换、资料清单、可移除的上传文件、快捷提问、回答复制、处理状态和外部服务二次确认。
- Swagger 多文件上传字段的浏览器文件选择器兼容修复和回归测试。
- 安全的 TXT、Markdown、HTML、DOCX 和 PDF 摄取，以及确定性切分和资源边界。
- Chroma 向量检索、BM25、RRF 融合、来源多样化和可选 CrossEncoder 重排序。
- 本地、混合、联网与拒答路由，引用白名单和有预算上限的研究模式。
- 有界多轮会话记忆，以及租户、知识库和浏览器会话三重隔离。
- FastAPI 服务、角色授权、请求前认证、限流、持久幂等和后台索引任务。
- 租户隔离的 SQLite job 快照归档：任务状态、ID 和有界结果可跨重启查询；中断执行会安全终态化，Catalog 恢复仍以新 worker 重新提交，避免伪装成可恢复的分布式队列。
- 显式 `PREPARING → PENDING → INDEXING → READY` 建库状态机和一次性不可变清单：完整上传可在重启后继续，部分上传会精确回滚，清理失败会保留墓碑与幂等预留，防止重复资源和不可达文件。
- 可重试的协作取消状态机：先耐久提交知识库 `CANCELLING` 意图，再向 worker 发信号；重启或幂等重放把残留意图收敛为 `FAILED`/`index_cancelled` 并提供可轮询 job，而已经提交的 `READY` 优先于迟到取消。
- SQLite 资源目录、租户文件存储、持久索引复用、崩溃恢复和完整删除流程。
- Prometheus 指标、隐私安全事件日志、健康检查、Docker Compose 和运维手册。
- 检索评测、路由阈值校准、故障注入测试、Python 3.11/3.12 CI 和依赖审计。
- 检索评测新增逐题失败诊断、首个相关排名、P50/P95/P99 延迟，以及绑定数据集摘要和 top-k 的冻结质量门禁；BM25 回归现在会直接阻止 CI。
- 新增严格的 200-question/50-family 检索套件：10 篇来源按 development/validation/test 隔离，覆盖本地、拒答、联网、难度和 12 类主题；自动拒绝重复问题、来源泄漏、危险路径、错误路由标签与覆盖不足，并在 CI 冻结完整 BM25 下限。
- 新增严格的 50-case/70-fact 结构化回答套件：冻结证据、原子事实、引用、拒答、split、难度、类别与风险标签覆盖；默认 CI 校验完整契约，真实云端评测可按 development/validation/test 手动运行。
- 结构化回答基准新增 split、类别、难度和风险标签质量切片，逐片报告 case/fact/claim 数、严格通过数、五项指标和失败 case ID；无适用事实的拒答切片明确显示 `N/A`。
- 路由置信度改为用真实词法覆盖调节 dense/sparse 一致性奖励，收紧弱本地证据的混合区间；18 题对抗开发集上的 Hybrid 路由由 `0.9444` 提升为 `1.0000`。
- 云端回答由自由文本引用重构为严格的原子结论—证据 JSON 契约；未知/重复/缺失引用、含混拒答和截断输出整体失败关闭，API 暴露可审计的 claim 映射。结构化调用关闭 Thinking 与随机采样，并只为可修复协议错误提供一次不回显原输出的有界重试。
- 将 `pypdf` 升级到 `6.15.0`；对唯一无修复的 Chroma 公告 `PYSEC-2026-311` 实行精确例外，并在 `2026-09-01` 关闭式失效。

### Changed

- 检索与回答评测套件统一使用共享的严格 JSON、规范化去重、内容摘要和冻结契约基础，避免两套治理逻辑继续分叉。

- Refactored platform workflows onto explicit repository, document-store, job-executor, index-lifecycle, and knowledge-service ports. Production readiness now composes fail-closed probes for catalog, document, job, and vector storage; the 28-module architecture spine is protected by strict mypy and dependency-direction gates.

- 本地入口改为模块化的 Gradio 工作台；原始 V1.0 继续保留在 `main` 分支。
- 云端生成和网络搜索改为请求级显式授权，默认保持关闭。
- 将供应商无关的回答协议、严格 JSON 解码和稳定错误契约从智谱 HTTP 适配器中分离；应用层不再反向依赖具体 Provider，并用架构测试锁定依赖方向。
- 将 HTTP 边界改为依赖框架无关的应用端口，并分离版本化 API schema 与集中错误分类；将上传策略、文档资产一致性、索引状态机和并发协调从平台门面中抽出。新增导入环、HTTP 依赖方向、双向作业登记和有界上传专项门禁。
- Catalog schema 升级为 v4；启动时事务化迁移受支持的 v2/v3 数据，以容纳显式 `PREPARING` 状态。旧程序回滚必须配对恢复升级前的完整卷快照。

### Security

- 上传、文档证据和网络摘要均按不可信输入处理。
- API 错误与结构化日志不回显密钥、租户标识、问题正文、文档正文或供应商响应体。
- 生产存储根使用单实例系统锁，阻止多个本地写进程同时操作 SQLite 与 Chroma。

[Unreleased]: https://github.com/XWsight/RAG-Search-System/compare/7ea6c4249d80c86ded88dcae98e0d409d8ba35d1...rag-studio
