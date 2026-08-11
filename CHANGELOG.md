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
- 可重试的协作取消状态机：先耐久提交知识库 `CANCELLING` 意图，再向 worker 发信号；重启或幂等重放把残留意图收敛为 `FAILED`/`index_cancelled` 并提供可轮询 job，而已经提交的 `READY` 优先于迟到取消。
- SQLite 资源目录、租户文件存储、持久索引复用、崩溃恢复和完整删除流程。
- Prometheus 指标、隐私安全事件日志、健康检查、Docker Compose 和运维手册。
- 检索评测、路由阈值校准、故障注入测试、Python 3.11/3.12 CI 和依赖审计。
- 检索评测新增逐题失败诊断、首个相关排名、P50/P95/P99 延迟，以及绑定数据集摘要和 top-k 的冻结质量门禁；BM25 回归现在会直接阻止 CI。
- 将 `pypdf` 升级到 `6.15.0`；对唯一无修复的 Chroma 公告 `PYSEC-2026-311` 实行精确例外，并在 `2026-09-01` 关闭式失效。

### Changed

- 本地入口改为模块化的 Gradio 工作台；原始 V1.0 继续保留在 `main` 分支。
- 云端生成和网络搜索改为请求级显式授权，默认保持关闭。
- Catalog schema 升级为 v3；启动时事务化迁移受支持的 v2 数据，以容纳耐久 `CANCELLING` 状态。旧程序回滚必须配对恢复升级前的完整卷快照。

### Security

- 上传、文档证据和网络摘要均按不可信输入处理。
- API 错误与结构化日志不回显密钥、租户标识、问题正文、文档正文或供应商响应体。
- 生产存储根使用单实例系统锁，阻止多个本地写进程同时操作 SQLite 与 Chroma。

[Unreleased]: https://github.com/XWsight/RAG-Search-System/compare/7ea6c4249d80c86ded88dcae98e0d409d8ba35d1...rag-studio
