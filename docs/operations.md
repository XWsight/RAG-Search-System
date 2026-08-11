# 单节点运维手册

本手册针对 `compose.yaml` 的单 API 容器与 `rag-studio-data` 持久卷。所有变更先在非生产环境演练。涉及删除或覆盖数据的命令必须先核对环境、Compose 项目、卷名和备份校验值。

## 日常检查

```bash
docker compose ps
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
docker compose logs --tail=200 api
docker compose exec api sh -c 'du -sh /data/* 2>/dev/null || true'
```

- `live` 只说明进程能够响应。
- `ready` 表示本地文档存储根、Catalog 与进程内 JobManager 可用，可用于基础流量门控；它不验证 Embedding/Chroma 查询或外部供应商，仍需独立的代表性业务探针。
- `/metrics` 需要 `operator` 凭据；监控采集器只能获得该最小权限密钥，且不得把响应或请求头写入日志。
- 应告警的最低集合包括：持续不就绪、5xx 比例、请求延迟、任务失败/积压、长时间停留的 `CANCELLING`、限流次数、磁盘剩余量、容器重启和备份陈旧时间。

应用事件日志使用结构化字段并主动避开问题正文、文档正文和密钥。容器日志采用本地轮转驱动，但日志仍应按组织策略转存与控制访问。

## 一致性备份

SQLite 数据库、Chroma 索引和原始文档共同构成一个恢复单元，不能分别在持续写入时复制。最稳妥的单节点流程是停止 API，让 SIGTERM 完成优雅关闭，再对整个 `/data` 做一致性归档。

1. 建立主机上的备份目录，确认它位于持久卷之外并有足够空间。
2. 停止写入并等待容器完全停止：

   ```bash
   docker compose stop api
   docker compose ps
   ```

3. 从只读卷创建归档。下面的卷名与 Compose 默认值一致；若设置过 `RAG_DATA_VOLUME`，必须替换为实际且已核对的精确名称。

   ```bash
   mkdir -p backups
   stamp=$(date -u +%Y%m%dT%H%M%SZ)
   docker run --rm \
     -v rag-studio-data:/data:ro \
     -v "$PWD/backups:/backup" \
     busybox:1.37.0 \
     tar -C /data -czf "/backup/rag-studio-${stamp}.tar.gz" .
   sha256sum "backups/rag-studio-${stamp}.tar.gz" \
     > "backups/rag-studio-${stamp}.tar.gz.sha256"
   ```

4. 立即恢复服务并验证 readiness：

   ```bash
   docker compose start api
   curl --fail http://127.0.0.1:8000/health/ready
   ```

5. 将归档、校验文件、镜像 digest、代码提交和配置版本复制到离机且加密的备份位置。不要把 `.env` 与数据备份放在相同的访问域；密钥应由专门的密钥系统备份。

备份成功的定义不是“tar 命令返回 0”，而是在隔离环境完成过定期恢复演练，并验证知识库清单、抽样文档、索引查询和回答链路。根据 RPO 安排频率，根据恢复演练测得的时间确认 RTO。

## 恢复演练与正式恢复

始终恢复到一个**新卷**，验证后再切换；不要先清空当前卷。

```bash
sha256sum --check backups/rag-studio-<时间>.tar.gz.sha256
docker compose down
docker volume create rag-studio-data-restore
docker run --rm \
  -v rag-studio-data-restore:/restore \
  -v "$PWD/backups:/backup:ro" \
  busybox:1.37.0 \
  tar -C /restore -xzf /backup/rag-studio-<时间>.tar.gz
RAG_DATA_VOLUME=rag-studio-data-restore docker compose up -d
curl --fail http://127.0.0.1:8000/health/ready
```

恢复后用只读请求核对租户边界、知识库数量和抽样检索，再允许写入。对于启动配置中已知的租户，恢复会重新提交 `PENDING`/`INDEXING`，并把快照中的 `CANCELLING` 收敛为 `FAILED`/`index_cancelled`；原 job ID 不会恢复。确认恢复有效之前，保留原卷不动。归档只能来自可信来源；不要解压未经验证的外部归档。

PowerShell 中可用 `Get-FileHash -Algorithm SHA256 <文件>` 校验，并用 `$env:RAG_DATA_VOLUME='rag-studio-data-restore'` 设置本次 Compose 进程的卷名。

## 升级与回滚

单节点不适用滚动升级，因为没有第二个可接管流量的安全写入副本。标准流程会产生短暂停机：

1. 完成预发布测试、依赖审计和数据迁移演练。
2. 记录当前镜像 digest 与配置版本。
3. 按上一节创建并校验停写快照。
4. 使用已在 CI 构建、签名并按 digest 固定的新镜像。
5. 启动后检查 `live`、`ready`、指标和代表性请求，再恢复上游流量。

如果新版本仅有代码故障且没有改变持久化格式，可以停止服务并切回旧镜像。如果已经执行不可逆的数据迁移，旧程序不能直接读取新数据：必须停止服务，将旧镜像与升级前的新卷快照配对恢复。回滚决策和兼容矩阵应随每个发行版本记录。

当前版本把 Catalog 升级到 schema v3。全新空存储会直接初始化为 v3；首次打开已有 schema v2 时，应用会在单个 SQLite 事务中重建目录表并写入 `user_version=3`，以支持耐久 `CANCELLING`。未知版本以及带旧表的未版本化数据库会被拒绝。该迁移对旧程序不向后兼容：升级前必须完成停写全卷快照，回滚时必须同时恢复旧镜像和升级前快照，不能只切回旧镜像，也不要手工修改 `PRAGMA user_version`。

不要用两个版本同时连接同一个卷进行“蓝绿”测试；这仍是并发写入同一 SQLite/Chroma 数据集。

## 密钥轮换

`RAG_API_KEYS_JSON` 和 `ZHIPU_API_KEY` 在进程启动时读取，修改后需要重启单节点服务。

API 调用方密钥采用重叠轮换：

1. 在密钥系统和 `.env` 中加入新高熵密钥，保留旧密钥，重启并验证新密钥。
2. 让调用方迁移，观察旧密钥不再使用。
3. 从配置删除旧密钥，再次重启并确认旧密钥被拒绝。

每次请求只能发送 `Authorization: Bearer <key>` 或 `X-API-Key: <key>` 之一；同时发送会被拒绝。轮换期间不要在命令历史、工单、日志或截图中粘贴真实密钥。上游供应商密钥应按供应商支持的先建后撤流程轮换；若供应商只能立即替换，应安排维护窗口。

## 索引任务取消

`DELETE /v1/jobs/{job_id}` 是协作取消，不是强制终止线程。对仍处于 `PENDING`/`INDEXING` 的知识库，平台先把 Catalog 状态耐久写成 `CANCELLING`，提交成功后才向 JobManager/worker 发信号。若该写入失败，API 返回 `503`，job 保持原执行状态且未收到取消信号；调用方应使用同一 job ID 重试。

发出信号后，尚可从执行器撤销的排队 job 会直接变为 `cancelled`；运行中 job 先变为 `cancelling`，直到 worker 在检查点退出后才成为 `cancelled` 并释放任务容量。Catalog 会尽力立即把知识库收敛为 `FAILED`/`index_cancelled`，因此知识库可能已经是 `FAILED`，而 worker job 仍是 `cancelling`。如果该终态写入失败或进程中断，耐久 `CANCELLING` 会保留；下一次启动或同一创建请求的幂等重放会完成收敛，不会重新提交索引。重放会返回新的可轮询 job ID。

如果取消到达时知识库已经耐久 `READY`，请求已经太迟，Catalog 不会回退。尚未从 worker 返回的 job 可能短暂显示 `cancelling`，但完成结果获胜，最终状态为 `succeeded`。job 快照和 ID 始终是进程内状态；重启后旧 job ID 失效，即使对应的知识库取消意图已安全收敛。

## 容量与性能规划

`/data` 同时容纳：上传原文、SQLite 目录、Chroma 向量数据和 Hugging Face 模型缓存。索引体积取决于切分数量、向量维度、元数据和 Chroma 实现，不能用一个固定倍率准确预测。

- 用代表性语料做基准导入，记录导入前后 `/data` 增量、峰值内存、耗时与查询延迟。
- 以实际高水位为基线，预留重建索引、临时文件和至少 20% 的磁盘余量。
- `RAG_JOB_WORKERS` 会提高并发索引的 CPU 与内存峰值；它不能超过压测确认的安全值。
- `RAG_MAX_JOBS` 是节点内保留的 active 与 terminal job 快照总上限；`RAG_MAX_JOBS_PER_TENANT` 对同样口径施加每租户上限。终态记录会按 TTL/LRU 回收。
- 每租户 job 数量限制防止一个租户占满全部快照，但不提供 worker 公平调度；强 QoS 仍需要外置公平队列或独立资源池。
- `RAG_MAX_CONCURRENT_ANSWERS` 是节点级非阻塞回答闸门；没有空槽时立即返回 `503`，不会让长请求在应用内无限排队。
- CPU 或内存限制变化后重新测试摄取与问答并发。OOM 会直接终止进程，重启策略不能修复容量不足。
- 模型缓存可重新下载，不属于权威业务数据，但清理它会增加下一次冷启动或首次查询延迟。

单节点达到磁盘、写入并发、恢复时间或可用性上限时，停止纵向堆叠实例，按部署文档的外置状态路径演进。

## 故障排查

### 容器无法启动

```bash
docker compose config --quiet
docker compose ps -a
docker compose logs --tail=300 api
```

优先检查：`.env` 是否存在；`RAG_API_KEYS_JSON` 是否为合法 JSON；`RAG_PERSIST_DATA` 与 `/data` 覆盖是否生效；发布端口是否占用；内存是否足够。不要把完整环境变量输出到共享日志。

### `/data` 权限错误

命名卷首次创建时会继承镜像中的 UID/GID `10001:10001`。绑定宿主目录时必须由管理员把精确目标目录授权给该 UID/GID，并确认目录不是符号链接。不要改成 root 运行，也不要递归修改未经核对的上级目录。

### 模型下载或首次查询失败

检查容器 DNS、HTTPS 出站、防火墙、代理和 Hugging Face 限流；需要时配置 `HF_TOKEN`。缓存位于 `/data/cache/huggingface`。不要把供应商 API 503 误判为本地 JSON 解析错误，应结合结构化错误类别和供应商状态页判断。

### SQLite locked、目录损坏或反复重启

确认只有一个容器连接该卷，没有备份程序在写模式打开文件，也没有第二套 Compose 指向同一卷。停止服务后再备份现状，保留证据并从最近一次已演练快照恢复。不要在生产数据上直接运行未经验证的 SQLite 或 Chroma 修复命令。

### 磁盘耗尽

先停止新的导入，定位 `/data` 各目录和容器日志占用。可以在服务停止后清理可再下载的模型缓存，但不要手工删除 `vector`、`documents` 或 `catalog.sqlite3` 的一部分；三者必须保持一致。扩容或迁移后再恢复写入。

## 数据删除与退役

知识库删除应始终走受认证的 API，让目录、原始文件和向量集合按同一资源身份处理。删除后验证资源无法通过原租户或其他租户访问。

逻辑删除不等于介质级不可恢复擦除：SQLite 空闲页、Chroma 段、容器存储、备份、云快照和 SSD 的写时复制都可能保留旧字节。严格删除要求应采用静态加密与每租户/每环境密钥策略，并把销毁加密密钥、过期备份、删除云快照和底层介质处置纳入流程。

整套环境退役时，先停止服务、执行保留策略并核对目标卷。以下操作不可恢复，只能在确认精确卷名为当前环境后由授权人员执行：

```bash
docker compose down
docker volume inspect rag-studio-data
# 人工核对 inspect 结果后，才可执行：docker volume rm rag-studio-data
```

不要使用 `docker compose down -v` 作为日常清理命令，它会绕过逐卷核对。删除生产卷后，还必须按保留策略处理离机备份、日志、导出文件和密钥。
