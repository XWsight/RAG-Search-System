# Evaluation data governance

仓库内评测数据是版本化质量资产，不是为了让指标好看而编写的演示题。任何新增、删除、改写或重新标注都必须与代码变更分开审查，并解释数据集摘要变化。

## Retrieval suite contract

[`retrieval_suite.json`](retrieval_suite.json) 使用“语义家族”而不是把同义改写冒充独立知识点：

- 一个 family 表示一个独立检索意图和一组固定相关来源；
- `questions` 是该意图的不同自然表达，只衡量改写鲁棒性；
- `relevance` 使用来源文件名和 1–3 级人工相关性；
- `expected_route` 只能是 `local`、`refused` 或 `web`；
- `allow_web` 仅在期望 `web` 时为 `true`；
- 同一来源文件只能属于 development、validation、test 中一个 split；
- test 用于冻结后的最终复核，不用于挑选阈值、模型或融合权重。

当前最低覆盖由 manifest 自身声明。严格 loader 会检查精确字段、问题去重、路由一致性、安全相对路径、普通来源文件、source-level split 隔离和覆盖矩阵。验证命令：

```powershell
python scripts\validate_retrieval_suite.py evals\retrieval_suite.json `
  --contract evals\gates\retrieval-suite.json
```

## Answer suite contract

[`answer_suite.json`](answer_suite.json) 的每个 case 都是独立问题，不用同义改写扩充数量。每个样例同时冻结：输入证据、必须覆盖的原子事实、事实允许的引用、是否应拒答、category、difficulty、risk tags 和 split。严格 loader 会拒绝重复问题或证据、歧义事实、无效引用、错误拒答标签、重复风险标签和覆盖不足。

```powershell
python scripts\validate_answer_suite.py evals\answer_suite.json `
  --contract evals\gates\answer-suite.json
```

仓库套件当前为 50 cases / 70 facts，其中 development/validation/test 为 20/15/15。公开 test 只能作为冻结回归集，不能声称为保密盲测；当前标注由仓库维护者整理，尚未完成双人独立标注和第三方裁决，因此也不能宣称为领域级金标准。真实模型运行保持手动，以避免云端费用和非确定性进入 CI。

## Annotation procedure

1. 先确定真实用户意图、目标 split 和候选语料，再编写问题；不要先看系统输出。
2. 两名标注者独立判断相关来源和等级。直接回答核心事实标为 3，必要但不充分的支持标为 2，仅有背景价值标为 1。
3. 分歧必须记录理由并由第三方或共同复核裁决。不能为了让当前检索器通过而删除合理相关来源。
4. 每个 family 的问法应覆盖自然措辞、术语表达、压缩表达和至少一种较困难改写，但不得只是交换标点或语序。
5. 无答案题必须确认 corpus 确实没有足够证据。时间敏感且明确允许联网的题标为 `web`；未允许联网、私有资源或系统无权访问的问题标为 `refused`。
6. 新来源先检查版权、隐私、许可和近重复，再提交脱敏后的仓库夹具；禁止提交客户文档、供应商响应或个人数据。

## Split discipline

- development：允许查看逐题结果并开发检索策略；
- validation：只用于选择模型、切分、权重和阈值；
- test：配置冻结后运行，用于确认泛化，不参与调参。

按问题随机切分会让同一文档的相邻事实泄漏到不同集合，因此本套件强制按来源隔离。随着真实授权数据增加，应进一步按租户、时间窗口和主题簇隔离，并维护不可公开的外部盲测集。仓库 test 公开后只能作为冻结回归集，不能声称是真正保密盲测。

## Reporting rules

每份检索结果必须同时报告：commit、suite ID、ground-truth digest、问题数、family 数、来源数、split、模型与切分配置、top-k、全部指标、路由混淆矩阵、按 split/category/difficulty/expected route 的切片，以及逐题失败。路由调优还必须保留不含正文的首名/次名、分差、ranker agreement、词法支撑和置信度信号，不能只报告最终阈值。回答结果必须报告 commit、suite digest、case/fact 数、split、provider/model 配置、总体五项指标，以及按 category、difficulty、risk tag 切分的全部结果和失败 case；不得把 50 个 case 描述为真实业务准确率，也不得隐藏较差切片。

`retrieval-suite.json` 冻结 manifest、全部 corpus 内容摘要和覆盖矩阵；BM25 full-suite gate 冻结确定性指标下限；Hybrid gate 是需要本地模型的手动下限。更新任何契约都必须附带失败样例、人工复核和明确理由，不能只修改 gate 让 CI 变绿。
