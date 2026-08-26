# 测试与评测

## 默认离线测试

```powershell
python -m pytest -m "not online"
python -m ruff check .
```

离线测试使用 fake model 与 fake embedding，不访问 Chat、Embedding、MinerU 或 arXiv，也不会产生 API 费用。覆盖范围包括：

- PDF 扩展名、magic bytes、可打开性、页码与页内稳定 chunk ID；
- SHA-256 去重、版本管理、失败回滚与 Chroma 的论文/版本过滤；
- 引用白名单、无引用拒答、摘要/引言优先与参考文献降权；
- arXiv ID/URL、Tool schema、调用范围、重试白名单与 SecretStr 脱敏；
- 持久化后台任务、会话消息、引用快照、失败重试、归档及论文删除生命周期；
- 评测 JSONL 的格式校验、论文范围、页码与拒答标注。

## 在线 smoke test

```powershell
research-copilot smoke
```

该命令显式验证 Chat、结构化输出、Tool Calling 和 Embedding 配置，可能产生外部 API 费用。

若要测试 MinerU，需要确认你有权将目标 PDF 上传到该服务：

```powershell
research-copilot smoke --mineru-pdf .\paper.pdf
```

## 私有 RAG 评测集

本仓库仅提供 `data/eval/questions.example.jsonl` 格式示例。请基于你有权处理的已导入论文，创建一个不提交 Git 的评测文件，例如：

```json
{"question":"这篇论文的核心方法是什么？","paper_ids":["your-paper-id"],"expected_pages":[2,3],"should_refuse":false}
```

字段含义：

- `question`：用户问题；
- `paper_ids`：本轮允许检索的论文 ID；
- `expected_pages`：预期命中的 PDF 物理页码；
- `should_refuse`：论文缺少证据时是否应拒答。

运行命令：

```powershell
# 只校验评测数据，不产生 API 费用
research-copilot validate-eval --dataset .\data\eval\questions.private.jsonl

# 检索评测：调用 Embedding API
research-copilot evaluate-retrieval --dataset .\data\eval\questions.private.jsonl

# 比较不同分块参数：调用 Embedding API，不修改生产 Chroma
research-copilot benchmark-chunking --dataset .\data\eval\questions.private.jsonl

# 端到端答案与引用评测：调用 Chat 与 Embedding API
research-copilot evaluate --dataset .\data\eval\questions.private.jsonl
```

报告会写入 Git 忽略的 `data/reports/`。建议记录以下指标：

- Hit@k：返回证据页与预期页是否有交集；
- 论文范围准确率：Citation 是否都属于本轮 `paper_ids`；
- 引用有效率：chunk ID 是否与 Citation 的论文、版本一致；
- 拒答准确率：`insufficient_evidence` 是否符合标注；
- 延迟：每条问题端到端耗时。

论文的 active version 更新后，应人工复核 `expected_pages`。不要把无权公开的论文内容、私人实验数据或完整评测结果提交到公开仓库。
