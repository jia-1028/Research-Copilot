# Tools、Prompt、Middleware 与记忆

## 八个高层 Tools

| Tool | 作用 | 是否高成本 | 是否 HITL |
|---|---|---:|---:|
| `list_papers` | 查询论文库和状态 | 否 | 否 |
| `import_local_paper` | 导入受控 uploads 中的 PDF | 是 | 是 |
| `search_arxiv` | 检索 arXiv 元数据 | 否；仅连接/超时可重试，429/503 进入冷却 | 否 |
| `import_arxiv_paper` | 下载并导入指定 arXiv PDF | 是 | 是 |
| `get_ingestion_status` | 查询 job/paper 导入进度 | 否 | 否 |
| `ask_papers` | 对指定论文做带证据问答 | 是 | 否 |
| `summarize_paper` | 覆盖全文的 Map-Reduce 画像 | 是 | 否 |
| `compare_papers` | 逐篇画像后对齐比较 | 是 | 否 |

PDF 校验、解析、清洗、分块、Embedding、向量写入、候选检索和引用校验是内部 Service，不允许 Agent 编排底层事务。删除、清空和重建也不注册为 Tool，只能通过 Streamlit 明确按钮直接调用 `PaperLibraryService`。

本地 PDF 未手动填写标题时，导入服务先读取 PDF metadata；metadata 缺失或不可靠时，依据首页顶部连续同字号文本、位置、长度和标题标点识别完整标题，并过滤期刊页眉、`Anonymous submission`、稿件编号等非标题字段。识别失败才回退文件名。标题刷新只更新 SQLite 与 Chroma metadata，不重新生成 Embedding。

## Prompt 合同

论文问答模型收到的证据块格式为：

```text
[C1] 论文=<title>; paper_id=<id>; version=<n>; PDF页=<page>
<原文>
```

基础 RAG 使用普通文本响应和固定 `<EVIDENCE_STATUS>` 尾部，程序在本地解析为 `GroundedAnswerDraft(answer, used_citation_ids, insufficient_evidence, limitations)`，避免实验模型不支持 `response_format` 导致整轮失败。页码、标题、版本与原文不由模型生成，而由程序根据正文中的短引用 ID 回填为 `Citation`。需要 JSON Schema 的论文画像、对比和记忆任务使用百炼 `qwen-plus`。

多模态导入默认绕过 MinerU。PyMuPDF 除逐页提取文字外，还以 120 DPI、JPEG 质量 85 在 `parsed/<paper_id>/v<version>/page_images/` 渲染完整页面；完整页可以保留矢量架构图、caption、公式和版面关系。普通事实问题仍只发送文本；涉及图、表、架构、方法机制或可视化的问题，按文字检索排名选择最多三张页面图像发送给视觉模型。文本证据使用 `[C#]`，页面图像使用 `[F#]`，两类引用都由程序映射并校验到当前论文版本和 PDF 页码。备用文本模型接管时会自动移除图像及 `[F#]` 提示，避免不支持视觉的模型误用视觉引用。

Agent 的系统提示只允许它做意图识别和高层 Tool 选择。凡是论文事实，必须走 `ask_papers`、`summarize_paper` 或 `compare_papers`。

Streamlit 提供三种显式模式。“快速论文问答”直接调用 `PaperRAGService`；标准模式中已选论文的事实问答也直接进入受控 RAG，只有导入、arXiv、论文库和其它高层操作才交给 Agent 规划，从而避免一次没有信息增益的模型调用；“多 Agent 深度分析”调用独立 `DeepAnalysisService`。宽泛的实验问题会通过确定性关键词扩展检索数据集、指标、baseline、定量结果、效率和消融实验，不额外消耗检索规划模型调用。

深度分析使用 LangGraph `StateGraph + Send`：协调 Agent 以 JSON-mode 规划 2–3 个互补任务，specialist 并行执行逐论文检索并生成带 `[T#-C#]` 引用的结构化报告，Reducer 再合并报告与原始证据。程序最终校验引用是否属于本轮 `paper_id + active_version`。Planner 失败时使用固定三维计划；单个 specialist 失败不会取消其它任务；最终协调模型失败时确定性拼接已经通过引用校验的子报告。并发上限固定为三个，避免 API 费用、429 和超时随任务数失控。

对话采用“问题即时回显 → 可折叠执行状态 → 最终答案与出处”的交互。快速 RAG 通过回调发布范围校验、检索、重排、证据页、模型生成和引用校验事件；标准 Agent 路径使用 LangGraph update stream 发布高层 Tool 选择与完成状态；深度模式展示规划出的 specialist 及逐个完成状态。完成后这些事件、助手消息、回答 payload 和引用快照写入 SQLite。界面不展示模型的隐式思维链，也不把密钥或隐式思维过程写入执行记录。

宽泛的“论文在研究什么”类问题会扩展为摘要、研究动机、现有局限、核心方法与贡献检索。候选块随后做确定性重排：研究问题优先 PDF 前三页、Abstract/Introduction 和 `we propose/to address` 等表述；实验问题优先 Table、dataset、DSC/Dice/HD95 等证据；References 标题、高年份/引文密度的参考文献块统一降权。模型只接收重排后的 4–6 个证据块。

多论文比较不再调用昂贵的全文 Map-Reduce 摘要。它对每篇论文分别按九个维度定向检索，使用一次较小的 JSON-mode 请求生成证据画像，并按 `paper_id + active_version + prompt_version` 缓存。完整 `PaperComparison` 也按论文组合和版本缓存。相似维度、差异维度与不可直接比较提示由程序从画像确定性生成，不再为了附加叙述发起模型调用。标题变化、索引重建或论文版本变化会使相关缓存失效。

## Middleware

- `ResearchTraceMiddleware`：只记录模型/Tool 名称、耗时、状态和随机 trace ID；不记录密钥、Prompt、参数或正文。
- `PaperToolPolicyMiddleware`：检查论文 ready 状态、论文数量和 upload ID 路径范围。
- `ModelRetryMiddleware`：仅 timeout、connection、429、5xx，最多两次。
- `ToolRetryMiddleware`：仅幂等 `search_arxiv`。
- `ModelCallLimitMiddleware`：每轮最多 6 次。
- `ToolCallLimitMiddleware`：每轮最多 8 次；每个高成本 Tool 最多一次。导入类 Tool 超限时直接终止；`ask_papers`、`summarize_paper`、`compare_papers` 等只读 Tool 只放行第一次，重复调用转为可恢复的 Tool 错误，让 Agent 使用首次结果继续回答。
- `HumanInTheLoopMiddleware`：本地/arXiv 导入暂停等待确认。
- `ContextEditingMiddleware`：长会话清理旧 Tool 大结果。
- `SummarizationMiddleware`：90,000 tokens 或 40 messages 触发，保留最近 16 条。使用绝对 token 阈值是因为自定义模型没有 LangChain profile metadata。

## 记忆分层

| 层 | 内容 | 生命周期 |
|---|---|---|
| `st.session_state` | 页面选择、临时重试输入 | 浏览器页面 |
| `SqliteSaver` | 标准 Agent messages、Tool 调用、HITL 中断 | thread 持久化 |
| SQLite conversations/messages/citations | 三模式可见历史、过程、引用快照、摘要 | 应用长期状态 |
| Chroma | 当前论文版本的原文 chunks | 论文版本生命周期 |

模型回答不会被写回长期论文知识。这样可以避免把幻觉固化为下一轮“证据”。

工作区、智能追问、迁移与删除语义详见[会话记忆说明](CONVERSATION_MEMORY.md)。

## MCP 决策

第一阶段不调用也不实现 MCP。当前所有能力在同一个 Python/Streamlit 应用内，MCP 只会增加协议、部署和权限面。后续只有在需要让 IDE、桌面客户端或其它 Agent 复用论文库时，才新增只读 MCP 资源/Tools；导入与删除仍应保持显式权限控制。
