# 会话记忆设计与使用说明

## 1. 目标与边界

Research Copilot 为每篇论文维护可长期浏览的对话，同时支持多个独立线程和精确的多论文工作区。刷新页面、关闭浏览器或重启 Streamlit 后，消息、回答模式、执行过程和引用仍可恢复。

会话记忆只解决“用户之前问过什么、当前追问指代什么”。论文事实仍必须从当前 active PDF version 重新检索。模型回答不会写入 Chroma，也不能因为出现在历史记录中就变成论文证据。

当前版本是本地单用户 Demo：没有账号、多租户、云同步和数据库加密。请依赖本机文件权限保护 `data/`。

## 2. 四层状态为什么分开

| 存储层 | 保存内容 | 是否为长期可信来源 |
|---|---|---|
| `st.session_state` | 当前页面控件、临时重试问题 | 否，刷新或会话结束后可丢失 |
| SQLite `app.db` | 工作区、可见消息、过程、回答 payload、引用快照、摘要 | 是，可见会话的唯一可信来源 |
| LangGraph `checkpoints.db` | 标准 Agent messages、Tool 状态、HITL 中断 | 仅作为 Agent 执行状态 |
| Chroma | active 论文版本的原文 chunks 与 embedding | 是，论文问答的事实来源 |

这种划分解决了两个常见问题：快速/深度模式不经过 Agent Checkpoint 也能持久化；Agent 对话压缩不会删掉页面上的完整历史。

## 3. 工作区与会话

### 3.1 工作区

- 未选择论文：`general`，用于 arXiv、导入和论文库操作。
- 单论文家族：`paper:<main_paper_id>`。
- 多论文集合：`papers:<sorted_main_ids>`。

Supplementary document 会解析到主论文根 ID。选择主论文、Supplementary document，或同时选择两者，都会进入同一个 `paper:<root-paper-id>` 工作区；实际检索覆盖家族内所有 ready 文档，引用仍保留具体来源 PDF。

多论文根 ID 会排序和去重，所以 `A+B` 与 `B+A` 是同一工作区。不同论文集合之间不共享线程。相关单论文会话只在折叠区域提供浏览入口，不自动注入多论文上下文。

### 3.2 多线程

一个工作区可以有多个会话。切换论文范围时，界面加载该精确工作区最近的未归档会话；没有历史时，提交第一条问题才创建线程。标题由第一条问题确定性截取生成，不产生额外 API 调用。

同一线程可以逐轮切换 `quick`、`standard_agent` 和 `deep_analysis`。三种模式共同写入 SQLite 时间线，标准 Agent 额外把内部 Tool/HITL 状态写入同 ID 的 LangGraph Checkpoint。

## 4. 一轮问答的数据流

```text
提交问题
  → 立即插入 completed 用户消息
  → 插入 pending 助手消息并锁定 pending_turn_id
  → 必要时将省略式追问改写为独立问题
  → RAG / Agent / Deep Analysis
  → 持续更新 process_json
  → 成功：回答 + payload + citations + trace，状态 completed
  → 失败：错误摘要与过程，状态 failed，可重试
  → HITL：状态 interrupted，Checkpoint 保存可恢复中断
```

同一会话只允许一个未结束 turn。`turn_id` 将用户消息和助手消息配对，`sequence` 保证稳定顺序。Streamlit rerun 不会重新提交已写入的 turn。

助手 payload 带 `schema_version`。使用过的 Citation 同时写入 `message_citations`，包含论文 ID、标题、版本、PDF 页码、chunk ID 和证据原文。历史出处不依赖 Chroma 中的 chunk 永远存在。

## 5. 智能追问与长会话

完整问题直接进入 RAG，不额外调用上下文模型。检测到“它、这个方法、上述、那实验呢、再详细一点、与前者相比”等指代表达时，才读取结构化摘要与最近 8 个完整问答轮次，通过结构化输出生成 `standalone_question`。

界面保存并显示用户原问题；独立问题用于检索并写入 retrieval trace。上下文模型只能补齐指代，不允许回答论文事实。若改写调用失败，系统使用最近用户问题进行确定性降级。

第 12 个完成问答轮次后生成首个结构化摘要，之后每 6 轮增量更新。摘要只允许包含用户目标、讨论主题、术语指代、明确偏好和未解决问题。

标准 Agent 另外使用 LangChain Summarization/Context Editing：90,000 tokens 或 40 messages 触发，保留最近 16 条并清理旧 Tool 大结果。该压缩不影响 SQLite 中的完整可见历史。

## 6. 数据表

### `conversations`

保存 `thread_id`、`scope_type`、`scope_key`、标题、摘要、归档时间、只读原因、最近消息时间和未完成 turn。`thread_id` 同时用于 LangGraph Checkpoint。

### `conversation_papers`

保存会话对应的论文家族根 ID、标题和版本快照。该关系不随论文删除级联消失。

### `conversation_messages`

保存消息 ID、turn、顺序、角色、问答模式、状态、原问题、独立问题、正文、过程、payload、错误、trace 和时间戳。状态包括 `pending`、`running`、`interrupted`、`completed`、`failed`、`rejected` 和 `legacy_incomplete`。

### `message_citations`

保存已使用引用的不可变快照。它不作为新问题的检索索引，只用于准确重放历史答案。

## 7. 版本、删除与恢复

论文版本不进入 scope。历史回答继续显示当时版本；新问题始终检索 active version。版本不一致时界面显示提示。

删除论文前，论文库会显示关联 supplementary 和受影响会话数量：

- 主论文仍有关联 supplementary 时阻止直接删除；
- 删除后，相关单论文和多论文线程自动归档并设为只读；
- 历史消息、过程和 Citation 快照保留；
- 多论文线程只要缺少一个成员就整体只读，避免悄悄改变比较范围。

归档不会删除数据或 Checkpoint。永久删除需要输入会话标题二次确认，并清除业务消息、引用和对应 Checkpoint，无法恢复。

HITL 导入中断保存于 Checkpoint，业务助手消息标为 `interrupted`。应用重启后，界面通过相同 `thread_id` 读取 Agent state，恢复确认/拒绝按钮。

## 8. 数据迁移与备份

`SQLiteRepository` 使用 `PRAGMA user_version` 执行幂等迁移。旧 schema 第一次升级前，将一致性备份写入：

```text
data/backups/app-<UTC timestamp>.sqlite.bak
data/backups/checkpoints-<UTC timestamp>.sqlite.bak
```

迁移会规范化已有 `active_paper_ids_json`，并尝试从最新 Checkpoint 恢复旧标准 Agent 的用户消息和最终 AI 消息。旧快速/深度模式若只有 retrieval trace，则只恢复问题和 trace，并标记 `legacy_incomplete`；旧版从未保存的回答正文无法事后恢复。

若升级失败，停止 Streamlit，将当前数据库复制到安全位置，再用对应 `.bak` 恢复。不要在服务运行和 WAL 活跃时直接覆盖数据库。

## 9. 用户操作

1. 在“智能对话”选择论文范围。
2. 从“会话”下拉框打开历史，或点击“新建会话”。
3. 每轮可自由选择快速、标准 Agent 或深度分析。
4. 展开“查看执行过程”查看可观测步骤；这里不包含隐式思维链。
5. 失败记录点击“重试本轮”。
6. 在“会话管理与状态”中修改标题、查看版本或只读提示。
7. 使用“归档”隐藏非活跃线程；需要彻底清除时再永久删除。

## 10. 开发接口与测试

核心入口为 `ConversationMemoryService`：

- `resolve_scope`：工作区规范化和论文家族展开；
- `create_conversation` / `get_or_create_conversation`；
- `resolve_question`：按需生成独立问题；
- `begin_turn` / `progress` / `complete_turn` / `fail_turn`；
- `archive` / `rename` / `delete_permanently`；
- `import_legacy_checkpoints`。

默认离线验证：

```powershell
python -m pytest -m "not online"
python -m ruff check src streamlit_app.py tests
```

会话测试覆盖 scope 顺序无关、supplementary 归并、消息/引用持久化、失败恢复、指代触发、只读删除、Checkpoint 清理和旧 schema 备份迁移。

## 11. 已知限制

- 当前不支持多用户隔离和会话跨设备同步。
- SQLite 内容未加密。
- 快速/深度任务执行过程中若 Python 进程被强制终止，不能从模型调用中点续跑；该 turn 会保留并可重试。
- HITL 只有标准 Agent 可以从 Checkpoint 精确恢复。
- 旧版未写入 SQLite/Checkpoint 的回答无法恢复。
- 上下文指代检测以中文论文问答表达为主，后续可扩展英文追问模式。
