# Research Copilot

Research Copilot 是一个面向本地 PDF 与 arXiv 论文的单机科研助手。它把论文内容转化为可检索证据，并让回答回链到真实 PDF 页码、原文片段与导入版本。

> 本仓库不包含真实论文、用户对话、向量库、API 密钥或私有评测数据。请仅导入你有权处理和上传的论文。

## 核心能力

- 导入本地 PDF、Streamlit 上传文件或 arXiv 论文，并进行 SHA-256 去重、版本管理与失败回滚；
- 默认基于 PyMuPDF 在本地解析文本与渲染页面图片；可选启用 MinerU，失败时自动回退本地解析；
- 使用 DashScope Embedding 与持久化 Chroma，为每篇论文的 active version 建立隔离的向量索引；
- 对“搜索并导入 Mamba 医学分割论文”这类明确请求，使用一次检索、年份/主题筛选、已入库去重和 Human-in-the-loop 确认的确定性流程；
- 单论文问答、全文 Map-Reduce 摘要和逐论文取证的多论文九维比较；
- 以 `[C#]` / `[F#]` 形式返回程序校验过的文本与页面图像证据，拒绝跨论文、跨版本和无证据的确定性引用；
- 提供快速 RAG、标准 Agent 与 LangGraph 多 Agent 深度分析三种模式；
- 按单论文家族、精确多论文集合或通用工作区持久化会话，支持追问改写、重启恢复、归档与引用快照；
- 能识别并清理异常中断后遗留的无效 Agent Tool-Call checkpoint，保留可见的 SQLite 会话记录并安全重试；
- 使用 Streamlit 提供论文库、智能对话、多论文比较、检索调试与 RAG 评测页面。

## 架构概览

```text
PDF / arXiv
    -> 校验、解析、页内分块、Embedding
    -> Chroma（论文证据向量） + SQLite（论文/会话/任务）
    -> PaperRAGService（检索、重排、引用校验）
    -> 快速 RAG / 标准 Agent / 深度分析
    -> Streamlit
```

Chroma 仅保存论文分块及其向量，用于检索；SQLite 保存论文元数据、后台任务、会话、消息与历史引用快照。模型回答不会回写为论文知识，新问题始终重新检索当前 active version。

## 快速开始

### 1. 创建环境并安装

```powershell
git clone <YOUR_REPOSITORY_URL>
Set-Location research-copilot

conda create -n research-copilot python=3.13
conda activate research-copilot
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

在 `.env` 中至少配置：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_key
DASHSCOPE_API_KEY=your_dashscope_key
DASHSCOPE_BASE_URL=your_dashscope_compatible_base_url
```

`DEEPSEEK_API_KEY` 用于默认 Chat 模型，`DASHSCOPE_API_KEY` 用于 Embedding 与可选备用模型。若启用 MinerU，再填写 `MINERU_API_TOKEN`；请确认你拥有将 PDF 上传至该第三方服务的授权。

若在 Windows 上将仓库放在含中文或其他非 ASCII 字符的目录，请额外为 Chroma 配置一个全英文绝对路径，避免 HNSW 持久化索引无法在重启后打开：

```dotenv
CHROMA_DATA_DIR=C:\\ResearchCopilotData\\chroma
```

### 2. 启动应用

```powershell
research-copilot serve
```

或：

```powershell
python -m streamlit run streamlit_app.py
```

打开终端显示的本地地址，在“论文库与导入状态”页上传 PDF，等待状态变为 `ready` 后即可在“智能对话”页提问。

## 常用 CLI

```powershell
# 导入本地 PDF；默认完全本地解析
research-copilot ingest .\paper.pdf

# 显式允许使用 MinerU 解析
research-copilot ingest .\paper.pdf --use-mineru

# 查看已导入论文
research-copilot list

# 带 PDF 页码证据的问答
research-copilot ask '这篇论文的核心方法是什么？' --paper-id <paper-id>

# 深度分析与多论文比较
research-copilot deep '分析方法、训练策略和实验设计' --paper-id <paper-id>
research-copilot compare --paper-id <paper-a> --paper-id <paper-b>

# 显式在线配置检查：会调用外部 API
research-copilot smoke
```

若应用提示 HNSW 索引无法读取，先完全停止 Streamlit，再运行下列命令。它会在 staging 目录重建、由全新 Python 进程重新打开并查询验证，成功后才替换旧索引：

```powershell
research-copilot repair-chroma
```

## 测试

```powershell
# 默认离线，不访问模型、Embedding、MinerU 或 arXiv
python -m pytest -m "not online"

# 代码风格检查
python -m ruff check .
```

`data/eval/questions.example.jsonl` 是评测 JSONL 格式示例。请按自己导入论文的 `paper_id`、PDF 页码和拒答标注创建私有评测集；真实评测数据应保存在被 Git 忽略的位置。

## 数据与安全

- `.env`、真实 PDF、解析 Markdown/JSON/页面图片、Chroma、SQLite、Checkpoint、上传文件、日志及评测报告均被 Git 忽略；
- 请勿提交 API Key、访问令牌、私人对话或无权公开的论文；若密钥已泄露，请先在对应平台立即撤销并重新生成；
- PDF 的文件物理页码可能与论文印刷页码不同，界面统一显示为“PDF 第 N 页”；
- 这是本地单用户 Demo，当前不提供账号体系、云同步或多租户隔离。
- 使用 `CHROMA_DATA_DIR` 时，该目录同样属于本机私有运行数据，不能提交到 Git。

## 文档

- [系统架构](docs/ARCHITECTURE.md)
- [设计决策：Tools、Prompt、Middleware 与记忆](docs/DESIGN.md)
- [会话记忆设计、使用与恢复](docs/CONVERSATION_MEMORY.md)
- [测试与评测](docs/TESTING.md)
- [演示脚本](docs/DEMO.md)
- [代码导读与面试说明](docs/CODE_WALKTHROUGH_INTERVIEW.md)
- [已知限制](docs/KNOWN_LIMITATIONS.md)
- [GitHub 发布前检查清单](docs/GITHUB_PUBLISHING.md)
- [本次公开版更新说明](docs/RELEASE_NOTES.md)

## License

尚未声明开源许可证。公开仓库前，请由仓库所有者根据代码复用、署名和商业使用需求选择并添加合适的 `LICENSE` 文件；在未声明许可证前，其他人默认不享有复制、修改或分发本项目代码的授权。
