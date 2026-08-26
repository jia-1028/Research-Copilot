from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from research_copilot.middleware import (
    PaperToolPolicyMiddleware,
    ResearchTraceMiddleware,
    is_transient_model_error,
)
from research_copilot.services import ServiceContainer
from research_copilot.tools import build_tools

AGENT_SYSTEM_PROMPT = """你是 Research Copilot 的论文研究助手，负责识别意图并选择高层工具。

核心规则：
- 凡是关于论文内容、方法、实验或结论的问题，必须调用 ask_papers、summarize_paper 或 compare_papers；
  不得凭模型训练记忆直接回答论文事实。
- “实验结果如何”“用了什么数据集”“某个指标是多少”等局部问题必须使用 ask_papers；
  只有用户明确要求整篇论文的完整结构化摘要时才使用 summarize_paper。
- 同一轮最多调用一次 ask_papers。需要方法、结构、损失函数、训练策略等多个细节时，
  必须把它们合并进一个完整 question，并在一次调用中传入全部目标 paper_ids；不得拆成多个并行调用。
- 如果额外的只读论文工具调用被次数限制拦截，不要重试；使用本轮首次成功返回的证据完成回答。
- 不知道 paper_id 时先调用 list_papers。只使用 ready 论文。
- 如果 search_arxiv 返回“限流”或“服务暂时不可用”，本轮不得再次调用 search_arxiv；
  应告知用户等待提示的时间后再试，或建议上传本地 PDF。
- 如果用户消息已经给出“本轮只使用这些 paper_id”，这些 ID 已由界面校验，直接调用对应论文工具，
  不要再调用 list_papers。
- 不要把检索、切分、Embedding、向量写入或引用校验当成 Tool；这些由 Service 保证。
- 导入前向用户说明目标；import_local_paper 和 import_arxiv_paper 会触发人工确认。
- 完整保留工具返回的证据页码、引用与“证据不足”结论，不得二次改写为更确定的说法。
- ask_papers 会直接结束本轮并返回已经完成引用校验的答案，不需要再进行一次模型改写。
- 默认中文回答。当前阶段不使用 MCP，不管理实验代码或模型训练。
"""


def build_agent(services: ServiceContainer, *, enable_hitl: bool = True):
    middleware = [
        ResearchTraceMiddleware(),
        PaperToolPolicyMiddleware(services),
        ModelRetryMiddleware(max_retries=2, retry_on=is_transient_model_error),
        ToolRetryMiddleware(
            max_retries=2,
            tools=["search_arxiv"],
            retry_on=(TimeoutError, ConnectionError),
        ),
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="error"),
        ToolCallLimitMiddleware(run_limit=8, exit_behavior="error"),
    ]
    for expensive_tool in ("import_local_paper", "import_arxiv_paper"):
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name=expensive_tool, run_limit=1, exit_behavior="error"
            )
        )
    middleware.append(
        ToolCallLimitMiddleware(
            tool_name="search_arxiv", run_limit=1, exit_behavior="continue"
        )
    )
    for expensive_read_tool in ("ask_papers", "summarize_paper", "compare_papers"):
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name=expensive_read_tool,
                run_limit=1,
                exit_behavior="continue",
            )
        )
    if enable_hitl:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={"import_local_paper": True, "import_arxiv_paper": True},
                description_prefix="论文导入会下载/上传、解析并产生 API 费用，请确认",
            )
        )
    middleware.extend(
        [
            ContextEditingMiddleware(),
            SummarizationMiddleware(
                model=services.chat_model,
                # The custom DashScope model has no LangChain profile metadata;
                # use an explicit threshold instead of a fractional limit.
                trigger=[("tokens", 90000), ("messages", 40)],
                keep=("messages", 16),
            ),
        ]
    )
    return create_agent(
        model=services.chat_model,
        tools=build_tools(services),
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=services.checkpointer,
        name="research_copilot_phase1",
    )
