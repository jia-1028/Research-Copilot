class ResearchCopilotError(Exception):
    """Base domain exception safe to present to the user."""


class ConfigurationError(ResearchCopilotError):
    pass


class InvalidPdfError(ResearchCopilotError):
    pass


class ParserError(ResearchCopilotError):
    pass


class PaperNotReadyError(ResearchCopilotError):
    pass


class CitationValidationError(ResearchCopilotError):
    pass


class UnsafePathError(ResearchCopilotError):
    pass


class ArxivTemporarilyUnavailableError(ResearchCopilotError):
    """A remote arXiv outage or rate limit that is safe to show in the UI."""

    def __init__(self, status_code: int, retry_after_seconds: int):
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        if status_code == 429:
            message = (
                "arXiv 当前限流（HTTP 429）。系统已停止本轮重复搜索，"
                f"请至少等待 {retry_after_seconds} 秒后再试；也可以改为上传本地 PDF。"
            )
        else:
            message = (
                f"arXiv 服务暂时不可用（HTTP {status_code}）。系统已停止本轮重复搜索，"
                f"建议约 {retry_after_seconds} 秒后重试。"
            )
        super().__init__(message)
