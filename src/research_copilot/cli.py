from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer

from research_copilot.config import PROJECT_ROOT, get_settings
from research_copilot.evaluation import (
    benchmark_chunking,
    evaluate_jsonl,
    evaluate_retrieval_jsonl,
    validate_eval_dataset,
)
from research_copilot.services import ServiceContainer, build_services
from research_copilot.smoke import run_online_smoke_tests
from research_copilot.vector_repair import repair_chroma_index

app = typer.Typer(no_args_is_help=True, help="Research Copilot phase-1 CLI")


def emit(value: object) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))


@contextmanager
def services_runtime() -> Iterator[ServiceContainer]:
    services = build_services()
    try:
        yield services
    finally:
        services.close()


@app.command()
def smoke(mineru_pdf: Path | None = typer.Option(None, exists=True, dir_okay=False)) -> None:
    """Run paid/network model smoke tests explicitly."""
    emit(run_online_smoke_tests(get_settings(), mineru_pdf=mineru_pdf))


@app.command()
def ingest(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False),
    title: str | None = None,
    use_mineru: bool = typer.Option(False, "--use-mineru"),
) -> None:
    with services_runtime() as services:
        emit(
            services.ingestion.ingest_local(
                pdf, title=title, prefer_mineru=use_mineru
            ).model_dump(mode="json")
        )


@app.command("list")
def list_papers(include_non_ready: bool = False) -> None:
    with services_runtime() as services:
        emit(services.repository.list_papers(status=None if include_non_ready else "ready"))


@app.command("render-page-images")
def render_page_images(paper_id: list[str] = typer.Option([], "--paper-id")) -> None:
    """Render local PDF page images for multimodal RAG without MinerU."""
    with services_runtime() as services:
        targets = paper_id or [
            item["paper_id"] for item in services.repository.list_papers(status="ready")
        ]
        emit({item: services.library.render_page_images(item) for item in targets})


@app.command()
def ask(question: str, paper_id: list[str] = typer.Option(..., "--paper-id")) -> None:
    with services_runtime() as services:
        emit(services.rag.ask(question, paper_id).model_dump(mode="json"))


@app.command()
def deep(question: str, paper_id: list[str] = typer.Option(..., "--paper-id")) -> None:
    """Run bounded supervisor/specialist deep paper analysis."""
    with services_runtime() as services:
        emit(services.deep_analysis.analyze(question, paper_id).model_dump(mode="json"))


@app.command()
def compare(paper_id: list[str] = typer.Option(..., "--paper-id")) -> None:
    with services_runtime() as services:
        emit(services.rag.compare(paper_id).model_dump(mode="json"))


@app.command()
def evaluate(dataset: Path = typer.Option(..., exists=True, dir_okay=False)) -> None:
    with services_runtime() as services:
        output = services.settings.project_data_dir / "reports" / "evaluation.json"
        emit(evaluate_jsonl(services.rag, dataset, output))


@app.command("validate-eval")
def validate_eval(
    dataset: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Validate paper IDs, PDF pages, size and refusal balance without API calls."""
    with services_runtime() as services:
        result = validate_eval_dataset(services.repository, dataset)
        emit(result)
        if not result["valid"]:
            raise typer.Exit(1)


@app.command("evaluate-retrieval")
def evaluate_retrieval(
    dataset: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    top_k: int = typer.Option(8, min=1, max=30),
) -> None:
    """Evaluate the current dense index without calling the answer model."""
    with services_runtime() as services:
        output = services.settings.project_data_dir / "reports" / "retrieval-evaluation.json"
        emit(evaluate_retrieval_jsonl(services.rag, dataset, output, top_k=top_k))


@app.command("benchmark-chunking")
def benchmark_chunking_command(
    dataset: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    top_k: int = typer.Option(8, min=1, max=30),
) -> None:
    """Paid dense benchmark for 800/120, 1200/180 and 1500/200."""
    with services_runtime() as services:
        output = services.settings.project_data_dir / "reports" / "chunking-benchmark.json"
        emit(
            benchmark_chunking(
                services.settings,
                services.repository,
                services.vector_index.embedding_model,
                dataset,
                output,
                top_k=top_k,
            )
        )


@app.command("repair-chroma")
def repair_chroma() -> None:
    """Rebuild all active paper vectors in staging, validate, then swap atomically."""
    messages: list[str] = []

    def progress(message: str) -> None:
        messages.append(message)
        print(message, flush=True)

    result = repair_chroma_index(get_settings(), progress_callback=progress)
    emit({**result, "steps": messages})


@app.command()
def serve() -> None:
    """Start the Streamlit demo."""
    raise typer.Exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "streamlit_app.py")],
            check=False,
        ).returncode
    )


if __name__ == "__main__":
    app()
