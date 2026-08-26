# Public Update Notes

This update synchronizes the public code copy with the current local implementation while keeping runtime data and research materials out of Git.

## Reliability and recovery

- Chroma now uses explicit HNSW settings and supports `CHROMA_DATA_DIR`, so the vector index can live outside a Windows repository path containing non-ASCII characters.
- `research-copilot repair-chroma` rebuilds into a staging directory and starts a fresh Python process to reopen and query the index before any production swap.
- A malformed LangGraph checkpoint containing an unpaired tool call can be detected and reset without deleting SQLite-backed visible conversation history. Intentional Human-in-the-loop pauses are preserved.

## RAG and agent behavior

- Full-paper summary generation now keeps each map response compact, splits an oversized segment when the provider reaches a completion limit, and reduces evidence locally. This avoids a second oversized structured response.
- Explicit requests to find and import a Mamba medical-image-segmentation paper use a deterministic workflow: one arXiv search, topic/year filtering, duplicate detection against the local library, then Human-in-the-loop confirmation only for a new candidate.
- arXiv PDF download no longer relies on the removed `Result.download_pdf` helper.

## Verification

The offline test suite covers duplicate-aware arXiv candidate selection, checkpoint recovery boundaries, bounded summary behavior, HNSW repair behavior, and direct import result handling. Tests use fake providers by default and do not require API keys.

## Privacy boundary

No PDF, parsed paper content, vectors, SQLite databases, checkpoints, API keys, local paths, private evaluation samples, or user conversations were added in this update.
