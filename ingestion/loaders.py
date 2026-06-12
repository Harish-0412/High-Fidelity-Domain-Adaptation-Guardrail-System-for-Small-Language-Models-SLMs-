from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentPage:
    source_id: str
    source_path: str
    page: int
    text: str


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def discover_documents(corpus_path: str | Path) -> list[Path]:
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {path}")
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_document(path: str | Path) -> list[DocumentPage]:
    document_path = Path(path)
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(document_path)
    if suffix in {".txt", ".md"}:
        return _load_text(document_path)
    raise ValueError(f"Unsupported document type: {document_path.suffix}")


def _source_id(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def _load_text(path: Path) -> list[DocumentPage]:
    return [
        DocumentPage(
            source_id=_source_id(path),
            source_path=str(path),
            page=1,
            text=path.read_text(encoding="utf-8"),
        )
    ]


def _load_pdf(path: Path) -> list[DocumentPage]:
    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF ingestion requires PyMuPDF. Install optional dependencies with "
            "`pip install -e .[retrieval]` or convert the PDF to text/markdown."
        ) from exc

    pages: list[DocumentPage] = []
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf, 1):
            pages.append(
                DocumentPage(
                    source_id=_source_id(path),
                    source_path=str(path),
                    page=page_index,
                    text=page.get_text("text"),
                )
            )
    return pages

