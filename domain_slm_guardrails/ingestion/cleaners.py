from __future__ import annotations

import re

from domain_slm_guardrails.ingestion.loaders import DocumentPage


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_page(page: DocumentPage) -> DocumentPage:
    return DocumentPage(
        source_id=page.source_id,
        source_path=page.source_path,
        page=page.page,
        text=clean_text(page.text),
    )

