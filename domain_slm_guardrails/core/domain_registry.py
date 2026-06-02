from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain_slm_guardrails.core.config import load_base_config, load_simple_yaml, project_root


REQUIRED_DOMAIN_FIELDS = {
    "domain_id",
    "name",
    "corpus_path",
    "processed_path",
    "index_path",
    "index_name",
}


@dataclass(frozen=True)
class DomainConfig:
    domain_id: str
    name: str
    corpus_path: Path
    processed_path: Path
    index_path: Path
    index_name: str
    root: Path
    settings: dict[str, Any]

    @property
    def chunks_path(self) -> Path:
        return self.processed_path / "chunks.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.index_path / "bm25.pkl"

    @property
    def dense_index_path(self) -> Path:
        return self.index_path / "dense_vectors.jsonl"


def _resolve_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _domain_config_path(domain_id: str, root: Path) -> Path:
    return root / "domains" / domain_id / "domain.yaml"


def list_domains(root: Path | None = None) -> list[str]:
    root = root or project_root()
    domains_dir = root / "domains"
    if not domains_dir.exists():
        return []
    return sorted(
        item.name
        for item in domains_dir.iterdir()
        if item.is_dir() and (item / "domain.yaml").exists()
    )


def get_domain_config(domain_id: str, root: Path | None = None) -> DomainConfig:
    root = root or project_root()
    path = _domain_config_path(domain_id, root)
    if not path.exists():
        known = ", ".join(list_domains(root)) or "none"
        raise ValueError(f"Unknown domain '{domain_id}'. Known domains: {known}")

    base = load_base_config(root)
    domain = load_simple_yaml(path)
    missing = REQUIRED_DOMAIN_FIELDS.difference(domain)
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(sorted(missing))}")

    settings = {**base, **domain}
    return DomainConfig(
        domain_id=str(domain["domain_id"]),
        name=str(domain["name"]),
        corpus_path=_resolve_path(root, domain["corpus_path"]),
        processed_path=_resolve_path(root, domain["processed_path"]),
        index_path=_resolve_path(root, domain["index_path"]),
        index_name=str(domain["index_name"]),
        root=root,
        settings=settings,
    )

