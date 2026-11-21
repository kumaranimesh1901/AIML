"""Store policy loading, chunking, embedding, and persistence into policy_documents."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sqlalchemy.orm import Session

from edg.config import _find_project_root
from edg.db.models import PolicyDocument
from edg.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class PolicyChunk(BaseModel):
    policy_id: str
    version: str
    title: str
    body: str
    chunk_index: int


def chunk_markdown(
    *,
    stem: str,
    version: str,
    text: str,
    max_chars: int = 600,
) -> list[PolicyChunk]:
    """Split a markdown policy document into deterministic heading/paragraph chunks."""
    lines = [line.rstrip() for line in text.strip().splitlines()]
    title = stem.replace("_", " ").title()
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # Split into logical blocks by headings or blank lines
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("## ") and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        elif not line.strip() and current:
            blocks.append("\n".join(current).strip())
            current = []
        elif line.strip():
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())

    # Merge adjacent small blocks up to max_chars
    merged: list[str] = []
    buf = ""
    for block in blocks:
        if not buf:
            buf = block
        elif len(buf) + len(block) + 2 <= max_chars:
            buf = f"{buf}\n\n{block}"
        else:
            merged.append(buf)
            buf = block
    if buf:
        merged.append(buf)

    return [
        PolicyChunk(
            policy_id=f"{version}:{stem}:chunk_{idx}",
            version=version,
            title=title,
            body=chunk_body,
            chunk_index=idx,
        )
        for idx, chunk_body in enumerate(merged)
    ]


def hash_embed_texts(texts: list[str], dim: int = 384) -> np.ndarray:
    """Deterministic unit-normalized token-hash embedding (offline/test fallback)."""
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            continue
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            mat[row_idx, bucket] += sign
        norm = float(np.linalg.norm(mat[row_idx]))
        if norm > 0.0:
            mat[row_idx] /= norm
    return mat


def embed_policy_texts(
    texts: list[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    dim: int = 384,
    use_sentence_transformers: bool = True,
) -> list[list[float]]:
    """Embed policy chunks using sentence-transformers when available, else hash fallback."""
    if not texts:
        return []
    if use_sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [[float(x) for x in row] for row in vecs]
        except Exception as exc:
            logger.warning(
                "policy_embedding_fallback",
                extra={"model_name": model_name, "reason": str(exc)},
            )
    vecs = hash_embed_texts(texts, dim=dim)
    return [[float(x) for x in row] for row in vecs]


def load_policy_chunks(
    policy_version: str = "v1",
    policies_dir: str | Path | None = None,
    max_chars: int = 600,
) -> list[PolicyChunk]:
    """Read and chunk all markdown policy files for the given version."""
    base_dir = (
        Path(policies_dir)
        if policies_dir is not None
        else _find_project_root() / "configs" / "policies" / policy_version
    )
    if not base_dir.exists():
        raise FileNotFoundError(f"Policy directory not found: {base_dir}")

    chunks: list[PolicyChunk] = []
    for md_path in sorted(base_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_markdown(
                stem=md_path.stem,
                version=policy_version,
                text=text,
                max_chars=max_chars,
            )
        )
    return chunks


def load_and_store_policies(
    session: Session,
    *,
    run_id: str,
    policy_version: str = "v1",
    model_name: str = "all-MiniLM-L6-v2",
    embedding_dim: int = 384,
    policies_dir: str | Path | None = None,
    use_sentence_transformers: bool = True,
) -> int:
    """Load, chunk, embed, and persist all policy chunks into policy_documents."""
    chunks = load_policy_chunks(policy_version=policy_version, policies_dir=policies_dir)
    embeddings = embed_policy_texts(
        [f"{c.title}\n{c.body}" for c in chunks],
        model_name=model_name,
        dim=embedding_dim,
        use_sentence_transformers=use_sentence_transformers,
    )

    for chunk, vec in zip(chunks, embeddings, strict=True):
        existing = session.get(PolicyDocument, chunk.policy_id)
        if existing is None:
            session.add(
                PolicyDocument(
                    policy_id=chunk.policy_id,
                    version=chunk.version,
                    title=chunk.title,
                    body=chunk.body,
                    chunk_index=chunk.chunk_index,
                    embedding=vec,
                    run_id=run_id,
                )
            )
        else:
            existing.title = chunk.title
            existing.body = chunk.body
            existing.chunk_index = chunk.chunk_index
            existing.embedding = vec
            existing.run_id = run_id

    session.flush()
    logger.info(
        "policies_loaded",
        extra={"policy_version": policy_version, "chunk_count": len(chunks)},
    )
    return len(chunks)
