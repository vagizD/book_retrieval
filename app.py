"""Streamlit application for semantic book discovery with Chroma."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import streamlit as st

from src.chroma_retrieval import search_chroma
from src.config import IndexManifest
from src.deployment import materialize_chroma_dir
from src.embeddings import load_manifest, validate_chroma_collection

LOGGER = logging.getLogger(__name__)
DEFAULT_INDEX_DIR = "data"
EXCERPT_CHARACTERS = 650


@st.cache_data(show_spinner=False)
def cached_manifest(index_dir: str) -> IndexManifest:
    """Load the small manifest once per artifact path."""
    return load_manifest(index_dir)


@st.cache_resource(show_spinner="Loading the semantic model…")
def cached_model(model_id: str) -> Any:
    """Load the CPU embedding model once per app process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device="cpu")


@st.cache_resource(show_spinner=False)
def cached_collection(index_dir: str, manifest: IndexManifest) -> Any:
    """Open and validate the read-mostly persistent Chroma collection."""
    import chromadb

    chroma_path = materialize_chroma_dir(index_dir, manifest.corpus_sha256)
    if not chroma_path.is_dir():
        raise ValueError("Chroma persistence directory is missing")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(
        name=manifest.collection_name, embedding_function=None
    )
    validate_chroma_collection(collection, manifest)
    return collection


def excerpt(text: str, limit: int = EXCERPT_CHARACTERS) -> str:
    """Shorten a description without splitting its final displayed word."""
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0].rstrip()
    return f"{shortened}…"


def render_result(row: Any) -> None:
    """Render one ranked recommendation."""
    with st.container(border=True):
        st.subheader(f"{int(row['rank'])}. {row['title']}")
        st.caption(f"Cosine similarity: {float(row['similarity_score']):.3f}")
        st.write(excerpt(str(row["description"])))


def render_how_it_works() -> None:
    """Explain the retrieval behavior and score semantics."""
    with st.expander("How it works"):
        st.write(
            "Your prompt and every book description are represented as vectors "
            "by the same MiniLM model. Chroma retrieves descriptions whose "
            "vectors are closest by cosine distance. This captures meaning, not "
            "just exact keyword overlap."
        )
        st.write(
            "Similarity scores are cosine similarities—not probabilities, "
            "ratings, or calibrated confidence. Results reflect description "
            "similarity rather than personal reading preferences."
        )


def main() -> None:
    """Render the application and handle visitor input safely."""
    st.set_page_config(page_title="Between the Lines", page_icon="📚")
    st.title("Between the Lines")
    st.subheader("Find books by meaning, not keywords.")

    index_dir = os.environ.get("BOOK_INDEX_DIR", DEFAULT_INDEX_DIR)
    try:
        manifest = cached_manifest(index_dir)
        collection = cached_collection(index_dir, manifest)
    except Exception:
        LOGGER.exception("Unable to load the book index")
        st.error(
            "The recommendation index is unavailable or malformed. "
            "Please try again later."
        )
        st.info(
            "For local setup, build the index with "
            "`python scripts/build_index.py --max-books 100 --source-limit 5000`."
        )
        render_how_it_works()
        return

    st.caption(
        f"Searching {manifest.row_count:,} book descriptions · "
        f"Model: {manifest.model_id} · Chroma cosine retrieval"
    )
    with st.form("book_search"):
        query = st.text_area(
            "Describe what you would like to read",
            placeholder=(
                "A mysterious atmospheric novel about a small town hiding "
                "dark secrets"
            ),
            height=120,
        )
        result_count = st.radio(
            "Number of recommendations", options=[5, 10], horizontal=True
        )
        submitted = st.form_submit_button("Find books", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Please describe the kind of book you want to read.")
        else:
            try:
                with st.spinner("Finding books with similar descriptions…"):
                    model = cached_model(manifest.model_id)
                    results = search_chroma(
                        query, model, collection, k=int(result_count)
                    )
                if results.empty:
                    st.info("No recommendations are available in this corpus.")
                else:
                    st.markdown("### Recommendations")
                    for _, row in results.iterrows():
                        render_result(row)
            except Exception:
                LOGGER.exception("Book search failed")
                st.error(
                    "The search could not be completed. Please try again shortly."
                )

    render_how_it_works()


if __name__ == "__main__":
    main()
