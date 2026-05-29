from __future__ import annotations
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHROMA_DIR = Path("storage/chroma_db")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "models/gemini-embedding-001"

VECTORSTORES: Dict[str, Chroma] = {}


class PDFLoadError(Exception):
    pass

def get_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

def register_vectorstore(
    session_id: str,
    vectorstore: Chroma,
) -> None:
    VECTORSTORES[session_id] = vectorstore

def get_vectorstore(
    session_id: str,
) -> Optional[Chroma]:
    return VECTORSTORES.get(session_id)

def build_vectorstore(
    uploaded_file,
    api_key: str,
    file_hash: str,
    source_name: str,
) -> Chroma:

    persist_dir = CHROMA_DIR / file_hash
    embeddings = get_embeddings(api_key)

    if persist_dir.exists():
        try:
            logger.info("Loading existing Chroma DB")
            return Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
            )
        except Exception:
            logger.warning("Corrupt Chroma DB. Rebuilding.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as tmp:

        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    try:
        pages = PyPDFLoader(pdf_path).load()

        for page in pages:
            page.metadata["file_hash"] = file_hash
            page.metadata["source"] = source_name

    except Exception as e:
        logger.exception("PDF loading failed")
        raise PDFLoadError(str(e))

    finally:
        os.unlink(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    return vectorstore

@tool
def pdf_lookup(
    query: str,
    session_id: str,
) -> str:
    """
    Search uploaded PDFs.
    Returns relevant excerpts with page citations.
    """

    vectorstore = get_vectorstore(session_id)

    if vectorstore is None:
        return "No PDF indexed for this session."

    docs = vectorstore.similarity_search_with_score(
        query,
        k=5,
    )

    if not docs:
        return "No relevant content found."

    results = []

    for doc, score in docs:

        if score > 1.5:
            continue

        page = doc.metadata.get("page", 0) + 1
        source = doc.metadata.get("source", "Unknown")

        results.append(
            f"""
            Source: {source}
            Page: {page}
            Similarity Score: {score:.3f}

            {doc.page_content}
            """
        )

    return "\n\n---\n\n".join(results)