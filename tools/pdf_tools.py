from __future__ import annotations
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.retrieval_service import build_hybrid_retriever
from services.summary_service import SummaryService
from services.rerank_service import rerank_documents

logger = logging.getLogger(__name__)

CHROMA_DIR = Path("storage/chroma_db")
CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EMBEDDING_MODEL = (
    "models/gemini-embedding-001"
)

VECTORSTORES: Dict[str, Chroma] = {}

DOCUMENT_METADATA: Dict[
    str,
    Dict,
] = {}

summary_service: Optional[SummaryService] = None
rerank_llm = None


class PDFLoadError(Exception):
    pass

def initialize_summary_service(
    llm,
):
    global summary_service
    summary_service = SummaryService(llm)

def get_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

def register_vectorstore(
    session_id: str,
    vectorstore: Chroma,
):
    VECTORSTORES[
        session_id
    ] = vectorstore

def get_vectorstore(session_id: str) -> Optional[Chroma]:
    return VECTORSTORES.get(
        session_id
    )

def build_vectorstore(
    uploaded_file,
    api_key: str,
    file_hash: str,
    source_name: str,
    session_id: str,
) -> Chroma:
    persist_dir = (
        CHROMA_DIR / file_hash
    )

    embeddings = get_embeddings(api_key)
    if persist_dir.exists():
        try:
            vectorstore = Chroma(
                persist_directory=str(
                    persist_dir
                ),
                embedding_function=embeddings,
            )
            register_vectorstore(session_id, vectorstore)

            return vectorstore

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

        DOCUMENT_METADATA.setdefault(session_id, {})
        DOCUMENT_METADATA[session_id][source_name] = {
            "pages": len(pages),
            "file_hash": file_hash,
        }

        for page in pages:
            page.metadata["source"] = source_name
            page.metadata["file_hash"] = file_hash

    except Exception as e:
        logger.exception("Failed loading PDF")
        raise PDFLoadError(str(e))

    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                ".",
                " ",
            ],
        )
    )

    chunks = splitter.split_documents(pages)

    vectorstore = (
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(
                persist_dir
            ),
        )
    )
    register_vectorstore(session_id, vectorstore)

    return vectorstore


def create_pdf_tools(session_id: str):
    @tool
    def pdf_lookup(query: str) -> str:
        """
        Search uploaded PDFs and return
        relevant passages with citations.
        """
        vectorstore = get_vectorstore(session_id)
        if vectorstore is None:
            return "No PDF indexed for this session."

        stored_docs = vectorstore.get()
        documents = [
            Document(
                page_content=text,
                metadata=meta,
            )
            for text, meta in zip(
                stored_docs["documents"],
                stored_docs["metadatas"],
            )
        ]

        retriever = build_hybrid_retriever(documents, vectorstore)
        docs = retriever.invoke(query)
        docs = rerank_documents(
            rerank_llm,
            query,
            docs,
            top_k=5,
        )
        if not docs:
            return "No relevant content found."

        results = []

        for doc in docs[:5]:
            page = doc.metadata.get("page", 0) + 1
            source = (
                doc.metadata.get("source", "Unknown")
            )

            results.append(
                f"""
                Source: {source}
                Page: {page}

                {doc.page_content}
                """
            )

        return "\n\n---\n\n".join(results)
        

    @tool
    def document_metadata() -> str:
        """
        Return metadata about
        uploaded documents.
        """
        session_docs = (
            DOCUMENT_METADATA.get(session_id)
        )

        if not session_docs:
            return "No documents indexed."

        output = []
        for (doc_name, metadata) in session_docs.items():
            output.append(
                f"""
                Document: {doc_name}
                Pages: {metadata['pages']}
                """
            )

        return "\n".join(output)

    @tool
    def summarize_document(summary_type: str = "executive") -> str:
        """
        Summarize uploaded documents.
        Available types:
        - executive
        - detailed
        - bullet
        - research
        """

        if summary_service is None:
            return "Summary service not initialized."
            
        vectorstore = get_vectorstore(session_id)
        if vectorstore is None:
            return "No PDF indexed for this session."

        docs = (vectorstore.similarity_search("main topics key findings conclusions", k=25))
        return summary_service.summarize(
            docs,
            summary_type,
        )
    return [
        pdf_lookup,
        summarize_document,
        document_metadata,
    ]

def initialize_reranker(llm):
    global rerank_llm
    rerank_llm = llm