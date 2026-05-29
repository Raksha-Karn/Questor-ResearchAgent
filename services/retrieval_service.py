from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever

def build_hybrid_retriever(
    docs: list[Document],
    vectorstore,
):
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10

    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 30,
            "lambda_mult": 0.5,
        },
    )

    return EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            dense_retriever,
        ],
        weights=[0.4, 0.6],
    )