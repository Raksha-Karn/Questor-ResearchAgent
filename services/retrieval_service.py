from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever

def get_text_content(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)

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

def expand_query(
    llm,
    query: str,
) -> str:

    prompt = f"""
            Expand the following query into related keywords, concepts, and alternate phrasings.

            Query:
            {query}
        """
    response = llm.invoke(prompt)

    return get_text_content(response).strip()