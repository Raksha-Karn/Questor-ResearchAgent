import re

def rerank_documents(
    llm,
    query: str,
    docs,
    top_k: int = 5,
):
    scored_docs = []
    for doc in docs:
        response = llm.invoke(
            f"""
            Rate relevance from 0-100.
            Only return a number.
            Query:
            {query}
            Document:
            {doc.page_content[:3000]}
            """
        )

        score_text = response.content.strip()
        match = re.search(
            r"\d+",
            score_text,
        )

        score = int(match.group()) if match else 0
        scored_docs.append(
            (
                score,
                doc,
            )
        )

    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        doc
        for _, doc in scored_docs[:top_k]
    ]