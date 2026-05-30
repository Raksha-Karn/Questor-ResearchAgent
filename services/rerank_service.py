import re

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

        score_text = get_text_content(response).strip()
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