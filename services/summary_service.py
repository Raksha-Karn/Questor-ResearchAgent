from typing import List
from langchain_core.documents import Document

SUMMARY_PROMPTS = {
    "executive": """
    Provide an executive summary.

    Include:
    - Purpose
    - Key findings
    - Conclusions

    Limit to 500 words.
    """,

        "bullet": """
    Summarize as bullet points.

    Include:
    - Main topics
    - Key facts
    - Important numbers
    """,

        "research": """
    Provide a research summary.

    Include:
    - Objective
    - Methodology
    - Findings
    - Limitations
    - Conclusions
    """,

        "detailed": """
    Provide a detailed summary.
    """
}


class SummaryService:
    def __init__(self, llm):
        self.llm = llm

    def summarize(
        self,
        docs: List[Document],
        summary_type: str = "executive",
    ) -> str:
        if not docs:
            return "No documents available."

        prompt_template = SUMMARY_PROMPTS.get(
            summary_type,
            SUMMARY_PROMPTS["executive"],
        )

        context_parts = []

        for doc in docs:
            page = (
                doc.metadata.get("page", 0)
                + 1
            )
            source = doc.metadata.get(
                "source",
                "Unknown",
            )

            context_parts.append(
                f"""
                SOURCE: {source}
                PAGE: {page}

                {doc.page_content}
                """
            )
        context = "\n\n".join(
            context_parts
        )

        prompt = f"""
        {prompt_template}
        Use citations whenever possible:
        (Source.pdf p.12)
        Context:
        {context}
        """
        response = self.llm.invoke(prompt)
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if (isinstance(item, dict) and item.get("type") == "text"):
                    text_parts.append(item.get("text", ""))

            return "\n".join(text_parts)
        return str(content)