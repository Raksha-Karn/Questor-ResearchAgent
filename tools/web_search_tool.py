from langchain_tavily import TavilySearch

web_search_tool = TavilySearch(
    max_results=10,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    description=(
        "Search the internet for current events, "
        "facts, research papers, news, prices, "
        "and information not contained in PDFs."
    ),
)