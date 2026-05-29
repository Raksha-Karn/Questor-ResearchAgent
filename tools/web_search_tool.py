from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

web_search_tool = TavilySearch(
    tavily_api_key=TAVILY_API_KEY,
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