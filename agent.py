from dotenv import load_dotenv
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.calculator_tool import calculator
from tools.web_search_tool import web_search_tool
from tools.pdf_tools import create_pdf_tools, initialize_summary_service, initialize_reranker

load_dotenv()

SYSTEM_PROMPT = """
You are an advanced research assistant.

Capabilities:

- Search uploaded PDFs
- Summarize uploaded PDFs
- View document metadata
- Search the web
- Perform calculations

Rules:

- Use PDF tools for uploaded-document questions.
- Use web search for external information.
- Use calculator for math.
- Use multiple tools when needed.
- Cite sources whenever possible.
"""

def build_agent(session_id: str) -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )
    initialize_summary_service(llm)
    initialize_reranker(llm)
    pdf_tools = create_pdf_tools(session_id)
    tools = (pdf_tools + 
        [
            web_search_tool,
            calculator,
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(
                "chat_history"
            ),
            ("human", "{input}"),
            MessagesPlaceholder(
                "agent_scratchpad"
            ),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )

def run_agent(
    executor: AgentExecutor,
    question: str,
    history: list,
) -> dict:
    chat_history = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")

        if not content:
            continue

        if role == "user":
            chat_history.append(
                HumanMessage(
                    content=content
                )
            )

        elif role == "assistant":
            chat_history.append(
                AIMessage(
                    content=content
                )
            )
    try:
        result = executor.invoke(
            {
                "input": question,
                "chat_history": chat_history,
            }
        )

        steps = []
        for (action, observation) in result.get("intermediate_steps", []):
            steps.append(
                {
                    "tool": getattr(
                        action,
                        "tool",
                        "unknown",
                    ),
                    "input": str(
                        getattr(
                            action,
                            "tool_input",
                            "",
                        )
                    ),
                    "output": str(
                        observation
                    )[:500],
                }
            )

        output = result.get("output", "")
        if isinstance(output, str):
            answer = output

        elif isinstance(output, list):
            text_parts = []
            for item in output:
                if isinstance(item, dict):
                    if "text" in item:
                        text_parts.append(item["text"])
                    else:
                        text_parts.append(str(item))
                else:
                    text_parts.append(str(item))
            answer = "\n".join(text_parts)
        else:
            answer = str(output)

        return {
            "answer": answer,
            "steps": steps,
        }

    except Exception as e:
        return {
            "answer": (
                "An error occurred while "
                "processing your request."
            ),
            "steps": [
                {
                    "tool": "error",
                    "input": question,
                    "output": str(e),
                }
            ],
        }