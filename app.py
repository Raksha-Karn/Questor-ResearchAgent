import hashlib
import json
import uuid
import os
import streamlit as st
from agent import build_agent, run_agent
from tools.pdf_tools import build_vectorstore, create_pdf_tools
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔍",
    layout="wide",
)
st.markdown(
"""
<style>

.main > div {
    padding-top: 1rem;
}

.block-container {
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background-color: #fafafa;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
}

.stChatMessage {
    border-radius: 12px;
}

</style>
""",
    unsafe_allow_html=True,
)

ICON_MAP = {
    "pdf_lookup": "📄",
    "summarize_document": "📑",
    "document_metadata": "📚",
    "calculator": "🔢",
    "web_search": "🌐",
    "error": "❌",
}

def generate_file_hash(uploaded_file):
    uploaded_file.seek(0)
    digest = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return digest

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = build_agent(
        st.session_state.session_id
    )

with st.sidebar:
    st.title("🔍 Research Agent")
    st.caption("Gemini • PDF Retrieval • Web Search • Summarization")
    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.pop("pdf_name", None)
        st.rerun()

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        st.info(f"{uploaded_file.name}")
        if st.button("Process PDF", type="primary"):
            try:
                with st.spinner("Embedding PDF.."):
                    build_vectorstore(
                        uploaded_file=uploaded_file,
                        api_key=GEMINI_API_KEY,
                        file_hash=generate_file_hash(uploaded_file),
                        source_name=uploaded_file.name,
                        session_id=st.session_state.session_id,
                    )

                st.success("PDF indexed successfully!")
                st.session_state["pdf_name"] = uploaded_file.name
                if "pdf_name" in st.session_state:
                    st.success(f"Active PDF: {st.session_state['pdf_name']}")

            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Quick Actions")
    summary_btn = st.button("Executive Summary")
    research_btn = st.button("Research Summary")
    metadata_btn = st.button("Document Metadata")
    st.divider()
    show_trace = st.toggle("Show Tool Trace", value=True)
    st.divider()
    export_data = json.dumps(st.session_state.messages, indent=2)

    st.download_button(
        label="Export Chat",
        data=export_data,
        file_name="research_chat.json",
        mime="application/json",
    )

st.title("🔍 Research Agent")
st.caption("Ask questions about uploaded documents, search the web, summarize information, and perform calculations.")

if summary_btn:
    try:
        pdf_tools = create_pdf_tools(st.session_state.session_id)
        summary_tool = pdf_tools[1]
        result = summary_tool.invoke({"summary_type": "executive"})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result,
                "steps": [],
            }
        )
        st.rerun()

    except Exception as e:
        st.error(str(e))

if research_btn:
    try:
        pdf_tools = create_pdf_tools(st.session_state.session_id)
        summary_tool = pdf_tools[1]
        result = summary_tool.invoke({"summary_type": "research"})

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result,
                "steps": [],
            }
        )
        st.rerun()

    except Exception as e:
        st.error(str(e))

if metadata_btn:
    try:
        pdf_tools = create_pdf_tools(st.session_state.session_id)
        metadata_tool = pdf_tools[2]
        result = metadata_tool.invoke({})

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result,
                "steps": [],
            }
        )
        st.rerun()

    except Exception as e:
        st.error(str(e))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            show_trace
            and message["role"] == "assistant"
            and message.get("steps")
        ):
            with st.expander(f"Tool Trace ({len(message['steps'])})"):
                for step in message["steps"]:
                    icon = ICON_MAP.get(step["tool"], "⚙️")
                    st.markdown(f"### {icon} {step['tool']}")
                    st.code(str(step["input"]))
                    st.caption(step["output"])

question = st.chat_input("Ask anything..")
if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and web.."):
            result = run_agent(
                st.session_state.agent,
                question,
                st.session_state.messages[:-1],
            )

        st.markdown(result["answer"])

        if (show_trace and result["steps"]):
            with st.expander(f"Tool Trace ({len(result['steps'])})"):
                for step in result["steps"]:
                    icon = ICON_MAP.get(step["tool"], "⚙️")
                    st.markdown(f"### {icon} {step['tool']}")
                    st.code(str(step["input"]))
                    st.caption(step["output"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "steps": result["steps"],
        }
    )