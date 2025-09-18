import os
import streamlit as st
from dotenv import load_dotenv
from typing import Iterator

from agno.agent import Agent, RunResponseEvent
from agno.memory.v2.memory import Memory

from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder

from agno.knowledge.url import UrlKnowledge
from agno.vectordb.search import SearchType
from agno.vectordb.weaviate import Distance, VectorIndex, Weaviate

import weaviate
from weaviate.classes.init import Auth

from agno.utils.pprint import pprint_run_response

load_dotenv()

WEAVIATE_URL=os.getenv("WEAVIATE_URL")
WEAVITE_API_KEY=os.getenv("WEAVIATE_API_KEY")

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

st.set_page_config(page_title="URL RAG (Gemini + Weaviate)", layout="wide")

def load_knowledge_base(urls: list[str] = None):
    embedder = GeminiEmbedder(id="text-embedding-004")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVITE_API_KEY),
    )

    collection_name="agentic-rag"

    vector_db = Weaviate(
        client=client,
        embedder=embedder,
        search_type=SearchType.hybrid,
        vector_index=VectorIndex.HNSW,
        distance=Distance.COSINE,
        local=False,
    )

    knowledge = UrlKnowledge(
        urls=urls or [],
        embedder=embedder,
        vector_db=vector_db,
        collection=collection_name,
    )

    knowledge.load()

    return knowledge,vector_db


def agentic_rag_response(query: str) -> Iterator[RunResponseEvent]:
    knowledge_base = st.session_state.get("knowledge_base", None)

    if knowledge_base is None:
        st.error("Knowledge base not loaded.Load it first.........")
        return

    agent = Agent(
        model=Gemini(id="gemini-2.5-flash"),
        knowledge=knowledge_base,
        memory=st.session_state.memory,
        add_history_to_messages=True,
        enable_session_summaries=True,
        enable_user_memories=True,
        search_knowledge=True,
        markdown=True,
    )

    return agent.run(query, stream=True)


col1, col2 = st.columns([4, 1])
with col1:
    st.markdown(
        "<h1>Agentic URL RAG (Gemini + Weaviate)</h1>",
        unsafe_allow_html=True,
    )

with col2:
    if st.button("Reset KB"):
        st.session_state.docs_loaded = False
        if "loaded_urls" in st.session_state:
            del st.session_state["loaded_urls"]
        if "memory" in st.session_state:
            del st.session_state["memory"]
        st.success("Memory reset!")
        st.rerun()


# Sidebar
with st.sidebar:
    st.markdown("### Knowledge Base URLs")
    if "urls" not in st.session_state:
        st.session_state.urls = [""]

    col1, col2 = st.columns([4, 1])
    with col1:
        for i, url in enumerate(st.session_state.urls):
            st.session_state.urls[i] = col1.text_input(
                f"URL {i + 1}", value=url, key=f"url_{i}", label_visibility="collapsed"
            )
    if col2.button("➕"):
        if st.session_state.urls and st.session_state.urls[-1].strip() != "":
            st.session_state.urls.append("")

    # Deduplicate & clean
    urls = list(dict.fromkeys([u for u in st.session_state.urls if u.strip()]))

    if st.button("Load Knowledge Base"):
        if urls:
            with st.spinner("Loading knowledge base..."):
                try:
                    knowledge_base, vector_db = load_knowledge_base(urls)
                    st.session_state.docs_loaded = True
                    st.session_state.loaded_urls = urls.copy()
                    st.session_state.knowledge_base = knowledge_base
                    st.session_state.vector_db = vector_db
                    st.success(f"Knowledge base loaded with {len(urls)} URL(s)!")
                except Exception as e:
                    st.error(f"Error loading KB: {e}")
        else:
            st.warning("Please add at least one URL.")

    if st.session_state.get("docs_loaded", False) and st.session_state.get(
        "loaded_urls"
    ):
        st.markdown("**Loaded URLs:**")
        for i, url in enumerate(st.session_state.loaded_urls, 1):
            st.markdown(f"{i}. {url}")


# Chat Input
query = st.chat_input("Ask a question...")
if query:
    if not st.session_state.get("docs_loaded", False):
        st.warning("Load the knowledge base first.")
    else:
        st.session_state.memory.add_message("user", query)

        response = agentic_rag_response(query)

        if response is not None:
            st.markdown("### Answer")
            answer = ""
            answer_placeholder = st.empty()

            for content in response:
                if hasattr(content, "event") and content.event == "RunResponseContent":
                    answer += content.content
                    answer_placeholder.markdown(answer, unsafe_allow_html=True)

            if answer.strip():
                st.session_state.memory.add_message("assistant", answer)