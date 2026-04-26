import os
import streamlit as st
from query import load_vectorstore, build_qa_chain, ask, format_sources

VECTORSTORE_PATH = "vectorstore"

st.set_page_config(
    page_title="DocQA Bot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Q&A Bot")
st.caption("Ask questions grounded in your documents. Powered by Groq LLaMA3 + FAISS.")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This bot uses a **RAG pipeline**:
    - Documents are chunked and embedded locally
    - Your query is matched to relevant chunks via FAISS
    - Groq LLaMA3 generates a grounded answer
    
    It will **not** answer from general knowledge —
    only from the indexed documents.
    """)
    st.divider()
    st.markdown("**Model:** `llama3-8b-8192` via Groq")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2` (local)")
    st.markdown("**Vector DB:** FAISS (persisted to disk)")


@st.cache_resource(show_spinner="Loading vector store and model...")
def get_chain():
    if not os.path.exists(VECTORSTORE_PATH):
        return None
    vectorstore = load_vectorstore()
    return build_qa_chain(vectorstore)


qa_chain = get_chain()

if qa_chain is None:
    st.error("Vector store not found. Run `python ingest.py` first to index your documents.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                answer, sources = ask(qa_chain, prompt)
                src_text = format_sources(sources)
                full_response = f"{answer}\n\n---\n**📎 Sources:**\n{src_text}"
            except Exception as e:
                full_response = f"Error: {e}"

        st.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })