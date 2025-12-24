import streamlit as st
from pathlib import Path

from src.pipelines.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="EV RAG Assistant",
    layout="wide",
)

st.title("⚡ EV Charging RAG Assistant")
st.write("Ask questions grounded in EV charging industry reports.")

# Sidebar controls
st.sidebar.header("Configuration")

pdf_path = st.sidebar.text_input("PDF path", value="data/electric_highways.pdf")

top_k = st.sidebar.slider(
    "Number of retrieved context chunks (k)", min_value=1, max_value=10, value=3
)

temperature = st.sidebar.slider(
    "LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05
)


# Cache the RAG pipeline so it doesn't rebuild every rerun
@st.cache_resource
def load_rag(pdf_path: str):
    rag = RAGPipeline(pdf_path=Path(pdf_path))
    rag.build_index()
    return rag


rag = None
if pdf_path:
    try:
        rag = load_rag(pdf_path)
    except Exception as e:
        st.error(f"Failed to load RAG pipeline: {e}")

# User query
query = st.text_input("Enter your question:")

if st.button("Run Query") and rag and query:
    with st.spinner("Retrieving context and generating answer..."):
        answer, sources = rag.query(
            question=query,
            k=top_k,
        )

    st.subheader("✅ Answer")
    st.write(answer)

    st.markdown("### 📚 Supporting Contexts")

    for i, source in enumerate(sources, start=1):
        st.markdown(f"**Context {i}:**")
        st.text(source["content"])
        st.markdown(f"_Source: {source['source']} (Page {source['page']})_")
        st.write("---")
