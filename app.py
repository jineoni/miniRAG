import streamlit as st

from src.config import TOP_K
from src.generation import answer_question
from src.indexing import load_embedding_model, open_collection

st.set_page_config(page_title="Mini RAG", page_icon="📚")
st.title("Mini RAG: Research Paper QA Assistant")
st.caption("Reinforcement learning for traffic signal control · Single-turn QA")


@st.cache_resource
def load_resources():
    collection = open_collection()
    return load_embedding_model(), collection


try:
    model, collection = load_resources()
except Exception as error:
    st.info("Build the knowledge base first: python build_index.py")
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Knowledge Base")
    records = collection.get(include=["metadatas"])
    papers = {item["paper_id"]: item for item in records["metadatas"]}
    st.metric("Indexed papers", len(papers))
    st.metric("Indexed chunks", collection.count())
    top_k = st.slider("Top-K", 1, 10, TOP_K)
    with st.expander("Paper library"):
        for paper in papers.values():
            st.link_button(paper["title"], paper["url"])
    st.caption("After rebuilding the index, restart this app.")


def show_result(result):
    st.markdown(result["answer"])
    for source in result["sources"]:
        with st.expander(f"[{source['source_id']}] {source['title']}"):
            st.caption(
                f"Page {source['page']} · Cosine similarity: {source['similarity']:.3f}"
            )
            st.text(source["text"])
            st.link_button("Open paper", source["url"])


if "history" not in st.session_state:
    st.session_state.history = []

for question, result in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        show_result(result)

question = st.chat_input("Ask a question about the indexed papers")
if question and question.strip():
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching papers and generating an answer..."):
                result = answer_question(question, model, collection, top_k)
        except Exception as error:
            st.error(str(error))
        else:
            show_result(result)
            st.session_state.history.append((question, result))
