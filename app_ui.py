import streamlit as st
import os
import tempfile


st.set_page_config(
    page_title="AskMyDocs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

hide_streamlit_style = """
<style>
#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# from app.evaluation import evaluate_rag

from app.ingest import SUPPORTED_EXTENSIONS
from app.rag_pipeline import RAGPipeline

st.title("📄 AskMyDocs")
st.caption("AI that answers from your documents, not assumptions.")


# Initialize pipeline
if "documents_folder" not in st.session_state:
    st.session_state.documents_folder = tempfile.mkdtemp(prefix="askmydocs_")

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(documents_folder=st.session_state.documents_folder)


rag = st.session_state.rag


# ---------------------------
# Upload Documents
# ---------------------------

st.header("Upload Documents")
st.info(
    "Upload one or more documents and ask questions in natural language. "
    "AskMyDocs provides accurate, context-grounded answers using only the content of your uploaded documents."
)

uploaded_files = st.file_uploader(
    "Upload PDF, TXT, Word, or PowerPoint files",
    type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True
)


if uploaded_files:

    for file in uploaded_files:

        os.makedirs(st.session_state.documents_folder, exist_ok=True)

        save_path = os.path.join(
            st.session_state.documents_folder,
            os.path.basename(file.name)
        )

        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

    st.success("Documents uploaded successfully!")

    rag.build_index()


# ---------------------------
# Ask Questions
# ---------------------------

st.header("Ask Question")

question = st.text_input("Enter your question", key="question_input")

if st.button("Clear Question Cache"):
    for key in ["question_input", "last_answer", "last_sources"]:
        st.session_state.pop(key, None)

    # Clear any Streamlit caches if present.
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Question cache cleared.")
    st.rerun()

if st.button("Ask"):

    if question:

        with st.spinner("Generating answer..."):
            answer, sources = rag.ask(question)

        st.session_state.last_answer = answer
        st.session_state.last_sources = sources

        st.subheader("Answer")
        # st.write(answer)
        st.success(answer)
        st.subheader("Sources")
        for source in sources:
            st.write(f"- {source}")
    else:
        st.warning("Please enter a question before asking!!")