import streamlit as st
import os
import tempfile

# from app.evaluation import evaluate_rag

from app.ingest import SUPPORTED_EXTENSIONS
from app.rag_pipeline import RAGPipeline

st.title("📄 AskMyDocs")
st.caption("AI that answers using your documents, not assumptions.")

st.markdown("---")

st.markdown("## ✨ Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
- 📄 Multi-document support
- 🤖 AI-powered Question Answering
- 🔍 Semantic Search
""")

with col2:
    st.markdown("""
- 📚 Retrieval-Augmented Generation (RAG)
- 📌 Source-grounded responses
- ⚡ Powered by Groq LLM
""")

st.markdown("---")

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
    """
📂 **Upload one or more documents**

Supported file formats:

• PDF  • DOCX  • PPTX  • TXT

Ask questions naturally, and AskMyDocs will answer **strictly using the uploaded documents**, providing context-grounded responses.
"""
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

st.header("💬 Ask Questions")

st.caption(
    "Ask anything about your uploaded documents. Responses are generated only from the uploaded content."
)

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


st.markdown("---")

st.markdown(
    """
<div style='text-align:center;color:gray;font-size:14px;'>

Built with ❤️ using **Streamlit • LangChain • FAISS • Hugging Face • Groq**

</div>
""",
unsafe_allow_html=True
)