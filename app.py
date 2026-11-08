import os
import uuid
import shutil
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Load Streamlit Cloud secrets when deployed
try:
    for key in ["GOOGLE_API_KEY", "MISTRAL_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from create_database import create_database
from main import get_answer


# Page configuration
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")

st.title("📚 Chat with your PDF")
st.caption("Upload a PDF and ask questions about its content.")


# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "vectorstore_path" not in st.session_state:
    st.session_state.vectorstore_path = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
with st.sidebar:

    st.header("Upload your document")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("Process PDF", type="primary", disabled=uploaded_file is None):

        with st.spinner("Processing PDF and creating embeddings..."):

            try:
                # Create vector database
                db_path = create_database(
                    pdf_bytes=uploaded_file.getvalue(),
                    session_id=st.session_state.session_id,
                )

                st.session_state.vectorstore_path = db_path
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []

                st.success("PDF processed successfully!")

            except Exception as e:
                st.error(f"PDF processing failed: {e}")

    # Show active document
    if st.session_state.pdf_name:

        st.divider()

        st.success(f"Active document: {st.session_state.pdf_name}")

        if st.button("Clear document"):

            db_path = st.session_state.vectorstore_path

            if db_path and os.path.exists(db_path):
                shutil.rmtree(db_path)

            st.session_state.vectorstore_path = None
            st.session_state.pdf_name = None
            st.session_state.chat_history = []

            st.rerun()


# Main chat interface
if st.session_state.vectorstore_path is None:

    st.info("Upload and process a PDF from the sidebar to start chatting.")

else:

    # Display chat history
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View sources"):
                    for source in message["sources"]:
                        st.write(source)

    # User input
    query = st.chat_input("Ask a question about your PDF...")

    if query:

        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Searching document and generating answer..."):

                try:
                    answer, docs = get_answer(st.session_state.vectorstore_path, query)

                    st.markdown(answer)

                    # Collect source pages
                    sources = []

                    for doc in docs:
                        page = doc.metadata.get("page", 0) + 1

                        source = f"Page {page}"

                        if source not in sources:
                            sources.append(source)

                    # Display sources
                    if sources:
                        with st.expander("View sources"):
                            for source in sources:
                                st.write(source)

                    # Save assistant response
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
