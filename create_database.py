import os
import shutil
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def create_database(pdf_bytes, session_id):

    # Create a separate database directory for each session
    persist_directory = os.path.join("chroma_db", session_id)

    # Replace the previous database for this session
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    # Save uploaded PDF temporarily
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No pages found in the PDF.")

        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No readable text found in the PDF.")

        # Mistral embeddings
        embedding_model = MistralAIEmbeddings(model="mistral-embed")

        # Create ChromaDB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="pdf_collection",
            persist_directory=persist_directory,
        )

        return persist_directory

    except Exception:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        raise

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
