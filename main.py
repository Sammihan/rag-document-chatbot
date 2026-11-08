from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# Initialize embedding model
embedding_model = MistralAIEmbeddings(model="mistral-embed")


# Initialize Gemini model
model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")


# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an intelligent PDF question-answering assistant.

        Instructions:
        1. Answer the user's question using the provided context.
        2. Give clear, complete and natural-language answers.
        3. Do not return Python dictionaries, JSON, metadata or signatures.
        4. Do not make up information.
        5. If the answer genuinely isn't available in the context, say:
           "I could not find the answer in the document."
        """,
        ),
        (
            "human",
            """Context:
        {context}

        Question:
        {question}

        Answer:""",
        ),
    ]
)


def get_answer(persist_directory, query):

    # Load existing ChromaDB
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name="pdf_collection",
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    print("\n--- RETRIEVED DOCUMENTS ---")

    for doc in docs:
        print("Page:", doc.metadata.get("page", 0) + 1)
        print(doc.page_content[:500])
        print("--------------------------")

    # Prepare context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate final prompt
    final_prompt = prompt.invoke({"context": context, "question": query})

    # Generate answer using Gemini
    response = model.invoke(final_prompt)

    # Extract only the actual text
    if isinstance(response.content, str):
        answer = response.content

    else:
        answer = "\n".join(
            block["text"]
            for block in response.content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and "text" in block
        )

    return answer, docs
