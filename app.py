import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Load environment variables from .env
load_dotenv()

# Create OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# Set up Streamlit page
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("Enterprise RAG Assistant")

st.write(
    "Upload company documents and ask questions based on their content."
)


# Load embedding model once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "multi-qa-MiniLM-L6-cos-v1"
    )


model = load_embedding_model()


# PDF uploader
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    st.success(f"Loaded: {uploaded_file.name}")

    # Read PDF
    reader = PdfReader(uploaded_file)

    document_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            document_text += text + "\n"


    st.write(
        f"Successfully extracted {len(document_text)} characters."
    )


    # Split document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(
        document_text
    )

    st.write(
        f"Created {len(chunks)} text chunks."
    )


    # Convert chunks into embeddings
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")


    # Create FAISS similarity index
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)


    st.success(
        "Document is ready for searching."
    )


    # Question input
    question = st.text_input(
        "Ask a question about the document"
    )


    if question:

        # Convert question into an embedding
        question_embedding = model.encode(
            [question],
            normalize_embeddings=True
        )

        question_embedding = np.array(
            question_embedding
        ).astype("float32")


        # Find the 3 most relevant chunks
        scores, indices = index.search(
            question_embedding,
            k=3
        )


        # Collect relevant chunks
        relevant_chunks = []

        for i in indices[0]:
            relevant_chunks.append(
                chunks[i]
            )


        # Combine retrieved chunks
        context = "\n\n".join(
            relevant_chunks
        )


        # Prompt sent to OpenAI
        prompt = f"""
Use only the document context below to answer the question.

If the answer is not in the document, say:
"I could not find that information in the document."

Document context:
{context}

Question:
{question}
"""


        # Ask the language model
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )


        # Display final answer
        st.subheader("Answer")

        st.write(
            response.output_text
        )


        # Allow user to inspect retrieved information
        with st.expander(
            "View retrieved document sections"
        ):

            for rank, i in enumerate(
                indices[0]
            ):

                st.write(
                    f"Result {rank + 1}"
                )

                st.write(
                    chunks[i]
                )

                st.caption(
                    f"Similarity score: "
                    f"{scores[0][rank]:.3f}"
                )

                st.divider()