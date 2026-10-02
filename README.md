# Enterprise RAG Assistant

A Python-based Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content.

The application retrieves relevant sections of the document using semantic search and provides them to an OpenAI language model to generate grounded answers.

## Features

- Upload and process PDF documents
- Extract and split document text into manageable chunks
- Generate semantic embeddings using Sentence Transformers
- Perform vector similarity search using FAISS
- Retrieve relevant document sections for a user's question
- Generate grounded answers using the OpenAI API
- Return a fallback response when information is not available in the document
- Display retrieved document sections and similarity scores
- Automated testing with Pytest
- Continuous Integration (CI) with GitHub Actions

## Technologies

- Python
- Streamlit
- OpenAI API
- Sentence Transformers
- FAISS
- PyPDF
- LangChain Text Splitters
- NumPy
- Pytest
- GitHub Actions

## How It Works

1. The user uploads a PDF document.
2. Text is extracted from the PDF.
3. The text is divided into smaller chunks.
4. Sentence Transformers convert the chunks into embeddings.
5. FAISS stores the embeddings for semantic similarity search.
6. The user's question is converted into an embedding.
7. The most relevant document chunks are retrieved.
8. The retrieved information is provided to the OpenAI model as context.
9. The model generates an answer based on that context.
10. The application displays the answer and the retrieved source sections.

## Testing

The project includes automated tests using Pytest.

Run the tests locally with:

```bash
pytest
```

GitHub Actions automatically installs the project dependencies and runs the test suite whenever changes are pushed to the repository.

## Setup

Clone the repository:

```bash
git clone https://github.com/RobJorge78/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git and should never be committed.

Run the application:

```bash
streamlit run app.py
```

## Security

API credentials are stored locally using environment variables and are excluded from version control through `.gitignore`.

Uploaded documents are processed by the application and are not included in the Git repository.

## Project Purpose

This project demonstrates practical experience with Retrieval-Augmented Generation, semantic search, LLM integration, prompt engineering, automated testing, CI, and secure API credential handling.

## Agentic Document Actions

The assistant includes a lightweight agent layer that selects an action based on the user's request.

Supported actions:

- **Search** – answers questions using relevant document sections retrieved through semantic search.
- **Summarize** – generates a summary using the uploaded document.
- **Unsupported Action Handling** – prevents unsupported requests from being executed.

The agent logic is separated into `agent.py` and is covered by automated tests in `tests/test_agent.py`.

This demonstrates basic agentic workflow design alongside the RAG pipeline.