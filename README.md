
# Enterprise Banking Knowledge Engine

A Retrieval-Augmented Generation (RAG) application that answers questions about banking policy documents using semantic search, a vector database, and a locally running large language model.

Built with Python, LangChain, ChromaDB, FastAPI, and Ollama.

## Features

- **Document ingestion:** Loads text-based banking documents.
- **Text chunking:** Splits documents into smaller, searchable sections.
- **Semantic search:** Uses sentence embeddings to retrieve relevant content.
- **Vector database:** Stores and searches document embeddings using ChromaDB.
- **Local AI inference:** Generates answers using a local Ollama model.
- **Source citations:** Returns source filenames, chunk IDs, and supporting text.
- **REST API:** Provides endpoints through FastAPI.
- **Input validation:** Validates incoming questions with Pydantic.
- **Automated tests:** Tests API endpoints and invalid input handling.

## Architecture

```text
Banking Policy Documents
          |
          v
   Document Ingestion
          |
          v
     Text Chunking
          |
          v
 Sentence Embeddings
          |
          v
    ChromaDB Vector Store
          |
          v
      User Question
          |
          v
     Semantic Search
          |
          v
  Relevant Document Chunks
          |
          v
   Local Ollama LLM
          |
          v
 Answer + Source Citations
          |
          v
       FastAPI
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| LangChain | RAG pipeline orchestration |
| Hugging Face Sentence Transformers | Text embeddings |
| ChromaDB | Vector storage and similarity search |
| Ollama | Local language model inference |
| Qwen 2.5 (1.5B) | Answer generation |
| FastAPI | REST API |
| Pydantic | Request and response validation |
| Pytest | Automated testing |

## Project Structure

```text
enterprise-banking-rag/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── main.py
├── data/
│   └── documents/
│       └── sample_loan_policy.txt
├── tests/
│   └── test_api.py
├── chroma_db/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- Python installed
- Ollama installed
- Git installed

The project was developed and tested in a local Python virtual environment.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd enterprise-banking-rag
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install the local language model

Make sure Ollama is installed, then run:

```bash
ollama pull qwen2.5:1.5b
```

### 5. Build the vector database

```bash
python -m app.retriever
```

This command creates the vector database from the text documents in `data/documents/`.

### 6. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

- Home: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

## Example API Request

Send a POST request to `/ask`:

```json
{
  "question": "What documents are needed for a personal loan?"
}
```

Example response structure:

```json
{
  "answer": "Applicants may be asked to provide identity verification, address verification, income proof, and employment-related documents.",
  "sources": [
    {
      "document": "sample_loan_policy.txt",
      "chunk_id": 0,
      "content": "Relevant retrieved document text..."
    }
  ]
}
```

The answer and retrieved source content may vary depending on the question and retrieval results.

## Running Tests

Run the automated test suite:

```bash
python -m pytest -v
```

The current test suite covers the home endpoint, health endpoint, question-answering endpoint, and rejection of short questions.

## Limitations

- The included loan policy is fictional and intended only for educational demonstration.
- The project does not connect to a real bank's internal systems.
- Answer quality depends on the available documents and retrieved context.
- The local language model may occasionally produce incomplete or inaccurate responses.
- This project is not intended for real financial decisions or production banking use.

## Future Improvements

- Support PDF document ingestion.
- Add document-level and chunk-level retrieval evaluation.
- Improve answer-grounding and citation verification.
- Add a web-based chat interface.
- Add authentication and rate limiting.
- Expand the document collection using suitable, authorized sources.

## Disclaimer

This is an educational software project. The included documents are fictional examples, not official banking policies, legal advice, or regulatory guidance.

## License

Add your chosen open-source license to this repository and include its corresponding license file.
