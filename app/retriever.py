
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.ingest import load_text_documents, split_documents


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Open-source embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "banking_documents"


def create_vector_store():
    """Create embeddings and store document chunks in ChromaDB."""

    documents = load_text_documents()
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text chunks found. Add a .txt file to data/documents."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Replace the existing collection during development.
    # This prevents duplicate or outdated chunks on repeated runs.
    try:
        vector_store.delete_collection()
    except Exception:
        pass

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]

    ids = [
        f"{chunk['source']}_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

    return vector_store


def search_documents(question, k=3):
    """Find the most relevant document chunks for a question."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    results = vector_store.similarity_search(
        question,
        k=k,
    )

    return results


if __name__ == "__main__":
    print("Creating the banking document vector database...")

    create_vector_store()

    print("\nSearching the documents...")

    question = "What documents are needed for a personal loan?"
    results = search_documents(question)

    print(f"\nQuestion: {question}")

    for index, document in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Source: {document.metadata['source']}")
        print(f"Text: {document.page_content}")
