from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def load_text_documents():
    """Load text documents from the documents directory."""

    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "text": text,
                "source": file_path.name,
            }
        )

    return documents


def split_documents(documents):
    """Split documents into smaller text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = []

    for document in documents:
        text_chunks = splitter.split_text(document["text"])

        for index, chunk in enumerate(text_chunks):
            chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "chunk_id": index,
                }
            )

    return chunks


if __name__ == "__main__":
    documents = load_text_documents()
    chunks = split_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Text chunks created: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 50)
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(chunk["text"])