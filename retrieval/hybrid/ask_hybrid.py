import sys
from pathlib import Path

from dotenv import load_dotenv

# this file lives at <repo_root>/retrieval/hybrid/ask_hybrid.py
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (
    BM25_CORPUS_FILENAME,
    EMBEDDINGS_FILENAME,
    HYBRID_OUTPUT_DIRECTORY,
    METADATA_FILENAME,
)
from retrieval.common.base import (
    CHAT_DEPLOYMENT,
    EMBEDDING_DEPLOYMENT,
    create_openai_client,
    load_hybrid_index,
    process_question,
)


load_dotenv(override=True)


def validate_hybrid_setup(index_directory=HYBRID_OUTPUT_DIRECTORY):
    """Validate the hybrid index and environment variables."""

    required = {
        "AZURE_OPENAI_ENDPOINT": __import__("os").getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": __import__("os").getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
        "AZURE_CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
    }

    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))

    index_path = Path(index_directory)
    for file_name in [EMBEDDINGS_FILENAME, METADATA_FILENAME, BM25_CORPUS_FILENAME]:
        if not (index_path / file_name).exists():
            raise FileNotFoundError(
                f"File not found: {index_path / file_name}. Run the hybrid index builder first."
            )


def main(index_directory=HYBRID_OUTPUT_DIRECTORY):
    """Run the hybrid Q&A flow using BM25 + dense retrieval."""

    validate_hybrid_setup(index_directory)
    client = create_openai_client()
    embeddings, metadata, corpus_texts = load_hybrid_index(index_directory)

    print("=" * 70)
    print("HYBRID RETRIEVAL RAG ASSISTANT")
    print("=" * 70)
    print(f"Indexed chunks: {len(metadata)}")
    print(f"Vector dimensions: {embeddings.shape[1]}")
    print(f"Embedding deployment: {EMBEDDING_DEPLOYMENT}")
    print(f"Chat deployment: {CHAT_DEPLOYMENT}")
    print()
    print("Type 'exit' to stop.")

    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Closing the assistant.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            process_question(
                client,
                embeddings,
                metadata,
                question,
                mode="hybrid",
                corpus_texts=corpus_texts,
            )
        except Exception as error:
            print()
            print("Question processing failed.")
            print(f"Error type: {type(error).__name__}")
            print(f"Error details: {error}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as error:
        print()
        print("Application startup failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")
        raise SystemExit(1)