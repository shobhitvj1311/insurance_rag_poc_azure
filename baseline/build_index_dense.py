import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

DOCUMENTS_DIRECTORY = Path("documents")
OUTPUT_DIRECTORY = Path("rag_data")

EMBEDDINGS_FILE = OUTPUT_DIRECTORY / "embeddings.npy"
METADATA_FILE = OUTPUT_DIRECTORY / "metadata.json"

# Character-based chunking is easier to inspect for the first POC.
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 500
EMBEDDING_BATCH_SIZE = 16


def validate_configuration():
    """Validate environment variables and local directories."""

    required_values = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
    }

    missing = [
        name
        for name, value in required_values.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing environment variables: " + ", ".join(missing)
        )

    if not DOCUMENTS_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIRECTORY}"
        )


def create_openai_client():
    """Create a client for the Azure OpenAI v1 endpoint."""

    base_url = (
        AZURE_OPENAI_ENDPOINT.rstrip("/")
        + "/openai/v1/"
    )

    return OpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=base_url,
    )


def clean_text(text):
    """Normalize whitespace extracted from a PDF page."""

    if not text:
        return ""

    lines = [
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.

    Whenever possible, the end of a chunk is moved backward to a
    paragraph, sentence, or word boundary.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        proposed_end = min(start + chunk_size, text_length)
        end = proposed_end

        if proposed_end < text_length:
            search_start = start + (chunk_size // 2)
            candidate_text = text[search_start:proposed_end]

            possible_breaks = [
                candidate_text.rfind("\n\n"),
                candidate_text.rfind(". "),
                candidate_text.rfind(" "),
            ]

            best_break = max(possible_breaks)

            if best_break != -1:
                end = search_start + best_break + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap

        # Protection against an accidental infinite loop
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def extract_pdf_chunks(pdf_path):
    """Extract page-level chunks and citation metadata from one PDF."""

    reader = PdfReader(str(pdf_path))
    records = []

    print(
        f"Reading: {pdf_path.name} "
        f"({len(reader.pages)} page(s))"
    )

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1

        try:
            page_text = clean_text(page.extract_text())
        except Exception as error:
            print(
                f"  Warning: Could not extract page {page_number}. "
                f"{type(error).__name__}: {error}"
            )
            continue

        if not page_text:
            print(
                f"  Warning: No selectable text found "
                f"on page {page_number}."
            )
            continue

        page_chunks = split_text(page_text)

        for chunk_index, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):
            chunk_id = (
                f"{pdf_path.stem}"
                f"-page-{page_number}"
                f"-chunk-{chunk_index}"
            )

            records.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page_number": page_number,
                    "chunk_number": chunk_index,
                    "content": chunk_text,
                }
            )

    return records


def extract_all_documents():
    """Extract chunks from every PDF in the documents directory."""

    pdf_files = sorted(
        DOCUMENTS_DIRECTORY.rglob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files were found inside the documents directory."
        )

    print(f"Found {len(pdf_files)} PDF file(s).")
    print()

    all_records = []

    for pdf_path in pdf_files:
        document_records = extract_pdf_chunks(pdf_path)
        all_records.extend(document_records)

        print(
            f"  Created {len(document_records)} chunk(s) "
            f"from {pdf_path.name}"
        )
        print()

    if not all_records:
        raise ValueError(
            "No text chunks were created. "
            "The PDFs may be scanned or image-based."
        )

    return all_records


def generate_embeddings(client, records):
    """Generate embeddings in small batches."""

    all_embeddings = []
    total_records = len(records)

    print(f"Generating embeddings for {total_records} chunks...")

    for batch_start in range(
        0,
        total_records,
        EMBEDDING_BATCH_SIZE,
    ):
        batch_end = min(
            batch_start + EMBEDDING_BATCH_SIZE,
            total_records,
        )

        batch_records = records[batch_start:batch_end]
        batch_texts = [
            record["content"]
            for record in batch_records
        ]

        maximum_attempts = 3

        for attempt in range(1, maximum_attempts + 1):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_DEPLOYMENT,
                    input=batch_texts,
                )

                ordered_data = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                batch_embeddings = [
                    item.embedding
                    for item in ordered_data
                ]

                all_embeddings.extend(batch_embeddings)

                print(
                    f"  Embedded chunks "
                    f"{batch_start + 1} to {batch_end}"
                )
                break

            except Exception as error:
                if attempt == maximum_attempts:
                    raise

                wait_seconds = attempt * 5

                print(
                    f"  Embedding attempt {attempt} failed: "
                    f"{type(error).__name__}"
                )
                print(
                    f"  Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    return np.asarray(all_embeddings, dtype=np.float32)


def normalize_embeddings(embeddings):
    """
    Normalize vectors so dot product can be used as cosine similarity.
    """

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1

    return embeddings / norms


def save_index(embeddings, records):
    """Save vectors and citation metadata locally."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_embeddings = normalize_embeddings(embeddings)

    np.save(
        EMBEDDINGS_FILE,
        normalized_embeddings,
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            records,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Local vector index saved successfully.")
    print(f"Embedding file: {EMBEDDINGS_FILE}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Number of chunks: {len(records)}")
    print(f"Vector dimensions: {embeddings.shape[1]}")


def main():
    validate_configuration()

    client = create_openai_client()

    records = extract_all_documents()
    embeddings = generate_embeddings(client, records)

    if len(records) != len(embeddings):
        raise ValueError(
            "Metadata and embedding counts do not match."
        )

    save_index(embeddings, records)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("Index creation failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")
        raise SystemExit(1)