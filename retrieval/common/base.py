import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# base.py lives at <repo_root>/retrieval/common/base.py
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (
    BM25_CORPUS_FILENAME,
    BM25_WEIGHT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DENSE_OUTPUT_DIRECTORY,
    DENSE_WEIGHT,
    DOCUMENTS_DIRECTORY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_BACKOFF_SECONDS,
    EMBEDDINGS_FILENAME,
    HYBRID_OUTPUT_DIRECTORY,
    METADATA_FILENAME,
    RERANK_CANDIDATE_POOL,
    RERANK_ENABLED,
    RERANK_TOP_K,
    TOP_K,
)

load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")


def validate_configuration():
    """Validate required environment variables and source documents."""

    required_values = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
    }

    missing = [
        name for name, value in required_values.items() if not value
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

    base_url = AZURE_OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/"
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
    """Split text into overlapping chunks while preserving readability."""

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
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def extract_pdf_chunks(pdf_path):
    """Extract page-level chunks and citation metadata for one PDF."""

    reader = PdfReader(str(pdf_path))
    records = []

    print(f"Reading: {pdf_path.name} ({len(reader.pages)} page(s))")

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
                f"  Warning: No selectable text found on page {page_number}."
            )
            continue

        page_chunks = split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = (
                f"{pdf_path.stem}-page-{page_number}-chunk-{chunk_index}"
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

    pdf_files = sorted(DOCUMENTS_DIRECTORY.rglob("*.pdf"))

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
            f"  Created {len(document_records)} chunk(s) from {pdf_path.name}"
        )
        print()

    if not all_records:
        raise ValueError(
            "No text chunks were created. The PDFs may be scanned or image-based."
        )

    return all_records


def generate_embeddings(client, records):
    """Generate embeddings in small batches."""

    all_embeddings = []
    total_records = len(records)

    print(f"Generating embeddings for {total_records} chunks...")

    for batch_start in range(0, total_records, EMBEDDING_BATCH_SIZE):
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_records)
        batch_records = records[batch_start:batch_end]
        batch_texts = [record["content"] for record in batch_records]

        for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
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
                    item.embedding for item in ordered_data
                ]
                all_embeddings.extend(batch_embeddings)

                print(
                    f"  Embedded chunks {batch_start + 1} to {batch_end}"
                )
                break

            except Exception as error:
                if attempt == EMBEDDING_MAX_RETRIES:
                    raise

                wait_seconds = attempt * EMBEDDING_RETRY_BACKOFF_SECONDS
                print(
                    f"  Embedding attempt {attempt} failed: "
                    f"{type(error).__name__}"
                )
                print(f"  Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    return np.asarray(all_embeddings, dtype=np.float32)


def normalize_embeddings(embeddings):
    """Normalize vectors so dot product can be used as cosine similarity."""

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def save_dense_index(embeddings, records, output_directory):
    """Persist vector embeddings and metadata for dense retrieval."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    normalized_embeddings = normalize_embeddings(embeddings)
    embeddings_file = output_directory / EMBEDDINGS_FILENAME
    metadata_file = output_directory / METADATA_FILENAME

    np.save(embeddings_file, normalized_embeddings)

    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    print()
    print("Local vector index saved successfully.")
    print(f"Embedding file: {embeddings_file}")
    print(f"Metadata file: {metadata_file}")
    print(f"Number of chunks: {len(records)}")
    print(f"Vector dimensions: {embeddings.shape[1]}")


def save_hybrid_index(embeddings, records, output_directory):
    """Persist dense index plus BM25 corpus."""

    save_dense_index(embeddings, records, output_directory)

    corpus_path = Path(output_directory) / BM25_CORPUS_FILENAME
    with open(corpus_path, "w", encoding="utf-8") as handle:
        json.dump(
            [record["content"] for record in records],
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"BM25 corpus saved: {corpus_path}")


def build_dense_index(output_directory=DENSE_OUTPUT_DIRECTORY):
    """Build a dense-only vector index."""

    validate_configuration()
    client = create_openai_client()
    records = extract_all_documents()
    embeddings = generate_embeddings(client, records)

    if len(records) != len(embeddings):
        raise ValueError("Metadata and embedding counts do not match.")

    save_dense_index(embeddings, records, output_directory)
    return output_directory


def build_hybrid_index(output_directory=HYBRID_OUTPUT_DIRECTORY):
    """Build a hybrid index by saving dense embeddings and BM25 corpus."""

    validate_configuration()
    client = create_openai_client()
    records = extract_all_documents()
    embeddings = generate_embeddings(client, records)

    if len(records) != len(embeddings):
        raise ValueError("Metadata and embedding counts do not match.")

    save_hybrid_index(embeddings, records, output_directory)
    return output_directory


def load_dense_index(index_directory):
    """Load a dense index from disk."""

    index_directory = Path(index_directory)
    embeddings_file = index_directory / EMBEDDINGS_FILENAME
    metadata_file = index_directory / METADATA_FILENAME

    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"File not found: {embeddings_file}. Run the index builder first."
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"File not found: {metadata_file}. Run the index builder first."
        )

    embeddings = np.load(embeddings_file)

    with open(metadata_file, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Embedding count and metadata count do not match. "
            f"Embeddings: {embeddings.shape[0]}, metadata records: {len(metadata)}"
        )

    return embeddings, metadata


def load_hybrid_index(index_directory):
    """Load a hybrid index from disk."""

    embeddings, metadata = load_dense_index(index_directory)
    corpus_path = Path(index_directory) / BM25_CORPUS_FILENAME

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"File not found: {corpus_path}. Run the hybrid index builder first."
        )

    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus_texts = json.load(file)

    return embeddings, metadata, corpus_texts


def normalize_vector(vector):
    """Normalize one vector for cosine similarity."""

    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Embedding vector has zero norm.")

    return vector / norm


def get_question_embedding(client, question, embedding_name=None):
    """Generate a normalized embedding for a question."""

    model_name = embedding_name or EMBEDDING_DEPLOYMENT
    response = client.embeddings.create(
        model=model_name,
        input=question,
    )
    embedding = response.data[0].embedding
    return normalize_vector(embedding)


def retrieve_dense_chunks(
    question_embedding,
    document_embeddings,
    metadata,
    top_k=TOP_K,
):
    """Retrieve the top-k chunks by cosine similarity."""

    if document_embeddings.shape[1] != question_embedding.shape[0]:
        raise ValueError(
            "Question and document embedding dimensions do not match. "
            f"Question: {question_embedding.shape[0]}, documents: {document_embeddings.shape[1]}"
        )

    similarity_scores = document_embeddings @ question_embedding

    result_count = min(top_k, len(metadata))
    top_indices = np.argsort(similarity_scores)[-result_count:][::-1]

    retrieved = []
    for rank, index in enumerate(top_indices, start=1):
        index = int(index)
        result = metadata[index].copy()
        result["rank"] = rank
        result["score"] = float(similarity_scores[index])
        retrieved.append(result)

    return retrieved


def retrieve_hybrid_chunks(
    question_embedding,
    document_embeddings,
    metadata,
    corpus_texts,
    question,
    top_k=TOP_K,
    dense_weight=DENSE_WEIGHT,
    bm25_weight=BM25_WEIGHT,
):
    """Blend dense cosine similarity with BM25 lexical matching."""

    dense_scores = document_embeddings @ question_embedding
    dense_min = float(np.min(dense_scores))
    dense_max = float(np.max(dense_scores))

    if dense_max != dense_min:
        dense_norm = (dense_scores - dense_min) / (dense_max - dense_min)
    else:
        dense_norm = np.zeros_like(dense_scores, dtype=np.float32)

    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = question.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_min = float(np.min(bm25_scores))
    bm25_max = float(np.max(bm25_scores))

    if bm25_max != bm25_min:
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_norm = np.zeros_like(bm25_scores, dtype=np.float32)

    combined = dense_weight * dense_norm + bm25_weight * bm25_norm
    top_indices = np.argsort(combined)[-top_k:][::-1]

    results = []
    for rank, index in enumerate(top_indices, start=1):
        index = int(index)
        result = metadata[index].copy()
        result["rank"] = rank
        result["dense_score"] = float(dense_scores[index])
        result["bm25_score"] = float(bm25_scores[index])
        result["score"] = float(combined[index])
        results.append(result)

    return results


def print_retrieved_chunks(retrieved_chunks):
    """Print retrieved passages for debugging."""

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for result in retrieved_chunks:
        print(
            f"{result['rank']}. {result['document_name']} | "
            f"Page {result['page_number']} | Score: {result['score']:.4f}"
        )

        preview = result["content"].replace("\n", " ").strip()
        if len(preview) > 300:
            preview = preview[:300] + "..."

        print(f"   {preview}")
        print()


def build_context(retrieved_chunks):
    """Format retrieved chunks for the chat model."""

    context_parts = []
    for result in retrieved_chunks:
        source_header = (
            f"[Source {result['rank']}: {result['document_name']}, "
            f"page {result['page_number']}]"
        )
        context_parts.append(f"{source_header}\n{result['content']}")

    return "\n\n".join(context_parts)


def generate_answer(client, question, retrieved_chunks, chat_name=None):
    """Generate an answer grounded only in retrieved policy context."""

    context = build_context(retrieved_chunks)
    system_prompt = """
You are an insurance policy and claims knowledge assistant.

Use only the document context supplied by the user.

Rules:
1. Do not use general knowledge to fill information gaps.
2. Do not invent policy language, coverage, exclusions, limits,
   definitions, conditions, or facts.
3. Clearly distinguish coverage, exclusions, conditions, duties,
   definitions, and limits.
4. Cite each important conclusion using labels such as [Source 1].
5. Do not cite a source unless it supports the statement.
6. If the context is insufficient, state that the supplied document
   context does not contain enough information.
7. Do not make a final coverage determination.
8. State that final coverage depends on the complete policy,
   endorsements, facts of loss, applicable law, and claims review.
9. Keep the answer concise and professionally worded.
""".strip()

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

QUESTION:

{question}
""".strip()

    response = client.chat.completions.create(
        model=chat_name or CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content
    if not answer:
        raise ValueError("The generation model returned an empty answer.")

    return answer


def process_question(
    client,
    embeddings,
    metadata,
    question,
    mode="dense",
    corpus_texts=None,
    use_rerank=RERANK_ENABLED,
):
    """Run retrieval and grounded generation for one question."""

    print("\nGenerating question embedding...")
    question_embedding = get_question_embedding(
        client,
        question,
        embedding_name=EMBEDDING_DEPLOYMENT,
    )

    print("Retrieving relevant policy passages...")

    # When reranking, over-fetch a wider candidate pool from the retriever;
    # the reranker then narrows it back down to the final top_k.
    retrieval_top_k = RERANK_CANDIDATE_POOL if use_rerank else TOP_K

    if mode == "hybrid":
        if corpus_texts is None:
            raise ValueError("Corpus text is required for hybrid retrieval.")
        retrieved_chunks = retrieve_hybrid_chunks(
            question_embedding,
            embeddings,
            metadata,
            corpus_texts,
            question,
            top_k=retrieval_top_k,
        )
    else:
        retrieved_chunks = retrieve_dense_chunks(
            question_embedding,
            embeddings,
            metadata,
            top_k=retrieval_top_k,
        )

    if use_rerank:
        from retrieval.common.reranker import rerank_chunks

        print("Reranking retrieved passages...")
        retrieved_chunks = rerank_chunks(
            question,
            retrieved_chunks,
            top_k=RERANK_TOP_K,
        )

    print_retrieved_chunks(retrieved_chunks)

    print("Generating grounded answer...")
    answer = generate_answer(
        client,
        question,
        retrieved_chunks,
        chat_name=CHAT_DEPLOYMENT,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("SOURCE REFERENCES")
    print("=" * 70)

    for result in retrieved_chunks:
        print(
            f"[Source {result['rank']}] {result['document_name']}, "
            f"page {result['page_number']}"
        )

    return answer