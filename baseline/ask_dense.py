import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")

EMBEDDINGS_FILE = Path("rag_data/embeddings.npy")
METADATA_FILE = Path("rag_data/metadata.json")

TOP_K = 5


def validate_setup():
    """Validate environment variables and local index files."""

    variables = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
        "AZURE_CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
    }

    missing = [
        name
        for name, value in variables.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing environment variables: "
            + ", ".join(missing)
        )

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {EMBEDDINGS_FILE}. "
            "Run build_index.py first."
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {METADATA_FILE}. "
            "Run build_index.py first."
        )


def create_client():
    """Create a client for the Azure OpenAI v1 endpoint."""

    base_url = (
        AZURE_OPENAI_ENDPOINT.rstrip("/")
        + "/openai/v1/"
    )

    return OpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=base_url,
    )


def load_index():
    """Load normalized document embeddings and chunk metadata."""

    embeddings = np.load(EMBEDDINGS_FILE)

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array."
        )

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Embedding count and metadata count do not match. "
            f"Embeddings: {embeddings.shape[0]}, "
            f"metadata records: {len(metadata)}"
        )

    return embeddings, metadata


def normalize_vector(vector):
    """Normalize one vector for cosine similarity."""

    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError(
            "Embedding vector has zero norm."
        )

    return vector / norm


def get_question_embedding(client, question):
    """Generate an embedding for the user's question."""

    response = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=question,
    )

    embedding = response.data[0].embedding

    return normalize_vector(embedding)


def retrieve_chunks(
    question_embedding,
    document_embeddings,
    metadata,
):
    """Retrieve the TOP_K chunks most similar to the question."""

    if (
        document_embeddings.shape[1]
        != question_embedding.shape[0]
    ):
        raise ValueError(
            "Question and document embedding dimensions "
            "do not match. "
            f"Question: {question_embedding.shape[0]}, "
            f"documents: {document_embeddings.shape[1]}"
        )

    # build_index.py stored normalized document vectors.
    # The question vector is also normalized.
    # Dot product therefore gives cosine similarity.
    similarity_scores = (
        document_embeddings
        @ question_embedding
    )

    result_count = min(
        TOP_K,
        len(metadata),
    )

    top_indices = np.argsort(
        similarity_scores
    )[-result_count:][::-1]

    retrieved_chunks = []

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):
        index = int(index)

        result = metadata[index].copy()
        result["rank"] = rank
        result["score"] = float(
            similarity_scores[index]
        )

        retrieved_chunks.append(result)

    return retrieved_chunks


def print_retrieved_chunks(retrieved_chunks):
    """Print retrieved passages for debugging."""

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for result in retrieved_chunks:
        print(
            f"{result['rank']}. "
            f"{result['document_name']} | "
            f"Page {result['page_number']} | "
            f"Score: {result['score']:.4f}"
        )

        preview = (
            result["content"]
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 300:
            preview = preview[:300] + "..."

        print(f"   {preview}")
        print()


def build_context(retrieved_chunks):
    """Format retrieved passages for the chat model."""

    context_parts = []

    for result in retrieved_chunks:
        source_header = (
            f"[Source {result['rank']}: "
            f"{result['document_name']}, "
            f"page {result['page_number']}]"
        )

        context_parts.append(
            f"{source_header}\n"
            f"{result['content']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(
    client,
    question,
    retrieved_chunks,
):
    """Generate an answer using only retrieved policy context."""

    context = build_context(
        retrieved_chunks
    )

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
        model=CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError(
            "The generation model returned an empty answer."
        )

    return answer


def process_question(
    client,
    embeddings,
    metadata,
    question,
):
    """Run retrieval and grounded generation for one question."""

    print("\nGenerating question embedding...")

    question_embedding = get_question_embedding(
        client,
        question,
    )

    print(
        "Retrieving relevant policy passages..."
    )

    retrieved_chunks = retrieve_chunks(
        question_embedding,
        embeddings,
        metadata,
    )

    print_retrieved_chunks(
        retrieved_chunks
    )

    print("Generating grounded answer...")

    answer = generate_answer(
        client,
        question,
        retrieved_chunks,
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
            f"[Source {result['rank']}] "
            f"{result['document_name']}, "
            f"page {result['page_number']}"
        )


def main():
    """Start the command-line RAG assistant."""

    validate_setup()

    client = create_client()
    embeddings, metadata = load_index()

    print("=" * 70)
    print("INSURANCE POLICY RAG ASSISTANT")
    print("=" * 70)
    print(f"Indexed chunks: {len(metadata)}")
    print(
        f"Vector dimensions: "
        f"{embeddings.shape[1]}"
    )
    print(
        f"Embedding deployment: "
        f"{EMBEDDING_DEPLOYMENT}"
    )
    print(
        f"Chat deployment: "
        f"{CHAT_DEPLOYMENT}"
    )
    print()
    print("Type 'exit' to stop.")

    while True:
        question = input(
            "\nEnter your question: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
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
            )

        except Exception as error:
            print()
            print("Question processing failed.")
            print(
                f"Error type: "
                f"{type(error).__name__}"
            )
            print(
                f"Error details: {error}"
            )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nApplication stopped.")

    except Exception as error:
        print()
        print("Application startup failed.")
        print(
            f"Error type: "
            f"{type(error).__name__}"
        )
        print(
            f"Error details: {error}"
        )
        raise SystemExit(1)