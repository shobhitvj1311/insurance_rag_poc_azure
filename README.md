# Insurance Claims & Policy RAG Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about insurance policy documents, endorsements, schedules and claims guides. Answers are grounded strictly in the retrieved document text, with source citations.

The project compares four retrieval configurations so their impact can be measured rather than assumed:

| Configuration | Retrieval | Reranking |
|---|---|---|
| Dense | Embedding similarity only | Off |
| Dense + Rerank | Embedding similarity | BGE cross-encoder |
| Hybrid | Embedding similarity + BM25 keyword score | Off |
| Hybrid + Rerank | Embedding similarity + BM25 keyword score | BGE cross-encoder |

All configurations share the same chunking, embedding model, and answer-generation step, so differences in results come from retrieval strategy alone.

## How it works

```
                 documents/*.pdf
                        |
        split into page-level chunks (3000 chars, 500 overlap)
                        |
        embed chunks with Azure OpenAI embeddings
                        |
        +---------------+----------------+
        |                                |
  rag_data/dense/                  rag_data/hybrid/
  embeddings + metadata            embeddings + metadata + BM25 corpus


 question --> embed --> retrieve (dense or hybrid)
                              |
                  [optional] over-fetch 20 candidates
                  and rerank with BGE cross-encoder
                              |
                       top 5 chunks
                              |
              grounded answer generation (Azure OpenAI chat)
                              |
                   answer + [Source N] citations
```

### Retrieval

- **Dense:** the question is embedded and compared with every chunk embedding by cosine similarity (embeddings are L2-normalised, so a dot product is used).
- **Hybrid:** dense similarity and BM25 keyword scores are each min-max normalised across the corpus and blended with configurable weights (default 0.5 / 0.5). BM25 helps with exact terms such as policy numbers, registrations and amounts.
- **Reranking:** when enabled, retrieval over-fetches a candidate pool (default 20) and a BGE cross-encoder (`BAAI/bge-reranker-base`) scores each (question, chunk) pair. The top 5 by reranker score are kept.

### Generation

The chat model receives the retrieved chunks and answers using only that context. The prompt requires `[Source N]` citations, forbids inventing policy terms, and includes refusal rules: if the retrieved documents do not concern the person, vehicle, policy or claim asked about, the assistant should say no matching document was found and stop, rather than offering another policyholder's details as an alternative.

## Repository layout

```
insurance_rag_poc_azure/
  config.py                        # all tunable, non-secret settings
  requirements.txt
  .env                             # Azure credentials and deployment names (not committed)
  documents/                       # source PDFs
  rag_data/
    dense/                         # embeddings.npy, metadata.json
    hybrid/                        # embeddings.npy, metadata.json, bm25_corpus.json
  retrieval/
    common/
      base.py                      # chunking, embeddings, retrieval, generation
      reranker.py                  # BGE cross-encoder reranking
    dense/
      build_index_dense.py         # build the dense index
      ask_dense.py                 # interactive CLI, dense retrieval
    hybrid/
      build_index_hybrid.py        # build the hybrid index
      ask_hybrid.py                # interactive CLI, hybrid retrieval
  evaluation/
    generate_questions.py          # build the test set with reference answers
    evaluate_rag.py                # RAGAS evaluation
    evaluation_questions.json      # generated test set
    results/                       # one folder per configuration
    README.md                      # evaluation guide
  ui/
    compare_app.py                 # Streamlit UI to compare configurations
```

## Setup

### 1. Install dependencies

Use a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` should include:

```
openai
azure-storage-blob
pypdf
numpy
python-dotenv
rank-bm25
ragas
pandas
scikit-learn
langchain-openai
datasets
sentence-transformers
streamlit
```

`sentence-transformers` pulls in `torch`, so the first install is large. The first reranked query downloads the BGE model (about 1 GB) to the local Hugging Face cache.

### 2. Configure credentials

Create a `.env` file at the repository root:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-key>
AZURE_EMBEDDING_DEPLOYMENT=<embedding deployment name>
AZURE_CHAT_DEPLOYMENT=<chat deployment name>
AZURE_OPENAI_API_VERSION=2024-12-01-preview      # optional, used by the evaluation scripts
```

Secrets and deployment names stay in `.env`. Everything else is in `config.py`.

### 3. Add documents

Place the policy, endorsement, schedule and guide PDFs in `documents/`. The PDFs must contain selectable text; scanned, image-only PDFs produce no chunks.

### 4. Build the indexes

Run from the repository root:

```bash
python retrieval/dense/build_index_dense.py
python retrieval/hybrid/build_index_hybrid.py
```

Rebuild whenever the PDFs, chunking settings, or the embedding model change. Embeddings from different models are not interchangeable, so changing `AZURE_EMBEDDING_DEPLOYMENT` requires rebuilding both indexes.

## Asking questions from the command line

```bash
python retrieval/dense/ask_dense.py
python retrieval/hybrid/ask_hybrid.py
```

Each accepts a per-session reranking flag. Without a flag, the default comes from `RERANK_ENABLED` in `config.py`.

```bash
python retrieval/hybrid/ask_hybrid.py --rerank
python retrieval/hybrid/ask_hybrid.py --no-rerank
python retrieval/dense/ask_dense.py --rerank
python retrieval/dense/ask_dense.py --no-rerank
```

The startup banner shows whether reranking is enabled. Type `exit` to quit.

For each question the CLI prints the retrieved sources, the generated answer, and the list of source references.

## Comparison UI

A local Streamlit app runs one question through the configurations you choose and shows the answers and retrieved sources side by side.

```bash
streamlit run ui/compare_app.py
```

This opens a page at `http://localhost:8501`. Type a question, tick one or more of Dense, Dense + Rerank, Hybrid, Hybrid + Rerank, and click Run. Only the indexes and models needed for the selected configurations are loaded. Each run makes live Azure OpenAI calls, so expect a few seconds per configuration and normal API cost.

## Configuration reference

All settings live in `config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 3000 / 500 | Chunk length and overlap in characters |
| `EMBEDDING_BATCH_SIZE` | 16 | Chunks embedded per API call |
| `EMBEDDING_MAX_RETRIES` | 3 | Retry attempts per embedding batch |
| `TOP_K` | 5 | Final number of chunks passed to the chat model |
| `DENSE_WEIGHT` / `BM25_WEIGHT` | 0.5 / 0.5 | Hybrid blend weights; must sum to 1.0 |
| `RERANK_ENABLED` | True | Default reranking behaviour for CLI scripts |
| `RERANK_MODEL_NAME` | `BAAI/bge-reranker-base` | Cross-encoder model |
| `RERANK_CANDIDATE_POOL` | 20 | Chunks fetched before reranking; must be at least `RERANK_TOP_K` |
| `RERANK_TOP_K` | `TOP_K` | Chunks kept after reranking |
| `RERANK_DEVICE` | `cpu` | Set to `cuda` if a GPU is available |
| `DEFAULT_RETRIEVAL_MODE` | `hybrid` | Default for `evaluate_rag.py` when `--mode` is omitted |
| `EVAL_TARGET_QUESTIONS` | 50 | Size of the generated evaluation set |

## Evaluation

The evaluation framework generates a 50-question test set with reference answers and scores each configuration with RAGAS metrics (context precision, context recall, faithfulness, answer relevancy). Results are written to a separate folder per configuration. See [`evaluation/README.md`](evaluation/README.md) for the full guide.

## Known limitations

- **No relevance floor before generation.** Retrieval always returns the top-k chunks, even when none are a good match. The refusal behaviour currently relies on the system prompt alone. A score threshold that skips generation when nothing relevant is retrieved is a planned improvement; it needs calibration per method because the scores are on different scales.
- **Scores are not comparable across methods.** The displayed score is raw cosine similarity for dense, a pool-relative blended 0-1 score for hybrid, and an unbounded cross-encoder score after reranking.
- **Chunks never span pages.** Chunking is applied per page, so a fact split across a page break sits in two separate chunks.
- **BM25 is rebuilt on every query** and uses simple lowercase whitespace tokenisation, with no stemming or punctuation handling. This is fine at the current corpus size but does not scale.
- **Single-document assumptions in questions.** Questions that need facts from several documents can score low on recall with top-5 retrieval.
- **Reasoning-style chat deployments** only accept the default temperature, which is why the evaluation script overrides the temperature RAGAS sends.

## Troubleshooting

| Problem | Fix |
|---|---|
| `File not found: embeddings.npy` | Run the matching `build_index_*.py` script first |
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers`, in the same environment you launch from |
| First reranked query is slow | The BGE model is downloading; later queries reuse the cached model |
| Reranking is slow on CPU | Set `RERANK_DEVICE = "cuda"` if a GPU is available, or use `--no-rerank` |
| Missing environment variables error | Check `.env` has all four Azure values |
| No chunks created | The PDFs are probably scanned images; use text-based PDFs or add OCR |