# RAG Pipeline Evaluation (RAGAS)

This directory contains the scripts used to evaluate the insurance RAG assistant (hybrid BM25 + dense retrieval) with [RAGAS](https://docs.ragas.io) metrics.

## Overview

The evaluation has two stages:

1. **`generate_questions.py`** creates 50 evaluation questions from the PDFs in `documents/`. Each question comes with a ground-truth (reference) answer written from the source document.
2. **`evaluate_rag.py`** runs the real RAG pipeline on every question (hybrid retrieval, then answer generation) and scores the results with RAGAS.

```
documents/*.pdf
      |
      v
generate_questions.py  --->  evaluation/evaluation_questions.json
                                    (question, reference, source_document)
                                                |
hybrid index (embeddings.npy,                   |
metadata.json, bm25_corpus.json)                |
      |                                         v
      +------------------------------->  evaluate_rag.py
                                                |
                                                v
                     evaluation/ragas_results_<timestamp>.json
                     evaluation/ragas_report_<timestamp>.csv
```

## Metrics

### RAGAS metrics (LLM-judged)

| Metric | Range | What it measures | Low score suggests |
|---|---|---|---|
| **Context Precision** | 0-1 | Are the useful chunks ranked near the top of the retrieved list? | Noisy or poorly ranked retrieval |
| **Context Recall** | 0-1 | Does the retrieved context contain everything needed for the reference answer? | Relevant information is not being retrieved |
| **Faithfulness** | 0-1 | Is every claim in the answer supported by the retrieved context? | Hallucination or use of outside knowledge |
| **Answer Relevancy** | 0-1 | Does the answer address the question that was asked? | Off-topic, incomplete or rambling answers |

### Deterministic retrieval checks (no LLM)

| Metric | What it measures |
|---|---|
| **Source Hit** | 1 if the expected `source_document` appears anywhere in the top-k chunks, else 0 |
| **Source Precision** | Fraction of the top-k chunks that come from the expected document |

## Prerequisites

### Python packages

Add these to `requirements.txt` (on top of the existing packages):

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
```

Install in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The scripts target the RAGAS 0.2/0.3 API (`EvaluationDataset`, `LLMContextPrecisionWithReference`). After a working install, pin the version:

```bash
pip show ragas
# then add e.g. ragas==0.3.x to requirements.txt
```

### Environment variables (`.env`)

```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_CHAT_DEPLOYMENT=...
AZURE_EMBEDDING_DEPLOYMENT=...
AZURE_OPENAI_API_VERSION=2024-12-01-preview   # optional
```

## Running the evaluation

Run all commands from the repository root.

### Step 1: Build the hybrid index

Only needed the first time, or whenever the PDFs, chunking or embedding settings change:

```bash
python retrieval/hybrid/build_index_hybrid.py
```

This creates `embeddings.npy`, `metadata.json` and `bm25_corpus.json` in the hybrid output directory. `evaluate_rag.py` loads these files.

### Step 2: Generate evaluation questions and reference answers

```bash
python evaluation/generate_questions.py
```

What it does:

1. Reads all pages of every PDF in `documents/` (capped at 12,000 characters per document).
2. Splits the 50-question target across the documents as evenly as possible. With 33 documents, 17 get two questions and 16 get one.
3. Asks the LLM for question and answer pairs per document. Questions must be answerable from that document alone, specific (amounts, dates, conditions, exclusions, claims steps) and a mix of single-fact and multi-part. Reference answers must use only facts stated in the document.
4. If any generation fails or returns too few items, it runs up to 3 top-up rounds to reach exactly 50.
5. Saves everything to `evaluation/evaluation_questions.json`.

**Review the references.** They are LLM-written. Spot-check 5-10 against the PDFs before trusting the recall and precision scores, and edit any that are wrong.

### Step 3: Run the RAGAS evaluation

```bash
python evaluation/evaluate_rag.py
```

What it does:

1. Loads the 50 questions with their reference answers. It stops with a clear error if any reference is missing.
2. Loads the hybrid index.
3. For each question:
   - embeds the question and retrieves the top 5 chunks with hybrid retrieval (BM25 + dense),
   - generates an answer from those chunks using the assistant prompt,
   - records the deterministic checks (`source_hit`, `source_precision`).
4. Builds a RAGAS dataset (`user_input`, `retrieved_contexts`, `response`, `reference`) and scores all four RAGAS metrics.
5. Writes the JSON results and CSV report, and prints a summary.

**Runtime and cost:** 50 questions x 4 metrics = 200 RAGAS jobs, each an LLM call, and the run takes roughly 25-30 minutes. Reduce `max_workers` in `RunConfig` if you hit rate limits.

## Output files

Files are written to `evaluation/`:

| File | Contents |
|---|---|
| `evaluation_questions.json` | 50 questions with `question`, `reference`, `source_document` |
| `ragas_results_<timestamp>.json` | Per-question answer, reference, retrieved documents (rank, page, score), all metric scores, plus summary averages |
| `ragas_report_<timestamp>.csv` | One row per question with the scores, good for Excel or pandas |

### `evaluation_questions.json` structure

```json
{
  "metadata": {
    "total_questions": 50,
    "documents_used": 33,
    "generation_date": "...",
    "evaluation_type": "ragas",
    "has_reference_answers": true
  },
  "questions": [
    {
      "question": "What is the Total Standard Excess ...?",
      "reference": "The Total Standard Excess is ...",
      "source_document": "P01_schedule_2024.pdf"
    }
  ]
}
```

### `ragas_results_<timestamp>.json` structure

```json
{
  "metadata": {
    "evaluation_date": "...",
    "total_questions": 50,
    "chat_model": "...",
    "embedding_model": "...",
    "retrieval_method": "hybrid",
    "top_k": 5,
    "framework": "ragas"
  },
  "summary": {
    "avg_context_precision": 0.0,
    "avg_context_recall": 0.0,
    "avg_faithfulness": 0.0,
    "avg_answer_relevancy": 0.0,
    "avg_source_hit": 0.0,
    "avg_source_precision": 0.0
  },
  "results": [
    {
      "question_id": 1,
      "question": "...",
      "reference": "...",
      "answer": "...",
      "source_document": "...",
      "retrieved_documents": [{"rank": 1, "document": "...", "page": 1, "score": 0.0}],
      "source_hit": 1.0,
      "source_precision": 0.6,
      "context_precision": 0.0,
      "context_recall": 0.0,
      "faithfulness": 0.0,
      "answer_relevancy": 0.0
    }
  ]
}
```

## Interpreting the results

| Symptom | Likely cause | What to try |
|---|---|---|
| Low **source hit** or **context recall** | The right content is not retrieved | Adjust chunk size and overlap, increase `top_k`, tune the BM25/dense weighting |
| Low **context precision** | Right chunks retrieved but ranked low, or lots of noise | Improve ranking or fusion, reduce `top_k`, consider reranking |
| Low **faithfulness** | The model adds information beyond the context | Tighten the answer prompt |
| Low **answer relevancy** | Answers are off-topic or incomplete | Check the prompt, and check whether retrieval supplies the needed context |

Notes:

- Multi-part questions naturally score lower on recall, because every part must be covered by the retrieved chunks.
- Questions that need several documents will show low recall with top-5 retrieval. That is a genuine retrieval limitation, not a scoring error.
- Scores from an LLM judge vary slightly between runs. Compare changes using the averages, not single questions.

## Iterating

1. Open the CSV and sort by the lowest scores.
2. Inspect those rows in the JSON (`retrieved_documents`, `answer`, `reference`).
3. Change one thing (chunking, `top_k`, prompt, fusion weights), rebuild the index if needed, and rerun.
4. Compare the summary averages with the previous run.

## Troubleshooting

### `No module named 'langchain_community.chat_models.vertexai'`
A version mismatch between RAGAS and `langchain-community`. Upgrade:
```bash
pip install -U ragas langchain-openai langchain-community
```
If you must stay on the current RAGAS, pin `langchain-community<0.4`.

### `Unsupported value: 'temperature' does not support 0.01 ... Only the default (1)`
Reasoning-style Azure deployments only accept temperature 1, but RAGAS sets its own. `evaluate_rag.py` handles this with the `FixedTemperatureAzureChat` class, which removes `temperature` from the request payload. If the error persists, run `pip install -U langchain-openai`. Remove the class and the `temperature=1` line if you use gpt-4o or gpt-4.1.

### `DeprecationHelper.__init__() takes 3 positional arguments but 4 were given`
Caused by subclassing `LangchainLLMWrapper`, which newer RAGAS wraps in a deprecation helper. Use the plain `LangchainLLMWrapper(llm)` and apply the temperature fix on the LangChain model instead (as the script does).

### Deprecation warning for `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper`
Harmless. RAGAS plans to move to newer wrapper classes, but the metrics still run.

### `LLM returned 1 generations instead of requested 3`
Comes from Answer Relevancy, which asks for 3 generations per answer by default. Your deployment returns 1, and RAGAS continues with it. To make this consistent and silence the warning:
```python
ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb, strictness=1)
```

### `Questions without 'reference' answers`
The questions file was created by an older generator. Re-run `python evaluation/generate_questions.py`.

### Fewer than 50 questions generated
Check the console for per-document errors. Increase `max_completion_tokens` in `generate_questions.py` if responses come back empty, which can happen with reasoning models.

### `File not found: embeddings.npy` / index errors
Build the hybrid index first: `python retrieval/hybrid/build_index_hybrid.py`.

### Many `Exception raised in Job[...]` lines, or `NaN` scores
Usually rate limiting or timeouts. Lower `max_workers` (for example to 2) or raise `timeout` in `RunConfig`, then rerun.

### Azure OpenAI connection failed
Verify the `.env` values: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_CHAT_DEPLOYMENT`, `AZURE_EMBEDDING_DEPLOYMENT`.

## File layout

```
evaluation/
  generate_questions.py            # question + reference generation
  evaluate_rag.py                  # RAGAS evaluation
  evaluation_questions.json        # generated (step 2)
  ragas_results_<timestamp>.json   # generated (step 3)
  ragas_report_<timestamp>.csv     # generated (step 3)
retrieval/
  common/base.py                   # shared retrieval and index code
  hybrid/build_index_hybrid.py     # builds the hybrid index
  hybrid/ask_hybrid.py             # interactive Q&A
documents/                         # source PDFs
```