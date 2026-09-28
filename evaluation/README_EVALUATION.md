# RAG Evaluation Guide

This folder contains the scripts used to measure the quality of the insurance RAG assistant with [RAGAS](https://docs.ragas.io) metrics, and to compare retrieval configurations (dense vs hybrid, with and without reranking) on the same test set.

## Overview

The evaluation has two stages:

1. **`generate_questions.py`** builds a 50-question test set from the PDFs in `documents/`. Each question comes with a ground-truth (reference) answer.
2. **`evaluate_rag.py`** runs the real retrieval and answer-generation pipeline for one chosen configuration on every question, then scores the results with RAGAS.

```
documents/*.pdf
      |
      v
generate_questions.py  --->  evaluation/evaluation_questions.json
                                   (question, reference, source_document)
                                               |
rag_data/dense or rag_data/hybrid              |
(built by build_index_*.py)                    |
      |                                        v
      +------------------------------>  evaluate_rag.py --mode ... --rerank/--no-rerank
                                               |
                                               v
                      evaluation/results/<mode>_<rerankon|rerankoff>/
                          ragas_results_<...>.json
                          ragas_report_<...>.csv
```

## Metrics

### RAGAS metrics (scored by an LLM judge)

| Metric | Range | What it measures | A low score suggests |
|---|---|---|---|
| **Context precision** | 0-1 | Are the useful chunks ranked near the top of the retrieved list? | Noisy or poorly ranked retrieval |
| **Context recall** | 0-1 | Does the retrieved context contain everything needed for the reference answer? | Relevant information is not being retrieved |
| **Faithfulness** | 0-1 | Is every claim in the answer supported by the retrieved context? | Hallucination or use of outside knowledge |
| **Answer relevancy** | 0-1 | Does the answer address the question that was asked? | Off-topic, incomplete, or hedging answers |

Context precision and context recall depend only on retrieval. The chat model under test does not affect them. Faithfulness and answer relevancy depend on the generation step.

### Deterministic retrieval checks (no LLM)

| Metric | What it measures |
|---|---|
| **Source hit** | 1 if the expected `source_document` appears anywhere in the retrieved chunks, otherwise 0 |
| **Source precision** | Fraction of retrieved chunks that come from the expected document |

Source precision is naturally low for questions that draw on several documents, and when retrieval returns several chunks from other documents. Read it alongside source hit.

## Prerequisites

- Dependencies installed from `requirements.txt` (includes `ragas`, `langchain-openai`, `datasets`, `sentence-transformers`).
- `.env` populated with the Azure OpenAI values.
- The index for each mode you want to evaluate:
  ```bash
  python retrieval/dense/build_index_dense.py
  python retrieval/hybrid/build_index_hybrid.py
  ```
- Reranked runs need `sentence-transformers`, and the BGE model downloads on first use.

## Step 1: Generate the question set

```bash
python evaluation/generate_questions.py
```

What it does:

1. Reads every page of each PDF in `documents/` (top-level folder only), capped at 12,000 characters per document.
2. Splits the 50-question target across the documents as evenly as possible.
3. Asks the chat model for question and answer pairs per document. Questions must be answerable from that document alone, specific (amounts, dates, conditions, exclusions, claims steps), and a mix of single-fact and multi-part. Reference answers must use only facts stated in the document.
4. Runs up to 3 top-up rounds if any generation fails or returns too few items.
5. Saves everything to `evaluation/evaluation_questions.json`.

Settings (`EVAL_TARGET_QUESTIONS`, `EVAL_MAX_DOC_CHARS`, `EVAL_TOPUP_ROUNDS`) are in `config.py`.

**Review the reference answers.** They are written by an LLM. Read 5 to 10 against the source PDFs and correct any that are wrong, because context recall is only as trustworthy as the references. Generate the set once and reuse it for every run so results stay comparable.

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
      "question": "What is the Total Standard Excess payable on a claim?",
      "reference": "The Total Standard Excess is ...",
      "source_document": "P01_schedule_2024.pdf"
    }
  ]
}
```

## Step 2: Run an evaluation

Each run evaluates one configuration.

```bash
python evaluation/evaluate_rag.py --mode dense  --no-rerank
python evaluation/evaluate_rag.py --mode dense  --rerank
python evaluation/evaluate_rag.py --mode hybrid --no-rerank
python evaluation/evaluate_rag.py --mode hybrid --rerank
```

### Command-line options

| Option | Meaning |
|---|---|
| `--mode dense` / `--mode hybrid` | Retrieval mode. Defaults to `DEFAULT_RETRIEVAL_MODE` in `config.py` |
| `--rerank` / `--no-rerank` | Turn BGE reranking on or off. Defaults to `RERANK_ENABLED` in `config.py` |
| `--chat-model <deployment>` | Override the chat deployment used to generate the answers being evaluated. Defaults to `AZURE_CHAT_DEPLOYMENT` |
| `--top-k <n>` | Override the final number of chunks. Defaults to `TOP_K` (or `RERANK_TOP_K` when reranking) |

### What each run does

1. Loads the 50 questions and the index for the chosen mode.
2. For each question: embeds it, retrieves chunks (over-fetching a candidate pool and reranking when reranking is on), and generates an answer from the chunks.
3. Records the deterministic checks (`source_hit`, `source_precision`).
4. Builds a RAGAS dataset of the question, retrieved chunks, generated answer, and reference, and scores all four RAGAS metrics.
5. Writes a JSON file and a CSV file, and prints a summary of the averages.

**Runtime and cost:** 50 questions x 4 metrics is 200 RAGAS scoring jobs, each an LLM call, so a run typically takes 25 to 30 minutes. Reranked runs also score the candidate pool locally with the cross-encoder. Lower `max_workers` in `RunConfig` if you hit rate limits.

## Output folders

Each configuration writes to its own folder, so results are never mixed up:

```
evaluation/results/
  dense_rerankoff/   ragas_results_dense_rerankoff_<chat>_<timestamp>.json
                     ragas_report_dense_rerankoff_<chat>_<timestamp>.csv
  dense_rerankon/    ...
  hybrid_rerankoff/  ...
  hybrid_rerankon/   ...
```

The folder name encodes retrieval mode and reranking state. The file name also includes a short tag for the chat model and a timestamp, so runs that differ only in chat model land in the same folder but stay distinguishable.

### `ragas_results_*.json` structure

```json
{
  "metadata": {
    "evaluation_date": "...",
    "total_questions": 50,
    "retrieval_mode": "hybrid",
    "rerank_enabled": true,
    "rerank_model": "BAAI/bge-reranker-base",
    "chat_model": "...",
    "ragas_judge_model": "...",
    "embedding_model": "...",
    "top_k": 5,
    "index_path": "...",
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
      "source_precision": 0.4,
      "context_precision": 0.0,
      "context_recall": 0.0,
      "faithfulness": 0.0,
      "answer_relevancy": 0.0
    }
  ]
}
```

The metadata block records every setting of the run, so a file identifies itself even if it is renamed or moved. The `summary` values are plain averages over all questions. The CSV has one row per question with the scores, which is convenient for sorting in a spreadsheet.

### Example summary output

From a hybrid + rerank run:

```
Mode: hybrid | Rerank: True | Chat model: <deployment>
Questions evaluated: 50
context_precision    0.895
context_recall       0.958
faithfulness         0.995
answer_relevancy     0.674
source_hit           0.98
source_precision     0.324
```

Answer relevancy is noticeably lower than the other metrics here. Sort the CSV by that column before drawing conclusions: a few very low outliers pull an average down very differently from a consistently mediocre score.

## Comparing configurations

Optimise one variable at a time and carry the winner forward, instead of running every combination.

**Stage 1: retrieval (4 runs).** Hold the embedding model and chat model fixed and run the four dense/hybrid x rerank on/off combinations. Choose the winner mainly on context precision and context recall, since these are the metrics that differ between retrieval configurations.

| Run | Question it answers |
|---|---|
| dense, no rerank | Baseline |
| dense, rerank | Does reranking help dense retrieval on its own? |
| hybrid, no rerank | Does adding BM25 help over dense alone? |
| hybrid, rerank | Does reranking still add value once hybrid retrieval has improved the candidates? |

**Stage 2: embedding model (1 run).** Rebuild the index of the winning mode with a different embedding deployment (for example `text-embedding-3-large`) and rerun. Compare context precision and recall with the Stage 1 winner. Rebuilding is required, because embeddings from different models are not comparable.

**Stage 3: chat model (1 to 2 runs).** Fix the winning retrieval and embedding setup and vary only `--chat-model`. Compare faithfulness and answer relevancy. Context precision and recall will not change, because the chat model does not affect retrieval.

Keep the same `evaluation_questions.json` throughout so every score is comparable. Because LLM-judged scores vary slightly between runs, compare averages, not single questions, and treat very small differences with caution.

## Interpreting results

| Symptom | Likely cause | What to try |
|---|---|---|
| Low source hit or context recall | The right content is not being retrieved | Adjust chunking, increase `TOP_K` or the rerank candidate pool, tune hybrid weights, try a stronger embedding model |
| Low context precision | Relevant chunks are retrieved but ranked low, or there is noise | Enable reranking, reduce `TOP_K` |
| Low faithfulness | The model adds information beyond the context | Tighten the answer prompt |
| Low answer relevancy | Answers are off-topic, hedged, or incomplete | Check the prompt, and check whether retrieval supplies what the question needs |

Multi-part questions naturally score lower on recall because every part must be covered by the retrieved chunks. Questions that need several documents will show low recall with top-5 retrieval, which reflects a genuine retrieval limitation.

## Design notes

- **The RAGAS judge model is fixed.** RAGAS always scores using the default `AZURE_CHAT_DEPLOYMENT`, even when `--chat-model` changes the model generating the answers. This keeps the scoring method constant across comparison runs, so score differences reflect answer quality and not judge variation.
- **Reference answers are LLM-generated** from the same document text and model family used elsewhere in the project. They can contain errors, and they may favour phrasing the same model family would produce. Spot-check them.
- **The evaluation prompt differs from the production prompt.** `evaluate_rag.py` uses its own, simpler answer-generation prompt, while the assistant (`base.py`) has a longer prompt with citation and refusal rules. Scores therefore describe the evaluation prompt's behaviour. If you want evaluation to reflect production behaviour exactly, make the two prompts match, or have `evaluate_rag.py` call `generate_answer()` from `base.py`.
- **The test set covers answerable questions only.** It does not test whether the assistant correctly refuses when no matching document exists. That needs a separate set of deliberately unanswerable questions.
- **Temperature workaround.** Reasoning-style Azure deployments only accept the default temperature, but RAGAS sets its own. `evaluate_rag.py` includes a `FixedTemperatureAzureChat` class that strips the temperature from the request. Remove it and the `temperature=1` line if you use a deployment that accepts custom temperatures.
- **`generate_questions.py` reads only top-level PDFs** in `documents/`, while the index builders search subfolders too. Keep PDFs in the top-level folder, or the two will cover different files.

## Troubleshooting

| Error or symptom | Cause and fix |
|---|---|
| `No module named 'langchain_community.chat_models.vertexai'` | Version mismatch between RAGAS and `langchain-community`. Run `pip install -U ragas langchain-openai langchain-community`, or pin `langchain-community<0.4` |
| `Unsupported value: 'temperature' does not support 0.01 ... Only the default (1)` | Reasoning deployment rejecting RAGAS's temperature. The `FixedTemperatureAzureChat` class handles this. If it persists, run `pip install -U langchain-openai` |
| `DeprecationHelper.__init__() takes 3 positional arguments but 4 were given` | Caused by subclassing `LangchainLLMWrapper`. Use the plain wrapper and apply the temperature fix on the LangChain model, as the script does |
| Deprecation warning for `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper` | Harmless. The metrics still run |
| `LLM returned 1 generations instead of requested 3` | Answer relevancy asks for 3 generations and the deployment returns 1. Set `strictness=1` on `ResponseRelevancy` to make this consistent |
| `Questions without 'reference' answers` | The questions file came from an older generator. Re-run `generate_questions.py` |
| Fewer than 50 questions generated | Check the console for per-document errors, and raise `max_completion_tokens` in `generate_questions.py` if responses come back empty |
| `File not found: embeddings.npy` | Build the index for the mode you are evaluating |
| Many `Exception raised in Job[...]` lines, or `NaN` scores | Usually rate limiting or timeouts. Lower `max_workers` (for example to 2) or raise `timeout` in `RunConfig`, then rerun |
| Azure connection failed | Check `.env` values |

The scripts target the RAGAS 0.2/0.3 API (`EvaluationDataset`, `LLMContextPrecisionWithReference`). Once you have a working install, pin the exact versions from `pip show ragas langchain-openai` in `requirements.txt`.