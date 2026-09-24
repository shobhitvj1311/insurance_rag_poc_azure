"""
RAGAS evaluation for the hybrid (BM25 + dense) insurance RAG pipeline.

Reads evaluation/evaluation_questions.json (question, reference, source_document),
runs your real retrieval + answer generation, then scores with RAGAS:

- context_precision : relevant chunks ranked high in the retrieved list
- context_recall    : retrieved context covers the reference answer
- faithfulness      : answer claims are supported by retrieved context
- answer_relevancy  : answer addresses the question

Plus deterministic retrieval checks: source_hit, source_precision.

pip install ragas langchain-openai datasets pandas numpy python-dotenv
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from retrieval.common.base import (
    CHAT_DEPLOYMENT,
    EMBEDDING_DEPLOYMENT,
    HYBRID_OUTPUT_DIRECTORY,
    create_openai_client,
    get_question_embedding,
    load_hybrid_index,
    retrieve_hybrid_chunks,
)

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

load_dotenv(override=True)

TOP_K = 5
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

EVALUATION_DIRECTORY = Path("evaluation")
QUESTIONS_FILE = EVALUATION_DIRECTORY / "evaluation_questions.json"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = EVALUATION_DIRECTORY / f"ragas_results_{TIMESTAMP}.json"
REPORT_FILE = EVALUATION_DIRECTORY / f"ragas_report_{TIMESTAMP}.csv"


def load_evaluation_questions(path=QUESTIONS_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f).get("questions", [])
    missing = [i + 1 for i, q in enumerate(questions) if not q.get("reference")]
    if missing:
        raise ValueError(
            f"Questions without 'reference' answers: {missing}. "
            "Re-run generate_questions.py (the new version writes references)."
        )
    return questions


def _norm_doc(name):
    return Path(str(name)).name.lower().strip()


def retrieve_context(client, question, embeddings, metadata, corpus_texts, top_k=TOP_K):
    try:
        q_emb = get_question_embedding(client, question, embedding_name=EMBEDDING_DEPLOYMENT)
        return retrieve_hybrid_chunks(
            q_emb, embeddings, metadata, corpus_texts, question, top_k=top_k
        )
    except Exception as e:
        print(f"  Error retrieving context: {e}")
        return []


def build_context_string(chunks):
    return "\n\n".join(
        f"[Source {c['rank']}: {c['document_name']}, page {c['page_number']}]\n{c['content']}"
        for c in chunks
    )


def generate_answer(client, question, context):
    if not context:
        return "No relevant context found to answer this question."

    system_prompt = """
You are an insurance policy and claims knowledge assistant.

Use only the document context supplied by the user to answer questions.

Rules:
1. Do not use general knowledge to fill information gaps.
2. Base your answer strictly on the provided context.
3. If the context is insufficient, state that clearly.
4. Be concise and professional.
""".strip()

    user_prompt = f"DOCUMENT CONTEXT:\n\n{context}\n\nQUESTION:\n\n{question}\n\nAnswer based strictly on the provided context:"

    try:
        response = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2000,
        )
        return response.choices[0].message.content or "Unable to generate answer."
    except Exception as e:
        print(f"  Error generating answer: {e}")
        return f"Error: {e}"


class FixedTemperatureAzureChat(AzureChatOpenAI):
    """Reasoning-style Azure deployments only accept the default temperature (1),
    but RAGAS sets its own tiny temperature (e.g. 0.01) by several routes.
    Remove temperature from the outgoing request so the model default is used."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("temperature", None)
        return payload

    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        kwargs.pop("temperature", None)
        return super().generate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        kwargs.pop("temperature", None)
        return await super().agenerate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)


def create_ragas_models():
    llm = FixedTemperatureAzureChat(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=AZURE_API_VERSION,
        azure_deployment=CHAT_DEPLOYMENT,
        temperature=1,  # reasoning-style deployments only accept 1; remove for gpt-4o/4.1
    )
    emb = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=AZURE_API_VERSION,
        azure_deployment=EMBEDDING_DEPLOYMENT,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def find_metric_column(df, *prefixes):
    for prefix in prefixes:
        for col in df.columns:
            if col.startswith(prefix):
                return col
    return None


def evaluate_rag_pipeline():
    print("=" * 70)
    print("RAGAS EVALUATION PIPELINE")
    print("=" * 70)

    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)

    questions = load_evaluation_questions()
    print(f"Loaded {len(questions)} questions with reference answers.")

    embeddings, metadata, corpus_texts = load_hybrid_index(HYBRID_OUTPUT_DIRECTORY)
    print(f"Loaded hybrid index with {len(metadata)} chunks.")

    client = create_openai_client()

    samples, extras = [], []
    print(f"\nRunning retrieval + generation (top_k={TOP_K})...")
    print("-" * 70)

    for idx, item in enumerate(questions, 1):
        question = item["question"]
        reference = item["reference"]
        source_doc = item.get("source_document", "Unknown")
        print(f"[{idx}/{len(questions)}] {question[:70]}...")

        chunks = retrieve_context(client, question, embeddings, metadata, corpus_texts)
        answer = generate_answer(client, question, build_context_string(chunks))

        retrieved_docs = [_norm_doc(c["document_name"]) for c in chunks]
        expected = _norm_doc(source_doc)
        source_hit = float(expected in retrieved_docs)
        source_precision = (
            sum(d == expected for d in retrieved_docs) / len(retrieved_docs)
            if retrieved_docs else 0.0
        )

        samples.append({
            "user_input": question,
            "retrieved_contexts": [c["content"] for c in chunks] or [""],
            "response": answer,
            "reference": reference,
        })
        extras.append({
            "source_document": source_doc,
            "retrieved_documents": [
                {"rank": c["rank"], "document": c["document_name"],
                 "page": c["page_number"], "score": c["score"]}
                for c in chunks
            ],
            "source_hit": source_hit,
            "source_precision": round(source_precision, 3),
        })

    print("\n" + "=" * 70)
    print("SCORING WITH RAGAS (many LLM calls, this can take a while)")
    print("=" * 70)

    ragas_llm, ragas_emb = create_ragas_models()
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[
            LLMContextPrecisionWithReference(llm=ragas_llm),
            LLMContextRecall(llm=ragas_llm),
            Faithfulness(llm=ragas_llm),
            ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ],
        run_config=RunConfig(timeout=240, max_workers=4, max_retries=5),
        show_progress=True,
    )
    df = result.to_pandas()

    col_map = {
        "context_precision": find_metric_column(df, "llm_context_precision", "context_precision"),
        "context_recall": find_metric_column(df, "context_recall"),
        "faithfulness": find_metric_column(df, "faithfulness"),
        "answer_relevancy": find_metric_column(df, "answer_relevancy", "response_relevancy"),
    }

    results = []
    for i, (sample, extra) in enumerate(zip(samples, extras)):
        row = {
            "question_id": i + 1,
            "question": sample["user_input"],
            "reference": sample["reference"],
            "answer": sample["response"],
            **extra,
        }
        for name, col in col_map.items():
            val = df.iloc[i][col] if col else None
            row[name] = None if val is None or pd.isna(val) else round(float(val), 3)
        results.append(row)

    metric_names = list(col_map.keys()) + ["source_hit", "source_precision"]
    summary = {}
    for m in metric_names:
        vals = [r[m] for r in results if r.get(m) is not None]
        summary[f"avg_{m}"] = round(float(np.mean(vals)), 3) if vals else None

    output = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "total_questions": len(results),
            "chat_model": CHAT_DEPLOYMENT,
            "embedding_model": EMBEDDING_DEPLOYMENT,
            "retrieval_method": "hybrid",
            "top_k": TOP_K,
            "index_path": str(HYBRID_OUTPUT_DIRECTORY),
            "framework": "ragas",
        },
        "summary": summary,
        "results": results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    pd.DataFrame([
        {
            "Question ID": r["question_id"],
            "Question": r["question"],
            "Source Document": r["source_document"],
            "Source Hit": r["source_hit"],
            "Source Precision": r["source_precision"],
            "Context Precision": r["context_precision"],
            "Context Recall": r["context_recall"],
            "Faithfulness": r["faithfulness"],
            "Answer Relevancy": r["answer_relevancy"],
        }
        for r in results
    ]).to_csv(REPORT_FILE, index=False, encoding="utf-8")

    print(f"\n✓ Results saved to: {RESULTS_FILE}")
    print(f"✓ Report saved to:  {REPORT_FILE}")
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Questions evaluated: {len(results)}")
    for m in metric_names:
        v = summary[f"avg_{m}"]
        print(f"{m:<20} {v if v is not None else 'n/a'}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        evaluate_rag_pipeline()
    except KeyboardInterrupt:
        print("\nEvaluation stopped.")
    except Exception as error:
        print(f"\nError: {type(error).__name__}")
        print(f"Details: {error}")
        raise SystemExit(1)