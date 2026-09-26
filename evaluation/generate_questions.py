"""
Generate exactly 50 evaluation questions WITH ground-truth reference answers
from the PDFs in ./documents, for RAGAS evaluation.

Output: evaluation/evaluation_questions.json
Each item: {"question", "reference", "source_document"}
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# this file lives at <repo_root>/evaluation/generate_questions.py
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (
    DOCUMENTS_DIRECTORY,
    EVAL_MAX_DOC_CHARS,
    EVAL_TARGET_QUESTIONS,
    EVAL_TOPUP_ROUNDS,
    EVALUATION_DIRECTORY,
)

load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")

OUTPUT_FILE = EVALUATION_DIRECTORY / "evaluation_questions.json"

TARGET_QUESTIONS = EVAL_TARGET_QUESTIONS
MAX_DOC_CHARS = EVAL_MAX_DOC_CHARS      # all pages are read, then capped at this many characters
MAX_TOPUP_ROUNDS = EVAL_TOPUP_ROUNDS


def create_openai_client():
    base_url = AZURE_OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/"
    return OpenAI(api_key=AZURE_OPENAI_API_KEY, base_url=base_url)


def extract_pdf_text(pdf_path):
    """Extract text from ALL pages of a PDF."""
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def collect_document_content():
    documents = {}
    pdf_files = sorted(DOCUMENTS_DIRECTORY.glob("*.pdf"))
    print(f"Extracting content from {len(pdf_files)} documents...")
    for idx, pdf_path in enumerate(pdf_files, 1):
        content = extract_pdf_text(pdf_path)
        if content:
            documents[pdf_path.name] = content[:MAX_DOC_CHARS]
            print(f"  [{idx}/{len(pdf_files)}] {pdf_path.name} ({len(content)} chars)")
    return documents


def allocate_counts(num_docs, total):
    """Spread `total` questions over documents as evenly as possible."""
    base, extra = divmod(total, num_docs)
    return [base + (1 if i < extra else 0) for i in range(num_docs)]


def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text or "")
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {}
    return {}


def generate_qa_for_document(client, doc_name, doc_content, num_questions, existing_questions=None):
    """Generate questions + ground-truth answers for one document."""
    if num_questions <= 0:
        return []

    avoid = ""
    if existing_questions:
        avoid = "Do NOT repeat or paraphrase these existing questions:\n- " + "\n- ".join(existing_questions) + "\n\n"

    prompt = f"""You are building a test set for a RAG system over insurance documents.

From the document below, write exactly {num_questions} question(s) with ground-truth answers.

Requirements:
- Each question must be fully answerable from THIS document alone.
- Make each question specific (amounts, dates, names, conditions, definitions, exclusions,
  claims procedures). Do not refer to "this document" or "the text"; name the policy holder,
  vehicle registration or policy section where relevant so the question is self-contained.
- Vary the question types across factual lookups, conditions/exclusions, and multi-part questions.
- Each answer must be complete, precise, and use ONLY facts stated in the document
  (include exact figures, dates, names and section references). Never invent information.

{avoid}DOCUMENT ({doc_name}):
{doc_content}

Return ONLY valid JSON in this format:
{{"items": [{{"question": "...", "reference": "..."}}]}}"""

    try:
        response = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an expert in insurance policies who writes accurate, document-grounded test questions and answers."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=6000,  # generous: reasoning models spend tokens before answering
        )
        data = parse_json_response(response.choices[0].message.content)
        items = []
        for it in data.get("items", []):
            q, a = (it.get("question") or "").strip(), (it.get("reference") or "").strip()
            if q and a:
                items.append({"question": q, "reference": a, "source_document": doc_name})
        return items[:num_questions]
    except Exception as e:
        print(f"  Error generating for {doc_name}: {e}")
        return []


def main():
    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not DOCUMENTS_DIRECTORY.exists():
        print(f"Error: Documents directory not found at {DOCUMENTS_DIRECTORY}")
        return

    documents = collect_document_content()
    if not documents:
        print("No documents found or no text could be extracted.")
        return

    client = create_openai_client()
    doc_names = list(documents.keys())
    counts = allocate_counts(len(doc_names), TARGET_QUESTIONS)

    print(f"\nGenerating {TARGET_QUESTIONS} questions across {len(doc_names)} documents...\n")

    all_items = []
    for idx, (doc_name, n) in enumerate(zip(doc_names, counts), 1):
        print(f"[{idx}/{len(doc_names)}] {doc_name}: {n} question(s)")
        items = generate_qa_for_document(client, doc_name, documents[doc_name], n)
        all_items.extend(items)
        print(f"  got {len(items)}")

    # Top up if some generations failed or returned too few
    rounds = 0
    while len(all_items) < TARGET_QUESTIONS and rounds < MAX_TOPUP_ROUNDS:
        rounds += 1
        missing = TARGET_QUESTIONS - len(all_items)
        print(f"\nTop-up round {rounds}: need {missing} more...")
        for doc_name in doc_names:
            if missing <= 0:
                break
            existing = [i["question"] for i in all_items if i["source_document"] == doc_name]
            new = generate_qa_for_document(client, doc_name, documents[doc_name], 1, existing)
            all_items.extend(new)
            missing -= len(new)

    all_items = all_items[:TARGET_QUESTIONS]

    output = {
        "metadata": {
            "total_questions": len(all_items),
            "documents_used": len({i["source_document"] for i in all_items}),
            "generation_date": datetime.now().isoformat(),
            "evaluation_type": "ragas",
            "has_reference_answers": True,
        },
        "questions": all_items,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print(f"✓ Total questions: {len(all_items)} (target {TARGET_QUESTIONS})")
    print(f"✓ Documents covered: {output['metadata']['documents_used']}")
    print("Tip: spot-check a few 'reference' answers by hand before trusting recall scores.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nError: {type(error).__name__}")
        print(f"Details: {error}")
        raise SystemExit(1)