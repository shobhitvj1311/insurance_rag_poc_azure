# Insurance Policy RAG POC

A proof-of-concept Retrieval-Augmented Generation application for answering questions from insurance policy documents.

## Architecture

1. Insurance PDFs are stored in Azure Blob Storage.
2. `download_documents.py` downloads documents locally.
3. `build_index.py` extracts PDF text and creates overlapping chunks.
4. Azure OpenAI generates embeddings for the chunks.
5. Embeddings and metadata are stored locally.
6. `ask.py` retrieves relevant chunks using cosine similarity.
7. An Azure OpenAI chat model generates a grounded answer with citations.

## Azure Resources

- Microsoft Foundry project
- Azure OpenAI embedding model deployment
- Azure OpenAI chat model deployment
- Azure Storage account
- Private Blob container

Azure AI Search and App Service are not required for this POC.

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

```python
pip install -r requirements.txt
cp .env.example .env
Update .env with the correct Azure endpoint, deployment names, API key, and Storage connection string.

## Running the application

```python
python download_documents.py
python build_index.py
python ask.py
