import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

client = OpenAI(
    api_key=api_key,
    base_url=endpoint.rstrip("/") + "/openai/v1/",
)

print("Testing embedding deployment...")
print(f"Deployment configured: {deployment}")

response = client.embeddings.create(
    model=deployment,
    input="Sudden and accidental damage from a collision.",
)

embedding = response.data[0].embedding

print("Embedding call succeeded.")
print(f"Vector dimensions: {len(embedding)}")
print(f"First five values: {embedding[:5]}")