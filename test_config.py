import os
from dotenv import load_dotenv

load_dotenv()

required_variables = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_EMBEDDING_DEPLOYMENT",
    "AZURE_CHAT_DEPLOYMENT",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_CONTAINER",
]

missing = [
    variable
    for variable in required_variables
    if not os.getenv(variable)
]

if missing:
    print("Missing environment variables:")
    for variable in missing:
        print(f"  - {variable}")
    raise SystemExit(1)

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "")
chat_deployment = os.getenv("AZURE_CHAT_DEPLOYMENT", "")
container = os.getenv("AZURE_STORAGE_CONTAINER", "")

print("Configuration loaded successfully.")
print(f"Model endpoint configured: {endpoint.startswith('https://')}")
print(f"Embedding deployment: {embedding_deployment}")
print(f"Chat deployment: {chat_deployment}")
print(f"Blob container: {container}")
print("Secrets were found but have not been displayed.")
