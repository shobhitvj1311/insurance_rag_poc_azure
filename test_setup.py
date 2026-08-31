import numpy
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from pypdf import PdfReader

print("All required Python packages imported successfully.")
print(f"NumPy version: {numpy.__version__}")
