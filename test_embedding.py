# test_embedding.py

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://azure-foundary-15.services.ai.azure.com",
    api_key="YOUR_KEY",
    openai_api_version="2024-02-01",
    azure_deployment="text-embedding-ada-002"
)

result = embeddings.embed_query("hello world")

print(len(result))
print("SUCCESS")