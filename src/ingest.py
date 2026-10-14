# --------------------------------------------------
# Import Libraries
# --------------------------------------------------

import os

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_openai import (
    AzureOpenAIEmbeddings
)

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# Load Enterprise Support Datasets
# --------------------------------------------------

csv_loader = DirectoryLoader(
    "./data",
    glob="*.csv",
    loader_cls=CSVLoader,
    loader_kwargs={
        "csv_args": {
            "delimiter": ","
        }
    }
)

enterprise_docs = csv_loader.load()

print(
    f"\nLoaded {len(enterprise_docs)} enterprise documents"
)

# --------------------------------------------------
# Preview Documents
# --------------------------------------------------

for doc in enterprise_docs[ : 3]:
    print("\n--------------------")
    print(doc.page_content[:200])
    

# --------------------------------------------------
# Split Documents
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        " ",
        ""
    ]
)

enterprise_chunks = text_splitter.split_documents(
    enterprise_docs
)

print(
    f"\nCreated {len(enterprise_chunks)} chunks"
)

# --------------------------------------------------
# Create Embeddings
# --------------------------------------------------

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv(
        "AZURE_OPENAI_EMBEDDING_ENDPOINT"
    ),
    api_key=os.getenv(
        "AZURE_OPENAI_EMBEDDING_API_KEY"
    ),
    openai_api_version=os.getenv(
        "OPENAI_API_VERSION"
    ),
    azure_deployment=os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
)

# --------------------------------------------------
# Create FAISS Index
# --------------------------------------------------

BATCH_SIZE = 64

enterprise_vectorstore = None

for start in range(
    0,
    len(enterprise_chunks),
    BATCH_SIZE
):

    batch = enterprise_chunks[
        start:start + BATCH_SIZE
    ]

    try:

        if enterprise_vectorstore is None:

            enterprise_vectorstore = (
                FAISS.from_documents(
                    documents=batch,
                    embedding=embeddings
                )
            )

        else:

            enterprise_vectorstore.add_documents(
                batch
            )

        print(
            f"Processed "
            f"{min(start + BATCH_SIZE, len(enterprise_chunks))}"
            f"/{len(enterprise_chunks)} chunks"
        )

    except Exception as e:

        print(
            f"\nError while indexing batch: {e}"
        )

        raise

# --------------------------------------------------
# Save Index
# --------------------------------------------------

if enterprise_vectorstore is not None:

    enterprise_vectorstore.save_local(
        "faiss_index"
    )

    print(
        "\n✅ Batch ingestion completed successfully!"
    )

    print(
        f"Total vectors indexed: "
        f"{enterprise_vectorstore.index.ntotal}"
    )

else:

    print("\n❌ No documents were indexed.")