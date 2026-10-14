import os
import time

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from langchain.chains import ConversationalRetrievalChain

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)

from langchain_cohere import CohereRerank

from langchain_openai import (
    AzureOpenAIEmbeddings,
    ChatOpenAI
)

load_dotenv()

# --------------------------------------------------
# Load Pipeline
# --------------------------------------------------

def load_pipeline():

    # Embeddings

    
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
    # Load FAISS

    if not os.path.exists("faiss_index"):

        raise FileNotFoundError(
        "faiss_index not found. "
        "Run ingest.py first."
    )

    faiss_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

    # LLM

    llm = ChatOpenAI(
    base_url=os.getenv(
        "AZURE_OPENAI_CHAT_ENDPOINT"
    ),
    api_key=os.getenv(
        "AZURE_OPENAI_CHAT_API_KEY"
    ),
    model=os.getenv(
        "AZURE_OPENAI_CHAT_DEPLOYMENT"
    )
)

    # Memory

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Semantic Retriever

    semantic_retriever = faiss_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8}
    )

    # BM25 Retriever

    documents = list(
        faiss_store.docstore._dict.values()
    )

    bm25_retriever = BM25Retriever.from_documents(
        documents=documents,
        k=8
    )

    # Ensemble Retriever

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            semantic_retriever,
            bm25_retriever
        ],
        weights=[0.6, 0.4]
    )

    # Cohere Reranker

#    reranker = CohereRerank(
#        cohere_api_key=os.getenv(
#            "COHERE_API_KEY"
#        ),
#        model="rerank-v3.5",
#        top_n=4
#    )

#    contextual_retriever = (
#        ContextualCompressionRetriever(
#            base_compressor=reranker,
#            base_retriever=ensemble_retriever
#      )
#   )

    # --------------------------------------------------
    # Enterprise Support Prompt
    # --------------------------------------------------

    prompt = PromptTemplate(
        input_variables=[
            "context",
            "chat_history",
            "question"
        ],

        template="""
You are an Enterprise Customer Support Assistant.

Your responsibility is to provide accurate and professional answers
using ONLY the information available in the retrieved context.

--------------------------------------------------

CONTEXT:

{context}

--------------------------------------------------

QUESTION:

{question}

--------------------------------------------------

CHAT HISTORY:

{chat_history}

--------------------------------------------------

INSTRUCTIONS:

1. Answer only from the provided context.

2. If the answer exists in the context,
   provide a clear and professional response.

3. Do not create policies,
   procedures, or resolutions that are not present
   in the retrieved information.

4. If sufficient information is unavailable,
   respond:

   "The available enterprise knowledge base does not contain enough information to answer this query."

5. When applicable, identify:

   - Issue Category
   - Recommended Resolution
   - Escalation Requirement
   - Relevant Policy

6. Use concise business language.

7. Refer to company policies whenever available.

8. Do not generate technical troubleshooting steps
   unless they exist in the retrieved context.

9. Mention source files used to formulate the answer.

--------------------------------------------------

FORMAT:

Answer:
[Provide direct answer]

Issue Category:
[IT Support / HR / Network / Security / Hardware /
Software / Policy]

Recommended Resolution:
[Resolution from retrieved context]

Escalation Required:
[Yes / No]

Relevant Policy:
[Policy name if available]

Sources:
enterprise_support_dataset.csv
customer_ticket_dataset.csv
company_policy_dataset.csv
"""
    )

    # --------------------------------------------------
    # Conversational RAG Chain
    # --------------------------------------------------

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt
        }
    )

    return chain


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

def ask_question(chain, question):

    start = time.time()

    result = chain.invoke(
        {
            "question": question
        }
    )

    latency = time.time() - start

    docs = result["source_documents"]

    retrieved_docs = [
        doc.page_content
        for doc in docs
    ]

    sources = [
        doc.metadata
        for doc in docs
    ]

    return {
        "answer": result["answer"],
        "retrieved_docs": retrieved_docs,
        "sources": sources,
        "latency": latency
    }