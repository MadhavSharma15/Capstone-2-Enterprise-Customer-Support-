import os
from typing import TypedDict

from dotenv import load_dotenv

from langchain.agents import Tool

from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from src.rag_pipeline import load_pipeline, ask_question

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# LLM
# --------------------------------------------------

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

# --------------------------------------------------
# Load RAG Pipeline
# --------------------------------------------------

rag_chain = load_pipeline()

# --------------------------------------------------
# Enterprise RAG Tool
# --------------------------------------------------

def enterprise_rag_tool_func(query: str) -> str:
    
    try:

        result = ask_question(
            rag_chain,
            query
        )

        answer = result.get(
            "answer",
            "No relevant information found."
        )

        return answer

    except Exception as e:

        import traceback

        traceback.print_exc()

        return f"RAG Error: {str(e)}"


enterprise_rag_tool = Tool(
    name="Enterprise Support RAG",
    func=enterprise_rag_tool_func,
    description="""
Retrieve enterprise support information from the internal
knowledge base.

Topics include:

- Password resets
- VPN access
- Email issues
- Laptop requests
- Software installations
- Account unlock requests
- HR policies
- Leave management
- Remote work policies
- Security policies

Always use this tool when answering
enterprise support questions.
"""
)

# --------------------------------------------------
# State
# --------------------------------------------------

class AgentState(TypedDict, total=False):

    question: str

    context: str

    retrieval_output: str

    issue_category: str

    priority_analysis: str

    final_response: str


# --------------------------------------------------
# Retrieval Agent
# --------------------------------------------------

retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Knowledge Retrieval Agent.

Your job is to organize enterprise support
knowledge retrieved from the RAG system.

Do not invent information.

Return:

Retrieved Information:
[content]

Relevant Policies:
[policies]

Missing Information:
[gaps]
"""
        ),
        (
            "human",
            """
User Question:
{question}

Retrieved Content:
{rag_result}
"""
        )
    ]
)


def retrieval_agent(state: AgentState):

    question = state["question"]

    rag_result = enterprise_rag_tool.invoke(question)

    messages = retrieval_prompt.format_messages(
        question=question,
        rag_result=str(rag_result)
    )

    response = llm.invoke(messages)

    return {

        "context": str(rag_result),

        "retrieval_output": response.content

    }


# --------------------------------------------------
# Classification Agent
# --------------------------------------------------

classification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an Issue Classification Agent.

Classify the issue into ONE category.

Categories:

- Password
- VPN
- Email
- Hardware
- Software
- Network
- HR
- Leave
- Security
- Account Access
- General Support

Return:

Issue Category:
[category]

Reason:
[explanation]
"""
        ),
        (
            "human",
            """
Question:
{question}

Retrieved Context:
{context}
"""
        )
    ]
)


def classification_agent(state: AgentState):

    messages = classification_prompt.format_messages(
        question=state["question"],
        context=state.get("context", "")
    )

    response = llm.invoke(messages)

    return {

        "issue_category": response.content

    }


# --------------------------------------------------
# Priority Agent
# --------------------------------------------------

priority_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Priority Assessment Agent.

Determine urgency.

Levels:

LOW
MEDIUM
HIGH
CRITICAL

Return:

Priority:
[level]

Escalation Required:
[Yes/No]

Reason:
[explanation]
"""
        ),
        (
            "human",
            """
User Query:
{question}

Issue Classification:
{issue_category}
"""
        )
    ]
)


def priority_agent(state: AgentState):

    messages = priority_prompt.format_messages(
        question=state["question"],
        issue_category=state.get(
            "issue_category",
            ""
        )
    )

    response = llm.invoke(messages)

    return {

        "priority_analysis":
        response.content

    }


# --------------------------------------------------
# Final Response Agent
# --------------------------------------------------

final_response_prompt = ChatPromptTemplate.from_template(
"""
You are the Final Enterprise Customer Support Agent.

User Question:
{question}

Retrieved Information:
{context}

Issue Category:
{issue_category}

Priority Analysis:
{priority_analysis}

Instructions:

1. Provide a professional answer.
2. Use retrieved information only.
3. Provide resolution guidance.
4. Mention escalation status.
5. Keep response concise.
6. Do not invent company policies.

Output Format:

Issue Summary

Recommended Resolution

Escalation Status

References
"""
)


def final_response_agent(state: AgentState):

    messages = final_response_prompt.format_messages(

        question=state["question"],

        context=state.get(
            "context",
            ""
        ),

        issue_category=state.get(
            "issue_category",
            ""
        ),

        priority_analysis=state.get(
            "priority_analysis",
            ""
        )
    )

    response = llm.invoke(messages)

    return {

        "final_response":
        response.content

    }


# --------------------------------------------------
# LangGraph Workflow
# --------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node(
    "retrieval",
    retrieval_agent
)

workflow.add_node(
    "classification",
    classification_agent
)

workflow.add_node(
    "priority",
    priority_agent
)
# --------------------------------------------------
# Add Final Response Node
# --------------------------------------------------

workflow.add_node(
    "final_response",
    final_response_agent
)

# --------------------------------------------------
# Graph Flow
# --------------------------------------------------

workflow.add_edge(
    START,
    "retrieval"
)

workflow.add_edge(
    "retrieval",
    "classification"
)

workflow.add_edge(
    "classification",
    "priority"
)

workflow.add_edge(
    "priority",
    "final_response"
)

workflow.add_edge(
    "final_response",
    END
)

# --------------------------------------------------
# Checkpointer
# --------------------------------------------------

checkpointer = InMemorySaver()

# --------------------------------------------------
# Compile Graph
# --------------------------------------------------

enterprise_support_graph = workflow.compile(
    checkpointer=checkpointer
)

print(
    "Enterprise Customer Support Assistant initialized successfully."
)

