from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import enterprise_support_graph

app = FastAPI(
    title="Enterprise Customer Support Assistant",
    version="1.0",
    description="Agentic RAG application for enterprise customer support."
)

class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default"


@app.get("/")
def home():
    return {
        "message":
        "Enterprise Customer Support Assistant Running"
    }


@app.post("/query")
def ask_question(request: QueryRequest):

    result = enterprise_support_graph.invoke(
        {
            "question": request.question,
            "context": "",
            "retrieval_output": "",
            "issue_category": "",
            "priority_analysis": "",
            "final_response": ""
        },
        config={
            "configurable": {
                "thread_id": request.thread_id
            }
        }
    )

    return {

        "question":
        request.question,

        "answer":
        result.get(
            "final_response",
            ""
        )
    }