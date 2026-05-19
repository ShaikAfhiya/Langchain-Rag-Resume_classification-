from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv
from langchain_groq import ChatGroq

import json

# load env variables
load_dotenv()

# fastapi app
app = FastAPI()

# load documents
with open("documents.json", "r") as f:
    documents = json.load(f)

# groq llm
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# request schema
class QueryRequest(BaseModel):
    question: str

# api route
@app.post("/")
def ask_question(request: QueryRequest):

    question = request.question.lower()

    matched_docs = []

    # simple retrieval
    for doc in documents:

        text = doc["content"].lower()

        if any(word in text for word in question.split()):
            matched_docs.append(doc["content"])

    # top docs
    context = "\n\n".join(matched_docs[:2])

    # prompt
    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {request.question}
    """

    # llm response
    response = llm.invoke(prompt)

    return {
        "question": request.question,
        "answer": response.content
    }
