from langchain_groq import ChatGroq
import pandas as pd
from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from pydantic import BaseModel
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
app=FastAPI()
#embeddings
embeddings=HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
#load vector database
vector_store=FAISS.load_local('faiss3_index',
                              embeddings,
                              allow_dangerous_deserialization=True)
#retrieve docs
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#re ranking
# reranker model
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# llm
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# request body
class Query(BaseModel):
    question: str

# ask route
@app.post("/")
def ask_question(query: Query):

    # retrieve docs
    docs = retriever.invoke(query.question)

    # prepare pairs
    pairs = [
        (query.question, doc.page_content)
        for doc in docs
    ]

    # rerank scores
    scores = reranker.predict(pairs)

    # combine docs + scores
    ranked_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    # top 3 reranked docs
    top_docs = [doc for score, doc in ranked_docs[:3]]

    # context
    context = "\n\n".join(
        [doc.page_content for doc in top_docs]
    )

    # final prompt
    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query.question}
    """

    # llm response
    response = llm.invoke(prompt)

    return {
        "question": query.question,
        "response": response.content
    }

