
from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# load environment variables
load_dotenv()

# fastapi app
app = FastAPI()

# embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# load faiss vector store
vector_store = FAISS.load_local(
    'faiss3_index',
    embeddings,
    allow_dangerous_deserialization=True
)

# retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# groq llm
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# request schema
class QueryRequest(BaseModel):
    question: str

# rag endpoint
@app.post("/")
def ask_question(request: QueryRequest):

    # retrieve docs
    docs = retriever.invoke(request.question)

    # combine context
    context = "\n\n".join([doc.page_content for doc in docs])

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

