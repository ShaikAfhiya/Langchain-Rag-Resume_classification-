AI Resume Analyzer RAG System

A production-style Retrieval-Augmented Generation (RAG) project built using LangChain, FastAPI, FAISS, HuggingFace Embeddings, Re-ranking, and Ollama.

This project performs intelligent semantic search over resume data and answers user queries using Retrieval-Augmented Generation.

Features
Resume semantic search using FAISS
RAG pipeline using LangChain
FastAPI backend
HuggingFace sentence embeddings
Cross-encoder re-ranking
Local LLM inference using Ollama
REST API support
Resume/job skill retrieval
Context-aware response generation
Tech Stack
Python
FastAPI
LangChain
FAISS
HuggingFace Embeddings
Sentence Transformers
Ollama
Cross Encoder Re-ranking
Project Structure
project/
│
├── app.py
├── train.py
├── requirements.txt
├── index.faiss
├── index.pkl
└── README.md
Dataset

Resume dataset containing multiple job categories and professional resume data.

Dataset columns:

Resume_str
Category
ID
How It Works
User Query
   ↓
FAISS Retriever
   ↓
Top Retrieved Resume Chunks
   ↓
Cross Encoder Re-ranking
   ↓
Best Relevant Chunks
   ↓
LLM Response Generation
Installation

Clone repository:

git clone YOUR_GITHUB_REPO_LINK

Move into project folder:

cd YOUR_PROJECT_NAME

Install dependencies:

pip install -r requirements.txt
Run Ollama

Start Ollama model:

ollama run gemma:2b

You can also use:

tinyllama
llama3
mistral
Create Vector Database

Run training pipeline:

python train.py

This creates:

index.faiss
index.pkl
Run FastAPI Server
uvicorn app:app --reload
API Endpoint

Open Swagger UI:

http://127.0.0.1:8000/docs
Example API Request
{
  "question":"What skills are required for HR jobs?"
}
Example Response
{
  "response":"HR jobs require communication, recruitment, leadership..."
}
Key Concepts Used
Retrieval-Augmented Generation (RAG)
Semantic Search
Vector Databases
Embeddings
Re-ranking
Context Retrieval
LLM Inference
REST APIs
