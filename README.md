# Custom RAG using LangChain and HuggingFace

This repository demonstrates how to set up a custom Retrieval-Augmented Generation (RAG) system using LangChain for PDF document retrieval and HuggingFace transformers for question-answering tasks.

## Introduction

This project combines LangChain, a library for building language-driven pipelines, with HuggingFace's transformers, a powerful tool for natural language processing tasks, to create a custom RAG system. The system retrieves relevant documents using a retrieval mechanism based on FAISS and then uses a language model to generate answers to queries based on these documents.

## Setup

### Installation

To run the RAG system locally, follow these steps:

1. Clone the repository:
```bash
   git clone https://github.com/GauravKParmar/custom-rag-langchain-huggingface.git
   cd custom-rag-langchain-huggingface
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
### Dependencies
- **LangChain**: Used for building language-driven pipelines.
- **HuggingFace Transformers**: Provides state-of-the-art models for natural language understanding and generation.
- **FAISS**: Efficient similarity search and clustering of dense vectors library, used for document retrieval.

### Usage
#### Running the Server
Start the Uvicorn server to host the RAG system:
```bash
   uvicorn api_app:app --reload --host localhost --port 8000
```

#### Using the API
Once the server is running, interact with the RAG system using HTTP POST requests:
- **Endpoint**: 'http://localhost:8000/custom_rag'
- **Request Body**: JSON object with a field "input" containing the query or question.

#### Run via terminal
1. Run api_app.py
   This command starts the FastAPI server.
   
   ```bash
   python .\custom_rag\api_app.py
   ```
2. Run client_app.py
   This command starts the Streamlit server.
   
   ```bash
   streamlit run .\custom_rag\client_app.py
   ```
