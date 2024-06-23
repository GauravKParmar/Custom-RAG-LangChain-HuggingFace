# Custom RAG using LangChain and HuggingFace

This repository demonstrates how to set up a custom Retrieval-Augmented Generation (RAG) system using LangChain for PDF document retrieval and HuggingFace transformers for question-answering tasks.

## Introduction

This project combines LangChain, a library for building language-driven pipelines, with HuggingFace's transformers, a powerful tool for natural language processing tasks, to create a custom RAG system. The system retrieves relevant documents using a retrieval mechanism based on FAISS and then uses a language model to generate answers to queries based on these documents.

## Setup

### Installation

To run the RAG system locally, follow these steps:

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/custom-rag-langchain-huggingface.git
   cd custom-rag-langchain-huggingface
