# EVRAG: Retrieval-Augmented Generation (RAG) with Automated Evaluation using DeepEval
End-to-end RAG pipeline with document ingestion, retrieval, LLM answer generation, and quantitative evaluation (Context Recall, Answer Relevancy, Faithfulness, Precision).

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system that answers domain-specific questions about EV charging from relevant PDF documents (e.g., reports).
It includes an evaluation pipeline powered by DeepEval to measure:
* Answer Relevancy — how semantically correct the model’s answer is,
* Faithfulness — whether the answer is grounded in the retrieved context (no hallucinations),
* Contextual Precision & Recall — how good the retriever is at finding relevant chunks.

The project is modular, reproducible, and structured for portfolio presentation. 

## Architecture
<p align="center">
  <img src="docs/Architecture.drawio.svg" width="200" />
</p>

## Project Structure
<p align="center">
  <img src="docs/project_structure.drawio.svg" width="200" />
</p>

## Features
| **Component**         | **Description** |
|------------------------|------------------|
| **PDFLoader**          | Extracts and preprocesses text from PDFs. |
| **TextChunker**        | Splits text into overlapping semantic chunks for retrieval. |
| **VectorStore (FAISS)**| Stores embeddings for fast semantic similarity search. |
| **LLMEngine**          | Generates final answers using a Google Gemini model. |
| **DeepEval Integration** | Provides automated evaluation metrics: Relevancy, Faithfulness, Precision, Recall. |
| **Validation Reports** | Produces `evaluation_results.xlsx` with all metric scores. |
| **CLI Interface**      | Allows users to run inference or validation using command-line arguments. |

## Installation
### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/rag-evaluation-pipeline.git
cd rag-evaluation-pipeline
