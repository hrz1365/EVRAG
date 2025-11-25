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
git clone https://github.com/hrz1365/EVRAG.git
cd EVRAG
```
### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # on macOS/Linux
venv\Scripts\activate         # on Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Set your environment variable for Gemini 
```bash
export GEMINI_API_KEY="your_api_key_here"     # macOS/Linux
setx GEMINI_API_KEY "your_api_key_here"       # Windows
```
## Usage
### 1. Ask questions interactively
```bash
python main.py --mode query --query "What type of facility has power demand equal to around 5 MW?"
```
Answer:
an outdoor professional sports stadium

### 2. Run full evaluation
```bash
python main.py --mode 'validate' --pdf './data/' --query 'What type of facility has power demand euqal to around 40 MW?' --validation './data/validation_set.xlsx' --report './data/validation_results.xlsx'
```
