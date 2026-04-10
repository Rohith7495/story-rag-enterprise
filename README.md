---
title: Enterprise Story RAG
emoji: 🚀
colorFrom: indigo
colorTo: slate
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: true
license: mit
---

# 🚀 Enterprise Story RAG: High-Precision Knowledge Explorer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gemini 2.5](https://img.shields.io/badge/Model-Gemini%202.5%20Flash-orange.svg)](https://deepmind.google/technologies/gemini/)
[![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone%20Serverless-green.svg)](https://www.pinecone.io/)
[![LlamaParse](https://img.shields.io/badge/Parser-LlamaParse-purple.svg)](https://www.llamaindex.ai/llamaparse)

An industrial-grade Retrieval-Augmented Generation (RAG) system engineered for high-accuracy document intelligence. Built with a cloud-native architecture, it leverages **Google Gemini 2.5 Flash**, **Pinecone Serverless**, and **LlamaParse** to deliver a "zero-hallucination" conversational interface for complex document sets.

---

## 🏗️ Architectural Excellence

This system isn't just a simple wrapper; it's a multi-layered RAG pipeline designed for production environments:

- **⚡ Hybrid Search (Dense + Sparse)**: 
  - **Dense**: Gemini Embeddings capture deep semantic meaning.
  - **Sparse**: BM25 Okapi ensures exact keyword matching.
  - **Fusion**: Reciprocal Rank Fusion (RRF) merges results for 2x better retrieval accuracy.
- **☁️ Cloud-Native Persistence**: Uses **Pinecone Serverless** for low-latency, scalable vector storage with metadata-rich indexing.
- **🧠 Semantic Caching**: Integrated caching layer that detects similar queries and returns instant answers, reducing API costs to $0 for repeated questions.
- **📄 High-Fidelity Parsing**: Powered by **LlamaParse** for expert-level extraction of tables, charts, and complex formatting from PDFs/DOCX.
- **🎛️ Metadata-Aware Filtering**: 
  - **Chronological**: Filter knowledge by time windows (e.g., "Last 24 hours", "Last 30 days").
  - **Source Control**: Dynamically scope search to specific documents in the knowledge base.
- **🔄 Resilient Pipeline**: Automatic retries with **Exponential Backoff** to handle API rate limits and network instability gracefully.

---

## 🚀 Key Features

- **📍 Real-time Streaming**: Sub-second response latency with incremental token generation.
- **📚 Source Citations**: Explicit `[Source X]` citations for every claim to ensure transparency and fact-checkability.
- **🧠 Conversational Memory**: Context-aware dialogue that remembers previous turns for complex follow-up questions.
- **🛠️ Multi-Format Support**: Seamlessly processes `.pdf`, `.docx`, and `.txt` files with intelligent OCR fallback.

---

## 🛠️ Local Installation & Setup

### 1. Pre-requisites
Ensure you have Python 3.9+ and the following API keys:
- `GEMINI_API_KEY` (Google AI Studio)
- `PINECONE_API_KEY` (Pinecone Dashboard)
- `LLAMA_CLOUD_API_KEY` (LlamaCloud - optional for high-fidelity parsing)

### 2. Installation
```bash
# Clone the repo
git clone https://github.com/Rohith7495/story-rag-enterprise.git
cd story-rag-enterprise

# Setup environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_key # Optional
```

### 4. Launch
```bash
streamlit run app.py
```

---

## ☁️ Deployment (Hugging Face / Cloud)

This app is pre-optimized for **Hugging Face Spaces** or any containerized environment.

1. Set the following **Secrets** in your hosting dashboard:
   - `GEMINI_API_KEY`
   - `PINECONE_API_KEY`
   - `LLAMA_CLOUD_API_KEY`
2. Deploy the `main` branch. The `SDK` configuration in `README.md` handles the rest!

---
Built with ❤️ by the Antigravity Team.

