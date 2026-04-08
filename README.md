---
title: Enterprise Story RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
license: mit
---

# 📚 Enterprise Story RAG Chat

An advanced Retrieval-Augmented Generation (RAG) system built with **Google Gemini**, **ChromaDB**, and **BM25 Hybrid Search**. 

This application allows you to upload documents (PDF, DOCX, TXT) and have a detailed, context-aware conversation with them using professional-grade retrieval techniques.

## 🚀 Key Features

- **⚡ Hybrid Search Architecture**: Combines Dense Vector Search (Gemini Embeddings) with Sparse Keyword Search (BM25) using **Reciprocal Rank Fusion (RRF)** for maximum retrieval accuracy.
- **💾 Persistent local storage**: Uses ChromaDB to save your knowledge base so you don't have to re-upload documents every session.
- **📄 Multi-Format Support**: Effortlessly parse text from `.pdf`, `.docx`, and `.txt` files.
- **🧠 Gemini 2.5 Flash Power**: Fast, low-latency, and highly intelligent synthesis of information.
- **🎨 Streamlit Interface**: Beautiful, responsive sidebar-based UI for easy document management and chatting.

## 🛠️ How to Run Locally

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Setup**: Add your API Key to a `.env` file:
   ```env
   GEMINI_API_KEY=your_key_here
   ```
4. **Launch the App**:
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment (Hugging Face Spaces)

This repository is pre-configured for Hugging Face Spaces! 

1. Create a new **Streamlit Space** on Hugging Face.
2. Go to **Settings** → **Variables and Secrets**.
3. Add a New Secret: 
   - Name: `GEMINI_API_KEY`
   - Value: `[Your Google API Key]`
4. Upload all project files (`app.py`, `enterprise_rag.py`, `requirements.txt`, and `README.md`).

---
Built with ❤️ using Google Gemini & Streamlit.
