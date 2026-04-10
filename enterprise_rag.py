import os
import numpy as np
import hashlib
import json
import nest_asyncio
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Import Google GenAI
try:
    from google import genai
except ImportError:
    print("Please install google-genai: pip install google-genai")
    exit(1)

class GeminiEmbeddingFunction:
    """Helper to wrap Google's GenAI SDK for embeddings."""
    def __init__(self, api_key: str = None, model_name: str = 'models/gemini-embedding-001'):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts
        )
        return [e.values for e in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

class EnterpriseRAG:
    def __init__(self, api_key: str = None):
        """Initializes the Enterprise RAG Pipeline (Pinecone + BM25 + Gemini)."""
        # Robust .env loading
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path, override=True)
        self.gemini_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.llama_api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        
        # Semantic Cache settings
        self.cache_path = os.path.join(os.path.dirname(__file__), 'semantic_cache.json')
        self.cache = self._load_cache()
        self.cache_threshold = 0.96 # Very high threshold for accuracy
        
        # Setup LlamaParse if key is present
        self.parser = None
        if self.llama_api_key:
            from llama_parse import LlamaParse
            nest_asyncio.apply()
            self.parser = LlamaParse(
                api_key=self.llama_api_key,
                result_type="markdown", # High fidelity for LLMs
                verbose=True
            )
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables.")

        if self.gemini_api_key:
            self.client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.client = genai.Client()
            
        self.generation_model = 'models/gemini-flash-latest'
        self.embedding_model = 'models/gemini-embedding-001' # 768 dimensions
        
        # 1. Setup Pinecone Index
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "enterprise-story-index"
        
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=768, # Match gemini-embedding-001
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
        self.index = self.pc.Index(self.index_name)
        self.gemini_ef = GeminiEmbeddingFunction(api_key=self.gemini_api_key, model_name=self.embedding_model)
        
        # Hybrid Search setup
        self.bm25 = None
        self.chunks = []
        self.chunk_ids = []
        
        # Note: In Pinecone, we usually fetch all if we want to hydrate BM25 
        # For simplicity in this demo, we'll keep the local metadata if available or re-index.
        # However, a real app would store chunks in Pinecone metadata and fetch.
        self._sync_local_metadata()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        return []

    def _save_to_cache(self, question_emb, answer, sources):
        self.cache.append({
            "embedding": question_emb,
            "answer": answer,
            "sources": sources
        })
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f)

    def _check_cache(self, query_emb):
        """Checks if a similar question has been answered before."""
        if not self.cache:
            return None
            
        for entry in self.cache:
            # Simple cosine similarity
            cached_emb = np.array(entry["embedding"])
            similarity = np.dot(query_emb, cached_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb))
            
            if similarity > self.cache_threshold:
                return entry
        return None

    def _sync_local_metadata(self):
        """Fetches existing chunks from Pinecone to populate local BM25."""
        # Pinecone free tier doesn't support 'list' operations easily, 
        # so we'll rely on the idea that documents are added in this session 
        # or we fetch them if we had stored their IDs.
        # For a full implementation, we'd query with a placeholder.
        pass

    def load_and_process_story(self, text: str, chunk_size: int = 1000, overlap: int = 100, metadata: dict = None):
        print("1. Chunking story intelligently...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        new_chunks = text_splitter.split_text(text)
        
        vectors_to_upsert = []
        embeddings = self.gemini_ef.embed_documents(new_chunks)
        
        # Base metadata (shared by all chunks in this file)
        base_meta = metadata or {}
        if "timestamp" not in base_meta:
            import time
            # Save as integer Unix timestamp for range filtering
            base_meta["timestamp"] = int(time.time())
            
        for i, (chunk, emb) in enumerate(zip(new_chunks, embeddings)):
            # Create a unique ID based on the content hash to prevent duplicates
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            
            if chunk_id not in self.chunk_ids:
                self.chunks.append(chunk)
                self.chunk_ids.append(chunk_id)
                
                # Combine base metadata with the chunk text
                chunk_meta = base_meta.copy()
                chunk_meta["text"] = chunk
                
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": emb,
                    "metadata": chunk_meta
                })
        
        if vectors_to_upsert:
            print(f"2. Upserting {len(vectors_to_upsert)} unique vectors to Pinecone...")
            self.index.upsert(vectors=vectors_to_upsert)
        else:
            print("2. No new content detected. Pinecone is already up to date!")
        
        print("3. Refreshing BM25 Exact Keyword Search...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Enterprise Indexing Complete!")

    def load_single_document(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            # Use LlamaParse for high-quality PDF parsing if available
            if self.parser:
                print(f"Using LlamaParse for {os.path.basename(file_path)}...")
                try:
                    documents = self.parser.load_data(file_path)
                    return "\n\n".join([doc.text for doc in documents])
                except Exception as e:
                    print(f"LlamaParse failed, falling back to basic PDF parsing: {e}")
            
            # Basic fallback
            import fitz
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        elif ext == '.docx':
            from docx import Document
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        return ""

    def load_documents_from_folder(self, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return

        import glob
        files = glob.glob(os.path.join(folder_path, "*"))
        combined_text = ""
        for file_path in files:
            if os.path.isfile(file_path) and not os.path.basename(file_path).startswith('.'):
                print(f"Reading {file_path}...")
                text = self.load_single_document(file_path)
                if text.strip():
                    # Associate each chunk in this file with its filename metadata
                    self.load_and_process_story(text, metadata={"filename": os.path.basename(file_path)})

    def _hybrid_search(self, question: str, filter: dict = None, top_k: int = 3):
        """Combines Pinecone Vector Match with BM25 Keyword Match."""
        # -- 1. Pinecone Vector Search --
        query_emb = self.gemini_ef.embed_query(question)
        vector_results = self.index.query(
            vector=query_emb,
            top_k=top_k * 2,
            include_metadata=True,
            filter=filter
        )
        
        vector_chunks = [match['metadata']['text'] for match in vector_results['matches']]
        vector_ids = [match['id'] for match in vector_results['matches']]
        
        # -- 2. BM25 Keyword Search --
        tokenized_query = question.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[-min(top_k * 2, len(self.chunks)):][::-1]
        bm25_ids = [self.chunk_ids[i] for i in top_bm25_indices]
        
        # -- 3. Reciprocal Rank Fusion (RRF) --
        rrf_scores = {}
        for rank, chunk_id in enumerate(vector_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (60 + rank)
            
        for rank, chunk_id in enumerate(bm25_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (60 + rank)
            
        best_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        # Retrieve final text for best chunks (and ensure uniqueness)
        best_chunks = []
        seen_texts = set()
        
        for cid in best_ids:
            chunk_text = None
            if cid in self.chunk_ids:
                idx = self.chunk_ids.index(cid)
                chunk_text = self.chunks[idx]
            else:
                for match in vector_results['matches']:
                    if match['id'] == cid:
                        chunk_text = match['metadata']['text']
                        break
            
            if chunk_text and chunk_text not in seen_texts:
                best_chunks.append(chunk_text)
                seen_texts.add(chunk_text)
                        
        return best_chunks
        
    def answer_question(self, question: str, chat_history: list = None, filter: dict = None, time_window: int = None):
        """Retrieves context and generates a streaming answer with Time & Metadata Filtering."""
        if not self.bm25:
            self.rehydrate_from_cloud()
            if not self.bm25:
                raise ValueError("No documents loaded in Pinecone yet.")
        
        # 1. Prepare Pinecone Filter
        final_filter = filter or {}
        if time_window:
            import time
            cutoff = int(time.time()) - time_window
            final_filter["timestamp"] = {"$gte": cutoff}
            
        # 2. Check Semantic Cache First (Speed)
        query_emb = self.gemini_ef.embed_query(question)
        cached_result = self._check_cache(query_emb)
        
        if cached_result:
            print("🚀 Semantic Cache Hit! Returning instant answer.")
            def cache_generator():
                yield "*(Cached Answer)*  \n"
                yield cached_result["answer"]
            
            return {
                "answer_stream": cache_generator(),
                "sources": cached_result["sources"],
                "is_cached": True
            }

        # 3. Proceed with RAG if Cache Miss
        print(f"\nRunning Hybrid Search with filter: {final_filter}")
        retrieved_chunks = self._hybrid_search(question, filter=final_filter)
        
        # Format the context with numbered sources for the LLM to cite
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"--- SOURCE {i+1} ---\n{chunk}")
        context_text = "\n\n".join(context_parts)
        
        # Format chat history for the prompt
        history_text = ""
        if chat_history:
            for msg in chat_history[-5:]: # Look at last 5 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        prompt = f"""You are an assistant answering questions based strictly on the provided sources and conversation history.
Your goal is to be extremely direct and precise.

INSTRUCTIONS:
1. Answer the question using ONLY the provided sources.
2. CITATIONS: You MUST cite the source number in brackets (e.g., [Source 1]) for every claim.
3. DIRECTNESS: Answer EXACTLY what is asked. Do not include extra information unless specific details are requested.
4. If the answer is not in the sources, say "I cannot answer this based on the provided sources."

Conversation History:
{history_text}

New Question: {question}

Sources:
{context_text}

Final Answer:"""

        print("Generating streaming answer with citations...")
        
        # Use generate_content_stream for real-time feedback with automatic retries
        import time
        max_retries = 3
        retry_delay = 2 # seconds
        
        for attempt in range(max_retries):
            try:
                response_stream = self.client.models.generate_content_stream(
                    model=self.generation_model,
                    contents=prompt,
                )
                
                def stream_generator():
                    full_response = ""
                    for chunk in response_stream:
                        full_response += chunk.text
                        yield chunk.text
                    
                    # Save to cache after streaming is complete
                    self._save_to_cache(query_emb, full_response, retrieved_chunks)

                return {
                    "answer_stream": stream_generator(),
                    "sources": retrieved_chunks,
                    "is_cached": False
                }
            except Exception as e:
                is_busy = "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)
                if is_busy and attempt < max_retries - 1:
                    print(f"⚠️ Gemini busy (Attempt {attempt+1}/{max_retries}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Exponential backoff
                else:
                    raise e

    def rehydrate_from_cloud(self):
        """Fetches top 100 documents from Pinecone to populate BM25 on startup."""
        print("Rehydrating BM25 from Pinecone cloud...")
        # Since Pinecone doesn't support easy 'list all', we query with a random vector
        # We use .tolist() to ensure standard Python floats for orjson serialization
        random_vector = np.random.rand(768).tolist()
        results = self.index.query(vector=random_vector, top_k=100, include_metadata=True)
        
        self.chunks = []
        self.chunk_ids = []
        for match in results['matches']:
            self.chunks.append(match['metadata']['text'])
            self.chunk_ids.append(match['id'])
            
        if self.chunks:
            tokenized_corpus = [doc.lower().split(" ") for doc in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)

if __name__ == "__main__":
    rag = EnterpriseRAG()
    # Usage would be similar to before...
