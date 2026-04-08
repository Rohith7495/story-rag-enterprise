import os
import numpy as np
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
                dimension=3072,
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

    def _sync_local_metadata(self):
        """Fetches existing chunks from Pinecone to populate local BM25."""
        # Pinecone free tier doesn't support 'list' operations easily, 
        # so we'll rely on the idea that documents are added in this session 
        # or we fetch them if we had stored their IDs.
        # For a full implementation, we'd query with a placeholder.
        pass

    def load_and_process_story(self, text: str, chunk_size: int = 1000, overlap: int = 100):
        print("1. Chunking story intelligently...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        new_chunks = text_splitter.split_text(text)
        start_idx = len(self.chunks)
        
        vectors_to_upsert = []
        embeddings = self.gemini_ef.embed_documents(new_chunks)
        
        for i, (chunk, emb) in enumerate(zip(new_chunks, embeddings)):
            chunk_id = f"segment_{start_idx + i}"
            self.chunks.append(chunk)
            self.chunk_ids.append(chunk_id)
            vectors_to_upsert.append({
                "id": chunk_id,
                "values": emb,
                "metadata": {"text": chunk}
            })
        
        print(f"2. Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        self.index.upsert(vectors=vectors_to_upsert)
        
        print("3. Indexing BM25 Exact Keyword Search...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Enterprise Indexing Complete!")

    def load_single_document(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
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
                combined_text += self.load_single_document(file_path) + "\n\n"
                
        if combined_text.strip():
            self.load_and_process_story(combined_text)

    def _hybrid_search(self, question: str, top_k: int = 3):
        """Combines Pinecone Vector Match with BM25 Keyword Match."""
        # -- 1. Pinecone Vector Search --
        query_emb = self.gemini_ef.embed_query(question)
        vector_results = self.index.query(
            vector=query_emb,
            top_k=top_k * 2,
            include_metadata=True
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
        
        # Retrieve final text for best chunks
        # We can optimize this by mapping IDs to text locally or fetching from Pinecone metadata
        best_chunks = []
        for cid in best_ids:
            # First look in current session chunks
            if cid in self.chunk_ids:
                idx = self.chunk_ids.index(cid)
                best_chunks.append(self.chunks[idx])
            else:
                # Fallback to Pinecone metadata if not in current session list
                for match in vector_results['matches']:
                    if match['id'] == cid:
                        best_chunks.append(match['metadata']['text'])
                        break
                        
        return best_chunks
        
    def answer_question(self, question: str) -> str:
        if not self.bm25:
            # If BM25 isn't ready (app just started), try to rehydrate from Pinecone
            self.rehydrate_from_cloud()
            if not self.bm25:
                raise ValueError("No documents loaded in Pinecone yet.")
            
        print("\nRunning Hybrid Search (Pinecone + BM25)...")
        retrieved_chunks = self._hybrid_search(question)
        context = "\n\n---\n\n".join(retrieved_chunks)
        
        prompt = f"""You are an assistant answering questions based strictly on the provided story excerpt.
Do not use outside knowledge. If the answer is not in the story excerpts, say "I cannot answer this based on the provided story."

Story Excerpts:
{context}

Question: {question}

Answer:"""

        print("Generating final synthesized answer...")
        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt,
        )
        return response.text

    def rehydrate_from_cloud(self):
        """Fetches top 100 documents from Pinecone to populate BM25 on startup."""
        print("Rehydrating BM25 from Pinecone cloud...")
        # Since Pinecone doesn't support easy 'list all', we query with a random vector
        random_vector = list(np.random.rand(768))
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
