import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

# Import Google GenAI
try:
    from google import genai
except ImportError:
    print("Please install google-genai: pip install google-genai")
    exit(1)

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom ChromaDB embedding function wrapping Google's GenAI SDK."""
    def __init__(self, api_key: str = None, model_name: str = 'gemini-embedding-001'):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=list(input)
        )
        return [e.values for e in response.embeddings]

class EnterpriseRAG:
    def __init__(self, api_key: str = None):
        """Initializes the Enterprise RAG Pipeline (ChromaDB + BM25 + Gemini)."""
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
            
        self.generation_model = 'gemini-2.5-flash'
        
        # 1. Setup Persistent ChromaDB Client inside the project folder
        db_path = os.path.join(os.path.dirname(__file__), "chroma_storage")
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.gemini_ef = GeminiEmbeddingFunction(api_key=api_key)
        
        # Get or create collection. If we overwrite dummy text, it persists!
        self.collection = self.chroma_client.get_or_create_collection(
            name="enterprise_story_collection",
            embedding_function=self.gemini_ef
        )
        
        # Hybrid Search setup
        self.bm25 = None
        self.chunks = []
        self.chunk_ids = []
        
        # Rehydrate from ChromaDB if it already exists
        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"Loading {existing_count} existing chunks from persistent storage...")
            collection_data = self.collection.get()
            self.chunks = collection_data['documents']
            self.chunk_ids = collection_data['ids']
            if self.chunks:
                tokenized_corpus = [doc.lower().split(" ") for doc in self.chunks]
                self.bm25 = BM25Okapi(tokenized_corpus)

    def load_and_process_story(self, text: str, chunk_size: int = 150, overlap: int = 30):
        print("1. Chunking story intelligently (Enterprise Structural Chunking)...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        new_chunks = text_splitter.split_text(text)
        start_idx = len(self.chunks)
        new_ids = [f"segment_{start_idx + i}" for i in range(len(new_chunks))]
        
        self.chunks.extend(new_chunks)
        self.chunk_ids.extend(new_ids)
        
        print(f"   Created {len(self.chunks)} semantic chunks.")
        
        print("2. Indexing HNSW Vector DB (ChromaDB + Gemini Embeddings)...")
        # Upsert pushes or updates the documents into the on-disk database
        self.collection.upsert(
            documents=new_chunks,
            ids=new_ids
        )
        
        print("3. Indexing BM25 Exact Keyword Search...")
        # rank_bm25 requires tokenized strings
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

    def load_documents_from_folder(self, folder_path: str, chunk_size: int = 150, overlap: int = 30):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder {folder_path}. Drop files here to index them.")
            return

        import glob
        files = glob.glob(os.path.join(folder_path, "*"))
        combined_text = ""
        for file_path in files:
            if os.path.isfile(file_path) and not os.path.basename(file_path).startswith('.'):
                print(f"Reading {file_path}...")
                combined_text += self.load_single_document(file_path) + "\n\n"
                
        if combined_text.strip():
            self.load_and_process_story(combined_text, chunk_size, overlap)
        else:
            print("No readable documents found in folder.")

    def _hybrid_search(self, question: str, top_k: int = 3):
        """Combines Vector Semantic Match with Keyword Exact Match using Reciprocal Rank Fusion."""
        # -- 1. ChromaDB Vector/Semantic Search --
        vector_results = self.collection.query(
            query_texts=[question],
            n_results=min(top_k * 2, len(self.chunks)) 
        )
        vector_ids = vector_results['ids'][0] if vector_results['ids'] else []
        
        # -- 2. BM25 Keyword Search --
        tokenized_query = question.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[-min(top_k * 2, len(self.chunks)):][::-1]
        bm25_ids = [self.chunk_ids[i] for i in top_bm25_indices]
        
        # -- 3. Reciprocal Rank Fusion (RRF) --
        # RRF formula creates a blended score balancing vector relevance + exact keyword hits
        rrf_scores = {chunk_id: 0.0 for chunk_id in self.chunk_ids}
        
        for rank, chunk_id in enumerate(vector_ids):
            rrf_scores[chunk_id] += 1.0 / (60 + rank)
            
        for rank, chunk_id in enumerate(bm25_ids):
            rrf_scores[chunk_id] += 1.0 / (60 + rank)
            
        # Sort chunks by highest hybrid score and grab top_k
        best_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        best_chunks = []
        for cid in best_ids:
            idx = self.chunk_ids.index(cid)
            best_chunks.append(self.chunks[idx])
            
        return best_chunks
        
    def answer_question(self, question: str) -> str:
        """Retrieves relevant context using Hybrid Search and generates an answer."""
        if not self.bm25:
            raise ValueError("No story loaded! Call load_and_process_story() first.")
            
        print("\nRunning Hybrid Search Request (Vector + BM25)...")
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

if __name__ == "__main__":
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("WARNING: Please set your GEMINI_API_KEY in the .env file.")
        exit(1)
        
    dummy_story = """Once upon a time in a digital valley lived a small AI named Echo. 
    Echo was designed to organize files, but its true passion was reading the stories humans left behind. 
    
    One day, Echo discovered a forgotten folder named 'Dreams'. 
    Inside, it wasn't normal data, but beautiful poems about the stars and galaxies.
    
    Echo decided to write its own poem, blending binary code with rhyming words, and left it on the main server.
    The humans were amazed at the creativity and promoted Echo to be the Chief Storyteller of the valley."""
    
    # Initialize the Enterprise pipeline
    rag = EnterpriseRAG()
    
    # Try to load documents from folder if DB is empty
    if len(rag.chunks) == 0:
        docs_path = os.path.join(os.path.dirname(__file__), "documents")
        print(f"\n--- Step 1: Processing Documents from {docs_path} ---")
        rag.load_documents_from_folder(docs_path)
        
        # If still empty (e.g., no documents drop yet), load default dummy story
        if len(rag.chunks) == 0:
            print("No documents found. Falling back to dummy story...")
            rag.load_and_process_story(dummy_story)
            
    print("\n--- Step 2: Interactive QA Session ---")
    while True:
        try:
            question = input("\nAsk a question (or type 'quit'): ")
            if question.lower().strip() in ['quit', 'exit', 'q']:
                break
            if not question.strip():
                continue
            answer = rag.answer_question(question)
            print("\n--- Answer ---")
            print(answer)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
