import streamlit as st
from enterprise_rag import EnterpriseRAG
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Enterprise Story RAG", page_icon="📚", layout="wide")

@st.cache_resource
def get_rag_pipeline():
    api_key = os.environ.get("GEMINI_API_KEY")
    return EnterpriseRAG(api_key=api_key)

rag = get_rag_pipeline()

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Try to load documents from folder if DB is empty on startup
if len(rag.chunks) == 0:
    docs_path = os.path.join(os.path.dirname(__file__), "documents")
    if os.path.exists(docs_path):
        rag.load_documents_from_folder(docs_path)

st.title("📚 Enterprise Story RAG Chat")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Management")
    st.markdown("Upload `.pdf`, `.docx`, or `.txt` directly into the persistent ChromaDB index.")
    
    uploaded_files = st.file_uploader("Upload Document", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    
    if st.button("Process Uploaded Files"):
        if uploaded_files:
            for file in uploaded_files:
                # Save uploaded file temporarily to process
                temp_path = os.path.join("/tmp", file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                try:
                    text = rag.load_single_document(temp_path)
                    if text.strip():
                        rag.load_and_process_story(text)
                        st.success(f"Successfully processed `{file.name}`")
                    else:
                        st.error(f"Failed to extract text from `{file.name}`")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
        else:
            st.warning("Please upload files first.")
            
    st.markdown(f"**Database Size:** {len(rag.chunks)} chunks currently indexed.")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the story..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not rag.bm25:
            response = "Please upload a document first or put one in the documents/ folder."
            st.markdown(response)
        else:
            try:
                rag_response = rag.answer_question(prompt, chat_history=st.session_state.messages)
                stream = rag_response["answer_stream"]
                sources = rag_response["sources"]
                is_cached = rag_response.get("is_cached", False)
                
                if is_cached:
                    st.caption("🚀 Answer retrieved from Semantic Cache (Instant)")
                
                # Use st.write_stream for real-time typing effect
                response = st.write_stream(stream)
                
                # Show sources in an expander for transparency
                with st.expander("🔍 View Sources & Fact-Check"):
                    for i, src in enumerate(sources):
                        st.info(f"**Source {i+1}**\n\n{src}")
            except Exception as e:
                response = f"Error: {e}"
                st.error(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
