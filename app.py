import streamlit as st
import os
import time
from enterprise_rag import EnterpriseRAG
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize RAG
@st.cache_resource
def get_rag_pipeline():
    api_key = os.environ.get("GEMINI_API_KEY")
    return EnterpriseRAG(api_key=api_key)

rag = get_rag_pipeline()

# 2. UI Layout
st.set_page_config(page_title="Story RAG Enterprise", layout="wide")
st.title("📚 Story RAG - Enterprise Edition")
st.subheader("Cloud-Native, High-Precision Knowledge Explorer")

# 3. Sidebar Configuration
docs_path = "documents"
if not os.path.exists(docs_path):
    os.makedirs(docs_path)

with st.sidebar:
    st.title("Knowledge Explorer")
    available_files = [f for f in os.listdir(docs_path) if not f.startswith('.')]
    
    selected_files = st.multiselect(
        "📁 Filter by Source Document:",
        options=available_files,
        default=available_files
    )
    
    st.divider()
    
    # Time Filtering
    st.write("🕒 **Chronological Filter**")
    time_options = {
        "All Time": None,
        "Last Hour": 3600,
        "Last 24 Hours": 86400,
        "Last 7 Days": 604800,
        "Last 30 Days": 2592000
    }
    time_label = st.select_slider(
        "Search information from:",
        options=list(time_options.keys()),
        value="All Time"
    )
    time_window = time_options[time_label]
    
    st.divider()
    if st.button("Reset Chat Session"):
        st.session_state.messages = []
        st.rerun()

# 4. Processing Interface
with st.expander("☁️ Cloud Data Management", expanded=False):
    uploaded_files = st.file_uploader("Upload new documents to Pinecone", accept_multiple_files=True)
    if st.button("🚀 Process & Index Files"):
        if uploaded_files:
            for file in uploaded_files:
                dest_path = os.path.join(docs_path, file.name)
                with open(dest_path, "wb") as f:
                    f.write(file.getbuffer())
                
                try:
                    text = rag.load_single_document(dest_path)
                    if text.strip():
                        # Tag with filename and current unix time
                        rag.load_and_process_story(text, metadata={"filename": file.name, "timestamp": int(time.time())})
                        st.success(f"Successfully indexed `{file.name}`")
                    else:
                        st.error(f"Failed to read `{file.name}`")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
            rag.rehydrate_from_cloud() 
        else:
            st.warning("Please upload files first.")
            
    st.markdown(f"**Index Health:** {len(rag.chunks)} unique chunks currently in active memory.")

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your knowledge base..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not rag.bm25:
            response = "Knowledge base is empty. Please upload documents first."
            st.markdown(response)
        else:
            try:
                # Prepare Pinecone metadata filter
                pinecone_filter = None
                if selected_files:
                    pinecone_filter = {"filename": {"$in": selected_files}}
                
                rag_response = rag.answer_question(
                    prompt, 
                    chat_history=st.session_state.messages, 
                    filter=pinecone_filter,
                    time_window=time_window
                )
                stream = rag_response["answer_stream"]
                sources = rag_response["sources"]
                is_cached = rag_response.get("is_cached", False)
                
                if is_cached:
                    st.caption("🚀 Answer retrieved from Semantic Cache (Instant)")
                
                response = st.write_stream(stream)
                
                with st.expander("🔍 Fact Check: View Sources"):
                    for i, src in enumerate(sources):
                        st.info(f"**Source {i+1}**\n\n{src}")
            except Exception as e:
                response = f"Error during synthesis: {e}"
                st.error(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
