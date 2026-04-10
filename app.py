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

    st.divider()
    if st.button("🔄 Force Cloud Sync"):
        with st.spinner("Syncing local memory with Pinecone cloud..."):
            success = rag.rehydrate_from_cloud()
            if success:
                st.success(f"Successfully loaded {len(rag.chunks)} chunks from Cloud.")
            else:
                st.warning("Cloud search returned 0 results. Use the management section to upload documents.")

    st.divider()
    with st.expander("🛠️ Debug & Repair Index"):
        health = rag.check_index_health()
        if health["status"] == "Ready":
            st.success("Internal Connection: OK")
            st.write(f"Index Dimension: `{health['dimension']}`")
            st.divider()
            if st.button("🔍 Run Smoke Test Query"):
                test = rag.run_smoke_test()
                if test["success"]:
                    st.success("🔥 Smoke Test Passed! Data exists in cloud.")
                    st.write(f"Sample Content Found: `{test['match'][:150]}...`")
                else:
                    st.error(f"Smoke Test Failed: {test['message']}")
            
            if not health["is_correct_dim"]:
                st.error("🚨 DIMENSION MISMATCH DETECTED!")
                st.warning("Your index is 3072D but Gemini needs 768D. This is why you see 0 vectors.")
                if st.button("🔥 Wipe & Recreate Index"):
                    if rag.delete_and_recreate_index():
                        st.success("Index Recreated! Please re-upload your files.")
                        st.rerun()
            else:
                st.info("Dimensions are correct (768D).")
        else:
            st.error(f"Connection Error: {health.get('message')}")

# 4. Processing Interface
with st.expander("☁️ Cloud Data Management", expanded=False):
    uploaded_files = st.file_uploader("Upload new documents to Pinecone", accept_multiple_files=True)
    if st.button("🚀 Process & Index Files"):
        if uploaded_files:
            with st.status("Processing and indexing documents...", expanded=True) as status:
                for file in uploaded_files:
                    dest_path = os.path.join(docs_path, file.name)
                    with open(dest_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    try:
                        status.write(f"Reading `{file.name}`...")
                        text = rag.load_single_document(dest_path)
                        if text.strip():
                            def progress_cb(msg):
                                status.update(label=f"Processing `{file.name}`: {msg}")
                                
                            # Tag with filename and current unix time
                            rag.load_and_process_story(
                                text, 
                                metadata={"filename": file.name, "timestamp": int(time.time())},
                                status_callback=progress_cb
                            )
                            st.toast(f"✅ Indexed `{file.name}`")
                        else:
                            st.error(f"Failed to read `{file.name}`")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
                
                status.update(label="All documents successfully indexed!", state="complete", expanded=False)
        else:
            st.warning("Please upload files first.")
            
    # Index Health Stats
    cloud_count = rag.get_cloud_stats()
    local_count = len(rag.chunks)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Local Active Memory", f"{local_count} Chunks")
    with col2:
        st.metric("Total Cloud Storage", f"{cloud_count} Vectors")
    
    if cloud_count > 0 and local_count == 0:
        st.info("💡 Your data is in the cloud, but local memory is empty. Use the **'Force Cloud Sync'** button in the sidebar to load it.")

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
