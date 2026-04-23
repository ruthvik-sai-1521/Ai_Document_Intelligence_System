import sys
from pathlib import Path
# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import os
import shutil

from src.core.config import DATA_DIR, FAISS_INDEX_PATH, CHUNKS_PATH, BM25_INDEX_PATH
from src.ingestion.document_processor import DocumentProcessor
from src.core.embedding_manager import EmbeddingManager
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.retriever import HybridRetriever
from src.llm.generator import LLMGenerator
from src.core.pipeline import RAGPipeline
from src.core.logger import setup_logger

logger = setup_logger(__name__)

st.set_page_config(page_title="Advanced AI Document Intelligence", layout="wide")

st.title("📚 Advanced AI Document Intelligence System")
st.markdown("Upload your documents and ask questions. The system uses **Hybrid Search** with **Re-ranking** to provide accurate answers.")

@st.cache_resource(show_spinner="Initializing AI Models (this may take a few minutes if downloading for the first time)...")
def load_models():
    logger.info("Loading Streamlit models...")
    embedding_manager = EmbeddingManager(index_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH)
    llm = LLMGenerator()
    return embedding_manager, llm

embedding_manager, llm = load_models()

if "keyword_search" not in st.session_state:
    st.session_state.keyword_search = KeywordSearch(BM25_INDEX_PATH)
if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever(
        embedding_manager,
        st.session_state.keyword_search
    )
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(st.session_state.retriever, llm)

st.title("📚 Advanced AI Document Intelligence System")
st.markdown("Upload your documents and ask questions. The system uses **Hybrid Search** with **Re-ranking** to provide accurate answers.")

with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload Documents (PDF, TXT)", type=['pdf', 'txt'], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            with st.spinner("Processing documents..."):
                file_paths = []
                for file in uploaded_files:
                    path = DATA_DIR / file.name
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(str(path))
                
                st.info("Extracting and chunking text...")
                processor = DocumentProcessor()
                chunks = processor.process_documents(file_paths)
                
                st.info("Generating embeddings and building FAISS index...")
                embedding_manager.add_chunks(chunks, save=True)
                
                st.info("Updating BM25 index...")
                st.session_state.keyword_search.add_chunks(chunks)
                
                # Re-initialize retriever and pipeline with new data
                st.session_state.retriever = HybridRetriever(
                    embedding_manager,
                    st.session_state.keyword_search
                )
                st.session_state.pipeline = RAGPipeline(st.session_state.retriever, llm)
                
                logger.info(f"Processed {len(uploaded_files)} files.")
                st.success(f"Successfully processed {len(uploaded_files)} files into {len(chunks)} chunks!")

    if st.button("Clear Data"):
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if FAISS_INDEX_PATH.exists(): FAISS_INDEX_PATH.unlink()
        if CHUNKS_PATH.exists(): CHUNKS_PATH.unlink()
        if BM25_INDEX_PATH.exists(): BM25_INDEX_PATH.unlink()
        
        # Reset indices
        embedding_manager._initialize_empty_index()
        st.session_state.keyword_search = KeywordSearch(BM25_INDEX_PATH)
        st.session_state.retriever = HybridRetriever(
            embedding_manager,
            st.session_state.keyword_search
        )
        st.session_state.pipeline = RAGPipeline(st.session_state.retriever, llm)
        logger.info("Cleared all data.")
        st.success("Data cleared.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            confidence = msg.get("confidence", 0.0)
            with st.expander(f"View Sources (Confidence: {confidence:.2f})"):
                for i, source in enumerate(msg["sources"]):
                    metadata = source.get('metadata', {})
                    source_file = metadata.get('source', 'Unknown Document')
                    page_num = metadata.get('page_number', 'N/A')
                    st.markdown(f"**Source {i+1}:** {source_file} (Page: {page_num})")
                    st.markdown(f"> {source['text']}")
                    if 'rerank_score' in source:
                        st.caption(f"Re-rank Score: {source['rerank_score']:.4f}")

if query := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        if not embedding_manager.chunks:
            st.warning("Please upload and process documents first.")
        else:
            with st.spinner("Processing pipeline..."):
                answer, meta = st.session_state.pipeline.run(query)
            
            st.markdown(answer)
            
            if meta['sources']:
                with st.expander(f"View Sources (Confidence: {meta['confidence']:.2f})"):
                    for i, source in enumerate(meta['sources']):
                        metadata = source.get('metadata', {})
                        source_file = metadata.get('source', 'Unknown Document')
                        page_num = metadata.get('page_number', 'N/A')
                        st.markdown(f"**Source {i+1}:** {source_file} (Page: {page_num})")
                        st.markdown(f"> {source['text']}")
                        if 'rerank_score' in source:
                            st.caption(f"Re-rank Score: {source['rerank_score']:.4f}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": meta['sources'],
                "confidence": meta['confidence']
            })
