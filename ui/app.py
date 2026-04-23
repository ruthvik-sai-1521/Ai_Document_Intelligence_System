import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import shutil
import time
import uuid
from datetime import datetime

from src.core.config import DATA_DIR, FAISS_INDEX_PATH, CHUNKS_PATH, BM25_INDEX_PATH
from src.ingestion.document_processor import DocumentProcessor
from src.core.embedding_manager import EmbeddingManager
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.retriever import HybridRetriever
from src.llm.generator import LLMGenerator
from src.core.pipeline import RAGPipeline
from src.core.chat_history import (
    save_chat, load_chat_history, load_today_history, 
    load_messages_for_date, clear_history,
    save_document_meta, load_document_meta, delete_document_meta, clear_all_metadata,
    save_query_metrics, load_analytics_summary
)
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Outfit', 'Inter', sans-serif;
    color: #2d3748;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    background-image: radial-gradient(at 0% 0%, rgba(30, 58, 138, 0.5) 0, transparent 50%), 
                      radial-gradient(at 50% 0%, rgba(76, 29, 149, 0.4) 0, transparent 50%);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] * { color: #f8fafc !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.05);
    padding: 0.6rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.12);
    border-color: rgba(255,255,255,0.3);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Header Banner */
.dashboard-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #581c87 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
}
.dashboard-header h1 { 
    margin: 0; 
    font-size: 2.2rem; 
    font-weight: 800; 
    letter-spacing: -0.025em; 
}
.dashboard-header p { 
    margin: 8px 0 0; 
    opacity: 0.9; 
    font-size: 1rem; 
    font-weight: 300; 
}

/* Metric Cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #f1f5f9;
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    border-color: #e2e8f0;
}
.metric-card .metric-value { 
    font-size: 2.25rem; 
    font-weight: 800; 
    color: #1e3a8a; 
    line-height: 1;
}
.metric-card .metric-label { 
    font-size: 0.8rem; 
    color: #64748b; 
    margin-top: 8px; 
    font-weight: 600;
    text-transform: uppercase; 
    letter-spacing: 0.05em; 
}
.metric-card .metric-icon { 
    font-size: 1.75rem; 
    margin-bottom: 8px; 
    opacity: 0.8;
}

/* Tab Container Customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    padding: 0 10px;
    border-bottom: 2px solid #f1f5f9;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    font-weight: 600;
    font-size: 0.95rem;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    color: #1e3a8a !important;
}

/* Chat bubble styling */
[data-testid="stChatMessage"] {
    border-radius: 20px;
    margin-bottom: 0.75rem;
    padding: 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #ffffff;
    border: 1px solid #e8e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* Typing dots animation */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 0;
}
.typing-indicator span {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #667eea;
    animation: bounce 1.2s infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30%            { transform: translateY(-6px); }
}

/* Message timestamp */
.msg-meta {
    font-size: 0.72rem;
    color: #aaa;
    margin-top: 2px;
}

/* Chat container scroll anchor */
#chat-bottom { height: 1px; }

/* Source citation cards */
.source-block {
    background: #ffffff;
    border: 1px solid #e8e8f0;
    border-left: 4px solid #667eea;
    padding: 12px 16px;
    border-radius: 4px 12px 12px 4px;
    margin: 8px 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}
.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    border-bottom: 1px solid #f0f0f5;
    padding-bottom: 4px;
}
.source-name { font-weight: 600; color: #302b63; font-size: 0.88rem; }
.source-page { font-size: 0.72rem; color: #888; background: #f0f2f6; padding: 2px 8px; border-radius: 4px; }
.source-text { font-style: italic; color: #555; font-size: 0.82rem; line-height: 1.4; }
.source-footer { margin-top: 6px; font-size: 0.7rem; color: #bbb; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE INITIALIZATION ─────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

defaults = {
    "messages": [],
    "uploaded_docs": load_document_meta(st.session_state.user_id),
    "selected_date": datetime.now().strftime("%Y-%m-%d"),
    "viewing_history": False,
    "suggested_query": None,
    "msg_limit": 15,          # Performance: only render last N messages
    "analytics": load_analytics_summary(st.session_state.user_id)
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Restore today's conversation from DB on first load
if not st.session_state.messages:
    today_msgs = load_today_history(st.session_state.user_id)
    if today_msgs:
        st.session_state.messages = today_msgs

# ─────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Initializing AI models — first run downloads ~80MB...")
def load_models():
    logger.info("Loading models...")
    em = EmbeddingManager(index_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH)
    llm = LLMGenerator()
    return em, llm

embedding_manager, llm = load_models()

def stream_data(text: str):
    """Generator to simulate a typing effect for Streamlit."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def export_history_to_markdown(history_dict):
    """Formats the entire SQLite history into a clean Markdown string."""
    md = "# 🧠 DocuMind AI - Chat History Export\n\n"
    md += f"*Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n"
    
    for date_str, messages in history_dict.items():
        md += f"## 📅 {date_str}\n"
        for msg in messages:
            role = "👤 **User**" if msg["role"] == "user" else "🤖 **DocuMind AI**"
            md += f"### {role} | {msg['timestamp']}\n"
            md += f"{msg['content']}\n\n"
            if msg.get("confidence") and msg["confidence"] > 0:
                md += f"*Confidence: {msg['confidence']:.2f}*\n\n"
        md += "---\n\n"
    return md

def highlight_text(text: str, query: str) -> str:
    """Wraps matching keywords in <mark> tags for visual highlighting."""
    if not query: return text
    import re
    # Split query into words to highlight each individually
    words = [re.escape(w) for w in query.split() if len(w) > 2]
    if not words: return text
    
    pattern = re.compile(f"({'|'.join(words)})", re.IGNORECASE)
    return pattern.sub(r"<mark style='background:#fde047; color:black; border-radius:2px; padding:0 2px;'>\1</mark>", text)

if "keyword_search" not in st.session_state:
    st.session_state.keyword_search = KeywordSearch(BM25_INDEX_PATH)
if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever(embedding_manager, st.session_state.keyword_search)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(st.session_state.retriever, llm)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
            <span style="font-size: 2rem;">🧠</span>
            <div>
                <h2 style="margin: 0; font-size: 1.4rem;">DocuMind AI</h2>
                <div class="status-pill">
                    <div class="status-dot"></div>
                    Groq API Connected
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if st.button("⚡ Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing..."):
                file_paths = []
                for file in uploaded_files:
                    path = DATA_DIR / file.name
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(str(path))

                processor = DocumentProcessor()
                chunks = processor.process_documents(file_paths, user_id=st.session_state.user_id)
                embedding_manager.add_chunks(chunks, save=True)
                st.session_state.keyword_search.add_chunks(chunks)

                st.session_state.retriever = HybridRetriever(embedding_manager, st.session_state.keyword_search)
                st.session_state.pipeline  = RAGPipeline(st.session_state.retriever, llm)

                # Track uploaded docs persistently
                for file in uploaded_files:
                    doc_chunks = [c for c in chunks if c.get("metadata", {}).get("source") == file.name]
                    chunk_count = len(doc_chunks)
                    save_document_meta(st.session_state.user_id, file.name, chunk_count)
                
                # Refresh session state
                st.session_state.uploaded_docs = load_document_meta(st.session_state.user_id)

                logger.info(f"Processed {len(uploaded_files)} files.")
                st.success(f"✅ {len(uploaded_files)} file(s) processed!")

    st.markdown("---")
    st.markdown("### 📚 Indexed Documents")
    if st.session_state.uploaded_docs:
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.08);border-radius:8px;padding:8px 12px;margin:4px 0;'>
                <b>📄 {doc['filename']}</b><br>
                <small>{doc['chunk_count']} chunks · {doc['upload_date']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No documents indexed yet.")

    st.markdown("---")

    # ── History Navigator ──────────────────────────────
    st.markdown("### 🗂️ Chat History")
    history = load_chat_history(st.session_state.user_id, limit_days=30)
    today_str = datetime.now().strftime("%Y-%m-%d")

    if not history:
        st.caption("No history yet.")
    else:
        for date_str in sorted(history.keys(), reverse=True):
            msgs      = history[date_str]
            q_count   = sum(1 for m in msgs if m["role"] == "user")
            is_active = (date_str == st.session_state.selected_date)

            if date_str == today_str:
                label = f"📅 Today ({q_count}Q)"
            else:
                # Format: Apr 22
                from datetime import datetime as _dt
                label = _dt.strptime(date_str, "%Y-%m-%d").strftime("%b %d") + f"  ({q_count}Q)"

            # Active session highlighted differently
            if is_active:
                st.markdown(
                    f'<div style="background:rgba(102,126,234,0.4);border:1px solid rgba(255,255,255,0.5);'
                    f'border-radius:8px;padding:8px 12px;margin:3px 0;font-weight:600;font-size:0.85rem;">'
                    f'▶ {label}</div>',
                    unsafe_allow_html=True
                )
            else:
                if st.button(label, key=f"hist_{date_str}", use_container_width=True):
                    st.session_state.selected_date  = date_str
                    st.session_state.viewing_history = True
                    st.session_state.messages = load_messages_for_date(st.session_state.user_id, date_str)
                    st.rerun()

    # New Chat button — returns to today live session
    st.markdown("")
    if st.session_state.viewing_history:
        if st.button("✏️ New Chat (Today)", use_container_width=True):
            st.session_state.selected_date  = today_str
            st.session_state.viewing_history = False
            st.session_state.messages = load_today_history(st.session_state.user_id)
            st.rerun()

    # ── Recent Queries ────────────────────────────────
    st.markdown("### 🕒 Recent Queries")
    recent = list(st.session_state.analytics["query_text_history"].keys())[-5:]
    if recent:
        for q in reversed(recent):
            if st.button(f"🔍 {q[:30]}...", key=f"recent_{q}", use_container_width=True, help=q):
                st.session_state.suggested_query = q
                st.rerun()
    else:
        st.caption("No recent queries.")

    st.markdown("---")
    if st.button("🗑️ Clear All Data", use_container_width=True):
        if DATA_DIR.exists(): shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for p in [FAISS_INDEX_PATH, CHUNKS_PATH, BM25_INDEX_PATH]:
            if p.exists(): p.unlink()
        embedding_manager._initialize_empty_index()
        st.session_state.keyword_search = KeywordSearch(BM25_INDEX_PATH)
        st.session_state.retriever = HybridRetriever(embedding_manager, st.session_state.keyword_search)
        st.session_state.pipeline  = RAGPipeline(st.session_state.retriever, llm)
        clear_all_metadata(st.session_state.user_id)
        st.session_state.uploaded_docs = []
        st.session_state.messages = []
        st.session_state.analytics = load_analytics_summary(st.session_state.user_id)
        st.success("All data cleared.")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <h1>🧠 DocuMind AI — Document Intelligence Dashboard</h1>
    <p>Hybrid Semantic + Keyword Search · Re-ranking · Anti-Hallucination · Source Attribution</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_chat, tab_history, tab_analytics, tab_docs = st.tabs(["💬 Chat", "🗂️ History", "📊 Analytics", "📂 Documents"])

# ════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════
with tab_chat:
    a = st.session_state.analytics

    # ── Chat header ──────────────────────────────────
    hcol1, hcol2 = st.columns([6, 2])
    with hcol1:
        st.markdown("#### 💬 Conversation")
    with hcol2:
        msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(
            f'<div style="text-align:right;padding-top:6px;">' +
            f'<span style="background:#667eea;color:white;padding:3px 10px;border-radius:12px;font-size:0.78rem;">' +
            f'{msg_count} message{"s" if msg_count != 1 else ""}</span></div>',
            unsafe_allow_html=True
        )

    # ── Render conversation history (with windowing) ──
    total_msgs = len(st.session_state.messages)
    display_msgs = st.session_state.messages[-st.session_state.msg_limit:]
    
    if total_msgs > st.session_state.msg_limit:
        if st.button(f"🔼 Load Older Messages ({total_msgs - st.session_state.msg_limit} hidden)", use_container_width=True):
            st.session_state.msg_limit += 15
            st.rerun()

    for msg in display_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Timestamp
            ts = msg.get("timestamp", "")
            st.markdown(f'<div class="msg-meta">{ts}</div>', unsafe_allow_html=True)

            # Sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                conf = msg.get("confidence", 0.0)
                if conf >= 0.5:
                    badge = f'<span class="conf-high">● High Confidence {conf:.2f}</span>'
                elif conf >= 0.25:
                    badge = f'<span class="conf-med">● Med Confidence {conf:.2f}</span>'
                else:
                    badge = f'<span class="conf-low">● Low Confidence {conf:.2f}</span>'
                st.markdown(badge, unsafe_allow_html=True)

                with st.expander("🔗 Reference Citations"):
                    for i, src in enumerate(msg["sources"]):
                        meta = src.get("metadata", {})
                        st.markdown(f"""
                        <div class="source-block">
                            <div class="source-header">
                                <span class="source-name">📄 {meta.get('source','Unknown')}</span>
                                <span class="source-page">Page {meta.get('page_number','N/A')}</span>
                            </div>
                            <div class="source-text">"{src.get('snippet', src['text'])[:300]}..."</div>
                            <div class="source-footer">Re-rank Quality: {src.get('rerank_score',0):.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Auto-scroll anchor (Streamlit renders top-to-bottom, chat_input stays pinned at bottom)
    st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

    # ── Chat input & Suggested Query Logic ──────────────
    query = st.chat_input("Ask anything about your documents...")
    
    # Handle clicks from "Recent Queries" sidebar
    if st.session_state.suggested_query:
        query = st.session_state.suggested_query
        st.session_state.suggested_query = None # Reset

    if st.session_state.viewing_history:
        st.info(
            f"📖 Viewing read-only history for **{st.session_state.selected_date}**. "
            "Click **✏️ New Chat (Today)** in the sidebar to resume chatting."
        )
    elif query:
        now = datetime.now().strftime("%H:%M")

        # Render user bubble immediately
        with st.chat_message("user"):
            st.markdown(query)
            st.markdown(f'<div class="msg-meta">{now}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": query, "timestamp": now})
        save_chat(user_id=st.session_state.user_id, role="user", content=query)

        # Render assistant bubble
        with st.chat_message("assistant"):
            if not embedding_manager.chunks:
                st.warning("⚠️ No documents indexed yet. Upload PDFs using the sidebar.")
            else:
                # Typing indicator while processing
                typing_slot = st.empty()
                typing_slot.markdown(
                    '<div class="typing-indicator"><span></span><span></span><span></span></div>',
                    unsafe_allow_html=True
                )

                start = time.time()
                answer, meta = st.session_state.pipeline.run(query, user_id=st.session_state.user_id)
                elapsed = time.time() - start

                # Replace typing indicator with streaming real answer
                typing_slot.empty()
                
                # Use st.write_stream for a professional typing effect
                full_response = st.write_stream(stream_data(answer))

                conf    = meta.get("confidence", 0.0)
                sources = meta.get("sources", [])
                ans_ts  = datetime.now().strftime("%H:%M")

                st.markdown(f'<div class="msg-meta">⏱ {elapsed:.2f}s &nbsp;·&nbsp; {ans_ts}</div>', unsafe_allow_html=True)

                # Update Persistent Analytics
                is_answered = "insufficient data" not in answer.lower()
                save_query_metrics(st.session_state.user_id, query, elapsed, conf, is_answered)
                
                # Refresh UI Analytics State
                st.session_state.analytics = load_analytics_summary(st.session_state.user_id)

                if sources:
                    if conf >= 0.5:
                        badge = f'<span class="conf-high">● High Confidence {conf:.2f}</span>'
                    elif conf >= 0.25:
                        badge = f'<span class="conf-med">● Med Confidence {conf:.2f}</span>'
                    else:
                        badge = f'<span class="conf-low">● Low Confidence {conf:.2f}</span>'
                    st.markdown(badge, unsafe_allow_html=True)

                    with st.expander("🔗 Reference Citations"):
                        for i, src in enumerate(sources):
                            meta_s = src.get("metadata", {})
                            st.markdown(f"""
                            <div class="source-block">
                                <div class="source-header">
                                    <span class="source-name">📄 {meta_s.get('source','Unknown')}</span>
                                    <span class="source-page">Page {meta_s.get('page_number','N/A')}</span>
                                </div>
                                <div class="source-text">"{src['text'][:300]}..."</div>
                                <div class="source-footer">Re-rank Quality: {src.get('rerank_score',0):.4f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "confidence": conf,
                    "timestamp": ans_ts
                })
                save_chat(user_id=st.session_state.user_id, role="assistant", content=answer, confidence=conf, sources=sources)

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            clear_history(st.session_state.user_id)
            st.rerun()

# ════════════════════════════════════════════
# TAB 2 — HISTORY
# ════════════════════════════════════════════
with tab_history:
    st.markdown("### 🗂️ Chat History")
    st.caption("Conversations are persisted in a local SQLite database grouped by date.")

    days = st.slider("Load history for the past N days", 1, 90, 30)
    history = load_chat_history(st.session_state.user_id, limit_days=days)

    if not history:
        st.info("No chat history found. Start a conversation in the Chat tab!")
    else:
        # ── Export Section ─────────────────────────────
        st.markdown("#### 📥 Export Data")
        e1, e2 = st.columns([3, 1])
        with e1:
            st.caption("Download your entire conversation history as a formatted document.")
        with e2:
            md_content = export_history_to_markdown(history)
            st.download_button(
                label="📥 Download .MD",
                data=md_content,
                file_name=f"documind_history_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        st.divider()

        for date_str, messages in history.items():
            # Format header: "Today", "Yesterday", or the actual date
            from datetime import date as dateobj
            today = dateobj.today().strftime("%Y-%m-%d")
            if date_str == today:
                label = f"📅 Today — {date_str}"
            else:
                label = f"📅 {date_str}"

            with st.expander(label, expanded=(date_str == today)):
                user_count = sum(1 for m in messages if m["role"] == "user")
                st.caption(f"{user_count} question(s) asked")

                for msg in messages:
                    role_icon = "🧑" if msg["role"] == "user" else "🤖"
                    role_label = "You" if msg["role"] == "user" else "DocuMind AI"
                    conf_text = ""
                    if msg["role"] == "assistant" and msg.get("confidence", 0) > 0:
                        conf_text = f" · Confidence: {msg['confidence']:.2f}"

                    st.markdown(
                        f"""<div class="source-block">
                            <b>{role_icon} {role_label}</b>
                            <span style="float:right;font-size:0.72rem;color:#aaa;">{msg['timestamp']}{conf_text}</span><br>
                            {msg['content'][:400]}{'...' if len(msg['content']) > 400 else ''}
                        </div>""",
                        unsafe_allow_html=True
                    )

                    # Show sources if any
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander("📎 Sources used"):
                            for src in msg["sources"]:
                                st.caption(
                                    f"📄 **{src.get('source','?')}** "
                                    f"(Page {src.get('page_number','N/A')}) — "
                                    f"Score: {src.get('rerank_score',0):.4f}"
                                )

# ════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════════
with tab_analytics:
    a = st.session_state.analytics
    st.markdown("### 📊 Advanced Performance Analytics")

    avg_latency = round(sum(a["response_times"]) / len(a["response_times"]), 2) if a["response_times"] else 0.0
    total_docs = len(st.session_state.uploaded_docs)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">📄</div>
            <div class="metric-value">{total_docs}</div>
            <div class="metric-label">Docs Indexed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">💬</div>
            <div class="metric-value">{a['total_queries']}</div>
            <div class="metric-label">Total Queries</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">{a['avg_confidence']:.2f}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">{avg_latency}s</div>
            <div class="metric-label">Avg Response Time</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 📈 Response Time Trend (seconds)")
        if a["response_times"]:
            st.line_chart(a["response_times"])
        else:
            st.info("Ask questions to see latency trends.")

    with col_right:
        st.markdown("#### 🔝 Most Frequent Queries")
        if a["query_text_history"]:
            # Sort and take top 5
            sorted_queries = dict(sorted(a["query_text_history"].items(), key=lambda item: item[1], reverse=True)[:5])
            st.bar_chart(sorted_queries)
        else:
            st.info("Query patterns will appear here.")

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown("#### 🛡️ Confidence Score History")
        if a["confidence_history"]:
            st.area_chart(a["confidence_history"])
    
    with col_r2:
        st.markdown("#### 📂 Knowledge Base Coverage")
        if a["doc_query_count"]:
            st.bar_chart(a["doc_query_count"])

# ════════════════════════════════════════════
# TAB 4 — DOCUMENTS
# ════════════════════════════════════════════
with tab_docs:
    st.markdown("### 📂 Document Explorer & Search")
    st.caption("Search across all documents or manage your library.")

    # ── KEYWORD SEARCH SECTION ───────────────────────
    search_query = st.text_input("🔍 Quick Keyword Search", placeholder="Type keywords to find direct matches...")
    
    if search_query:
        matches = st.session_state.keyword_search.search(search_query, top_k=5, user_id=st.session_state.user_id)
        if matches:
            st.markdown(f"**Found {len(matches)} relevant snippets:**")
            for m in matches:
                m_meta = m.get('metadata', {})
                highlighted = highlight_text(m['text'], search_query)
                st.markdown(f"""
                <div class="source-block" style="border-left-color: #fde047;">
                    <div class="source-header">
                        <span class="source-name">📄 {m_meta.get('source','Unknown')}</span>
                        <span class="source-page" style="background:#fde04750;">Page {m_meta.get('page_number','N/A')}</span>
                    </div>
                    <div class="source-text">{highlighted}...</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No direct keyword matches found.")
    
    st.markdown("---")

    docs = st.session_state.uploaded_docs
    total_docs   = len(docs)
    total_chunks = len(embedding_manager.chunks)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">📄</div>
            <div class="metric-value">{total_docs}</div>
            <div class="metric-label">Documents Indexed</div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">🧩</div>
            <div class="metric-value">{total_chunks}</div>
            <div class="metric-label">Total Chunks in Vector Store</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if docs:
        for doc in docs:
            fname = doc['filename']
            query_count = a["doc_query_count"].get(fname, 0)
            
            with st.container():
                cols = st.columns([5, 2, 2, 1])
                with cols[0]:
                    st.markdown(f"**📄 {fname}**")
                    st.caption(f"Uploaded: {doc['upload_date']}")
                with cols[1]:
                    st.markdown(f"🧩 **{doc['chunk_count']}** chunks")
                with cols[2]:
                    st.markdown(f"🔍 **{query_count}** queries")
                with cols[3]:
                    if st.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                        with st.spinner(f"Deleting {fname}..."):
                            # 1. Remove from Vector Store (FAISS)
                            embedding_manager.remove_document(fname)
                            # 2. Remove from BM25
                            st.session_state.keyword_search.remove_document(fname)
                            # 3. Remove from SQLite
                            delete_document_meta(st.session_state.user_id, fname)
                            # 4. Remove raw file
                            path = DATA_DIR / fname
                            if path.exists():
                                path.unlink()
                            
                            # Update Session State
                            st.session_state.uploaded_docs = load_document_meta(st.session_state.user_id)
                            st.success(f"Deleted {fname}")
                            st.rerun()
                st.divider()
    else:
        st.info("No documents uploaded yet. Use the sidebar to upload PDFs or TXT files.")
