"""
StratEdge — Business Strategy Consultant Chatbot
Streamlit App · Groq API + FAISS + all-MiniLM-L6-v2
Deployable on Streamlit Cloud
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── CONFIG ──────────────────────────────────────────────────────
INDEX_PATH   = "./faiss_index/index.bin"
CHUNKS_PATH  = "./faiss_index/chunks.pkl"
EMBED_MODEL  = "all-MiniLM-L6-v2" 
GROQ_MODEL   = "llama-3.1-8b-instant"   # fast + free. alternatives: mixtral-8x7b-32768, gemma2-9b-it
TOP_K        = 4
MIN_SCORE    = 0.25

# ── PAGE CONFIG ─────────────────────────────────────────────────
st.set_page_config(
    page_title="StratEdge — Business Consultant AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,400;0,700;1,400&display=swap');

:root {
    --bg:         #f5f3ee;
    --surface:    #ffffff;
    --surface2:   #f0ede6;
    --border:     #e2ddd6;
    --accent:     #2d6a4f;
    --accent2:    #1b4332;
    --accent-lt:  #d8f3dc;
    --text:       #1a1a1a;
    --muted:      #6b7280;
    --user-bg:    #e8f4f0;
    --user-border:#b7e0cc;
    --warn-bg:    #fff8e6;
    --danger:     #dc2626;
    --success:    #16a34a;
    --shadow:     0 2px 12px rgba(0,0,0,0.07);
}

html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 2px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.main .block-container { padding: 2rem 2.5rem 5rem; max-width: 820px; }

.app-header {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 24px;
    background: linear-gradient(135deg, #2d6a4f 0%, #1b4332 100%);
    border-radius: 16px; margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(45,106,79,0.25);
}
.app-header .icon {
    font-size: 2.2rem;
    background: rgba(255,255,255,0.15);
    border-radius: 12px; width: 54px; height: 54px;
    display: flex; align-items: center; justify-content: center;
}
.app-header .title {
    font-family: 'Fraunces', serif; font-size: 1.85rem;
    color: #fff; margin: 0; letter-spacing: -0.5px;
}
.app-header .sub {
    font-size: 0.74rem; color: rgba(255,255,255,0.65);
    text-transform: uppercase; letter-spacing: 0.8px; margin-top: 3px;
}
.header-badge {
    margin-left: auto;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.72rem; color: rgba(255,255,255,0.85);
}

.welcome-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 12px;
    padding: 16px 20px; margin-bottom: 20px;
    font-size: 0.875rem; color: var(--muted); line-height: 1.75;
    box-shadow: var(--shadow);
}
.welcome-card strong { color: var(--text); }
.welcome-card em { color: var(--accent); font-style: normal; font-weight: 600; }

[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
    box-shadow: var(--shadow) !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="stChatMessage"]:hover {
    box-shadow: 0 4px 18px rgba(0,0,0,0.1) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--user-bg) !important;
    border-color: var(--user-border) !important;
    border-left: 4px solid #059669 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 4px solid var(--accent) !important;
}
[data-testid="stChatMessage"] p {
    color: var(--text) !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
}

.src-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.src-pill {
    background: var(--accent-lt); border: 1px solid #95d5b2;
    color: var(--accent2); border-radius: 20px;
    padding: 3px 11px; font-size: 0.7rem; font-weight: 600;
}
.no-src-pill {
    background: var(--warn-bg); border: 1px solid #fcd34d;
    color: #92400e; border-radius: 20px;
    padding: 3px 11px; font-size: 0.7rem; font-weight: 600;
}

[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 2px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(45,106,79,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stButton button {
    background: var(--surface2) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.78rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.15s ease !important;
}
.stButton button:hover {
    background: var(--accent-lt) !important;
    border-color: var(--accent) !important;
    color: var(--accent2) !important;
    transform: translateX(3px) !important;
}

.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; vertical-align:middle; }
.dot.g { background:#16a34a; box-shadow:0 0 5px #16a34a55; }
.dot.r { background:#dc2626; box-shadow:0 0 5px #dc262655; }
.dot.a { background:#f59e0b; box-shadow:0 0 5px #f59e0b55; }
.srow  { display:flex; align-items:center; font-size:0.8rem; color:var(--muted); padding:4px 0; }

[data-testid="metric-container"] {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 14px;
}
[data-testid="metric-container"] label {
    color: var(--muted) !important; font-size: 0.72rem !important;
    text-transform: uppercase !important; letter-spacing: 0.5px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important; font-size: 1.3rem !important; font-weight: 700 !important;
}

.chunk-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.78rem;
}
.chunk-src   { color: var(--accent); font-weight: 600; }
.chunk-score { color: var(--muted); margin-left: 8px; }
.chunk-text  { color: var(--text); margin-top: 6px; line-height: 1.55; }

.sidebar-logo {
    font-family: 'Fraunces', serif; font-size: 1.35rem; color: var(--accent);
    border-bottom: 2px solid var(--border); padding-bottom: 8px; margin-bottom: 14px;
}

hr { border-color: var(--border) !important; }

@keyframes slideIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] { animation: slideIn 0.25s ease !important; }
</style>
""", unsafe_allow_html=True)


# ── GROQ API KEY ─────────────────────────────────────────────────
# For local run: set GROQ_API_KEY in your environment or paste it below
# For Streamlit Cloud: add it in App Settings → Secrets as GROQ_API_KEY = "your_key"

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ── CACHED RESOURCES ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL, cache_folder="./model_cache")


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ── RAG CORE ────────────────────────────────────────────────────

def retrieve(query, index, chunks, embedder, top_k=TOP_K, min_score=MIN_SCORE):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score >= min_score:
            results.append({**chunks[idx], "score": float(score)})
    return results


def build_prompt(query, context_chunks):
    context_str = ""
    for i, c in enumerate(context_chunks, 1):
        context_str += f"[Source {i}: {c['source']} | relevance: {c['score']:.2f}]\n{c['text']}\n\n"
    return f"""You are StratEdge, an expert business strategy consultant specializing in startup and company failures.
Provide evidence-based, actionable strategic advice grounded ONLY in the provided context.

STRICT RULES:
1. Answer ONLY from the context below. Do NOT use outside knowledge.
2. If context is insufficient, say: "I don't have enough data on this in my knowledge base."
3. Always reference the source document when citing specific findings.
4. Structure your response with clear numbered points when appropriate.
5. Be direct, specific, and actionable.
6. Never fabricate statistics, case names, or company data.

--- CONTEXT ---
{context_str}--- END CONTEXT ---

QUESTION: {query}

STRATEGIC ANALYSIS:"""


def stream_response(query, index, chunks, embedder, client):
    retrieved = retrieve(query, index, chunks, embedder)

    if not retrieved:
        yield ("chunk", "⚠️ I don't have relevant information in my knowledge base to answer this confidently.\n\nTry asking about startup failures, funding challenges, market fit issues, or business model problems.")
        yield ("sources", [])
        yield ("raw_chunks", [])
        yield ("grounded", False)
        return

    prompt = build_prompt(query, retrieved)

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
            stream=True
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield ("chunk", token)
    except Exception as e:
        yield ("chunk", f"❌ Groq API error: {str(e)}")
        yield ("sources", [])
        yield ("raw_chunks", [])
        yield ("grounded", False)
        return

    yield ("sources", list({c["source"] for c in retrieved}))
    yield ("raw_chunks", retrieved)
    yield ("grounded", True)


# ── SESSION STATE ───────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ── LOAD RESOURCES ──────────────────────────────────────────────

embedder      = load_embedder()
index, chunks = load_index()
client        = get_groq_client()
kb_ok         = index is not None and chunks is not None
groq_ok       = client is not None


# ── SIDEBAR ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div class='sidebar-logo'>📊 StratEdge</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:18px;'>Business Strategy AI</div>", unsafe_allow_html=True)

    st.markdown("**System Status**")
    st.markdown(f"""
    <div class='srow'><span class='dot {"g" if kb_ok else "r"}'></span>
        Knowledge Base: {"Loaded (" + str(index.ntotal) + " vectors)" if kb_ok else "Not found"}</div>
    <div class='srow'><span class='dot {"g" if groq_ok else "r"}'></span>
        Groq API: {"Connected" if groq_ok else "Key missing"}</div>
    <div class='srow'><span class='dot g'></span> Embedder: {EMBED_MODEL}</div>
    <div class='srow'><span class='dot a'></span> LLM: {GROQ_MODEL}</div>
    """, unsafe_allow_html=True)

    if not kb_ok:
        st.warning("⚠️ Run `rag_pipeline.ipynb` to build the knowledge base first.")

    if not groq_ok:
        st.error("❌ Groq API key missing.\n\nLocal: set `GROQ_API_KEY` env variable.\nCloud: add to Streamlit Secrets.")
        with st.expander("How to get a free Groq key"):
            st.markdown("""
            1. Go to [console.groq.com](https://console.groq.com)
            2. Sign up (free)
            3. Click **API Keys** → **Create API Key**
            4. Copy the key
            
            **Local:** set in terminal before running:
            ```
            set GROQ_API_KEY=your_key_here
            streamlit run app.py
            ```
            
            **Streamlit Cloud:** App Settings → Secrets:
            ```
            GROQ_API_KEY = "your_key_here"
            ```
            """)

    st.markdown("---")

    if kb_ok:
        c1, c2 = st.columns(2)
        with c1: st.metric("Vectors", f"{index.ntotal:,}")
        with c2: st.metric("Chunks",  f"{len(chunks):,}")
    st.metric("Queries Asked", st.session_state.query_count)

    st.markdown("---")
    st.markdown("**Settings**")
    TOP_K     = st.slider("Chunks to retrieve", 1, 8, TOP_K)
    MIN_SCORE = st.slider("Min relevance score", 0.10, 0.70, MIN_SCORE, 0.05)
    show_ctx  = st.toggle("Show retrieved context", False)

    st.markdown("---")
    st.markdown("**💡 Quick Questions**")
    suggestions = [
        "Why do most startups fail?",
        "What role does cash flow play in failures?",
        "How does poor market fit cause failure?",
        "What are early warning signs of collapse?",
        "How do funding issues destroy startups?",
        "What mistakes do founders commonly make?",
    ]
    for q in suggestions:
        if st.button(q, key=f"s_{q}", use_container_width=True):
            st.session_state["prefill"] = q
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()


# ── MAIN ────────────────────────────────────────────────────────

st.markdown("""
<div class='app-header'>
    <div class='icon'>📊</div>
    <div>
        <div class='title'>StratEdge</div>
        <div class='sub'>AI Business Strategy Consultant · Groq LLM + RAG · Powered by Your Documents</div>
    </div>
    <div class='header-badge'>⚡ Groq Powered</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class='welcome-card'>
        👋 <strong>Welcome to StratEdge!</strong><br>
        Ask anything about startup failures, business model collapse, funding issues,
        and strategic pitfalls — answered from <em>your own documents</em>, powered by Groq LLM.<br><br>
        Responses are grounded in your knowledge base. Sources are always cited. Try a question from the sidebar!
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    role    = msg["role"]
    content = msg["content"]
    with st.chat_message(role, avatar="🧑" if role == "user" else "📊"):
        st.markdown(content)
        if role == "assistant":
            sources    = msg.get("sources", [])
            raw_chunks = msg.get("raw_chunks", [])
            if sources:
                pills = "".join([f"<span class='src-pill'>📄 {s}</span>" for s in sources])
                st.markdown(f"<div class='src-row'>{pills}</div>", unsafe_allow_html=True)
            if show_ctx and raw_chunks:
                with st.expander(f"🔍 View retrieved context ({len(raw_chunks)} chunks)"):
                    for i, chunk in enumerate(raw_chunks, 1):
                        st.markdown(f"""
                        <div class='chunk-card'>
                            <span class='chunk-src'>#{i} {chunk['source']}</span>
                            <span class='chunk-score'>score: {chunk['score']:.3f}</span>
                            <div class='chunk-text'>{chunk['text'][:400]}{'…' if len(chunk['text'])>400 else ''}</div>
                        </div>""", unsafe_allow_html=True)

# ── INPUT ───────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill", "")
prompt  = st.chat_input("Ask a business strategy question…") or prefill

if prompt and prompt.strip():
    if not kb_ok:
        st.error("Knowledge base not found. Run `rag_pipeline.ipynb` first.")
    elif not groq_ok:
        st.error("Groq API key missing. See sidebar for setup instructions.")
    else:
        query = prompt.strip()

        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.query_count += 1

        with st.chat_message("assistant", avatar="📊"):
            placeholder    = st.empty()
            full_answer    = ""
            final_sources  = []
            final_chunks   = []
            final_grounded = True

            with st.spinner("Searching knowledge base…"):
                for event_type, payload in stream_response(query, index, chunks, embedder, client):
                    if event_type == "chunk":
                        full_answer += payload
                        placeholder.markdown(full_answer + "▌")
                    elif event_type == "sources":
                        final_sources = payload
                    elif event_type == "raw_chunks":
                        final_chunks = payload
                    elif event_type == "grounded":
                        final_grounded = payload

            placeholder.markdown(full_answer)

            if final_sources:
                pills = "".join([f"<span class='src-pill'>📄 {s}</span>" for s in final_sources])
                st.markdown(f"<div class='src-row'>{pills}</div>", unsafe_allow_html=True)

            if show_ctx and final_chunks:
                with st.expander(f"🔍 View retrieved context ({len(final_chunks)} chunks)"):
                    for i, chunk in enumerate(final_chunks, 1):
                        st.markdown(f"""
                        <div class='chunk-card'>
                            <span class='chunk-src'>#{i} {chunk['source']}</span>
                            <span class='chunk-score'>score: {chunk['score']:.3f}</span>
                            <div class='chunk-text'>{chunk['text'][:400]}{'…' if len(chunk['text'])>400 else ''}</div>
                        </div>""", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role":        "assistant",
            "content":     full_answer,
            "sources":     final_sources,
            "raw_chunks":  final_chunks,
            "is_grounded": final_grounded
        })
