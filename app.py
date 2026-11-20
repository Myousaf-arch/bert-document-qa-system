import os
import re
import time
import torch
import streamlit as st
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import extract_and_chunk_pdfs
from vector_store import VectorStoreManager

# Threading optimization
torch.set_num_threads(os.cpu_count() or 4)

# Streamlit Page Configuration
st.set_page_config(
    page_title="CognitiveDoc AI | Enterprise Neural QA Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- EXACT WARM CREAM & MATTE CHARCOAL SAAS THEME ---
st.markdown("""
    <style>
    /* Global Page Background & Font */
    .stApp {
        background-color: #ECE7E1;
        color: #1A1A1A;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Default Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Force Hide Sidebar completely since Control Hub is now on Main Page */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* Custom File Uploader Card & Uploaded Document Details Fix */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #C8C0B4 !important;
        border-radius: 14px !important;
        padding: 10px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
    }
    
    /* Uploaded File Details Box (Card + Text) -> Made White & Visible */
    [data-testid="stFileUploaderFileData"] {
        background-color: #1E1E1E !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    [data-testid="stFileUploaderFileData"] * {
        color: #FFFFFF !important;
    }

    /* Success Alert Box Fix (Green Bar Text -> White) */
    .stAlert [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    .stAlert {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }

    /* Typography & Headers */
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A1A1A;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #666056;
        margin-bottom: 1.8rem;
    }

    /* Matte Charcoal Analytics Grid Cards */
    .metric-container {
        display: flex;
        gap: 16px;
        margin-bottom: 25px;
    }
    .metric-card {
        flex: 1;
        background: #1E1E1E;
        color: #ECE7E1;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #D4A373;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #A0A0A0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }

    /* Matte Charcoal Output Card */
    .result-card {
        background: #1E1E1E;
        color: #ECE7E1;
        border-left: 5px solid #D4A373;
        border-radius: 12px;
        padding: 22px;
        margin-top: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .result-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #D4A373;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .result-body {
        font-size: 1.2rem;
        font-weight: 500;
        color: #FFFFFF;
        line-height: 1.5;
    }

    /* Control Hub Section Box */
    .control-hub-box {
        background: #E2DDD5;
        border: 1px solid #D5CEC4;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
    }

    /* Amber / Gold Accent Interactive Buttons */
    .stButton>button {
        background: #D4A373 !important;
        color: #1A1A1A !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.65rem 1.4rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        background: #1E1E1E !important;
        color: #ECE7E1 !important;
        transform: translateY(-1px);
    }

    /* Clean Input Box */
    .stTextInput>div>div>input {
        background-color: #F7F4EF !important;
        color: #1A1A1A !important;
        border: 1px solid #D5CEC4 !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    /* Tabs Layout */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px;
        background-color: #E2DDD5;
        color: #555048;
        border: 1px solid #D5CEC4;
        padding: 0px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E1E1E !important;
        color: #ECE7E1 !important;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"

@st.cache_resource(show_spinner=False)
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_vector_manager():
    return VectorStoreManager()

tokenizer, model = load_qa_model()
vector_manager = load_vector_manager()

def deep_clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\xa0', ' ')
    text = re.sub(r'(?<=\b[a-zA-Z])\s+(?=[a-zA-Z]\b)', '', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])(?=[a-zA-Z])', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def intent_rerank_chunks(user_query: str, retrieved_chunks: list):
    query_lower = user_query.lower()
    stop_words = {"what", "where", "does", "do", "they", "occur", "create", "and", "the", "are", "from", "is", "in", "it", "of", "to", "for", "with", "how"}
    query_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
    
    scored_chunks = []
    for chunk in retrieved_chunks:
        chunk_lower = chunk.lower()
        score = 0.0

        if "light reaction" in query_lower:
            if "thylakoid" in chunk_lower or "light reactions" in chunk_lower: score += 25.0
            if "nadph" in chunk_lower or "atp" in chunk_lower or "oxygen" in chunk_lower: score += 20.0
        
        if "dark reaction" in query_lower or "calvin" in query_lower:
            if "stroma" in chunk_lower or "calvin" in chunk_lower: score += 25.0

        if any(term in query_lower for term in ["create", "produce", "form", "yield"]):
            if any(p in chunk_lower for p in ["atp", "nadph", "oxygen", "carbohydrate", "sugar"]): score += 15.0

        for word in query_words:
            score += chunk_lower.count(word) * 2.5

        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_matched_score = scored_chunks[0][0] if scored_chunks else 0.0
    return [item[1] for item in scored_chunks], top_matched_score

# State Management
if "processed" not in st.session_state: st.session_state.processed = False
if "chunk_count" not in st.session_state: st.session_state.chunk_count = 0
if "file_count" not in st.session_state: st.session_state.file_count = 0
if "uploaded_files_list" not in st.session_state: st.session_state.uploaded_files_list = []

# --- MAIN DASHBOARD INTERFACE ---
st.markdown('<div class="brand-title">CognitiveDoc AI Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Enterprise Neural Document Intelligence & Fact Extraction Engine</div>', unsafe_allow_html=True)

# Executive Analytics Grid
st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">Engine Status</div>
            <div class="metric-value">{'ONLINE' if st.session_state.processed else 'STANDBY'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Active Index</div>
            <div class="metric-value">{st.session_state.chunk_count} Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Core Architecture</div>
            <div class="metric-value">BERT + FAISS</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- CONTROL HUB ON MAIN PAGE (RIGHT BELOW METRICS / SEARCH) ---
st.markdown("""
    <div style="font-size: 1.25rem; font-weight:700; color:#1A1A1A; margin-bottom: 12px; margin-top: 10px;">
        ⚡ Control Hub (Document Ingestion Engine)
    </div>
""", unsafe_allow_html=True)

with st.container():
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.uploaded_files_list = uploaded_files

    # Display Uploaded Files with Individual Delete (Cross) Buttons
    if st.session_state.uploaded_files_list:
        st.markdown("<p style='font-size:0.9rem; font-weight:600; margin-top:10px;'>Active Uploaded Documents:</p>", unsafe_allow_html=True)
        for idx, file in enumerate(list(st.session_state.uploaded_files_list)):
            cols_file = st.columns([0.85, 0.15])
            with cols_file[0]:
                st.text(file.name)
            with cols_file[1]:
                if st.button("✕ Remove", key=f"del_file_{idx}"):
                    st.session_state.uploaded_files_list.pop(idx)
                    st.rerun()

    if st.button("⚡ Index Documents", use_container_width=True):
        if st.session_state.uploaded_files_list:
            with st.spinner("Executing FAISS Indexing Pipeline..."):
                raw_chunks = extract_and_chunk_pdfs(st.session_state.uploaded_files_list)
                cleaned_chunks = [deep_clean_pdf_text(c) for c in raw_chunks if len(c.strip()) > 20]
                
                if cleaned_chunks:
                    success = vector_manager.build_index(cleaned_chunks)
                    if success:
                        st.session_state.processed = True
                        st.session_state.file_count = len(st.session_state.uploaded_files_list)
                        st.session_state.chunk_count = len(cleaned_chunks)
                        st.success(f"System Ready: {len(cleaned_chunks)} Chunks Vectorized")
                else:
                    st.error("Text Extraction Failed.")
        else:
            st.warning("Upload PDF source first.")

st.divider()

# Main UI Tabs Navigation
tab1, tab2 = st.tabs(["🔍 Neural Search Console", "📊 System Pipeline Inspector"])

with tab1:
    user_query = st.text_input("Natural Language Query", placeholder="e.g. What are the key outputs of light dependent reactions?")
    
    if st.button("Run Neural Query", type="primary"):
        if not st.session_state.processed:
            st.error("No Vector Index loaded! Upload documents in the Control Hub above first.")
        elif not user_query.strip():
            st.warning("Query prompt cannot be empty.")
        else:
            start_time = time.time()
            with st.spinner("Running Transformer Span Extraction..."):
                retrieved_chunks = vector_manager.search_similar(user_query, k=6)
                cleaned_chunks = [deep_clean_pdf_text(c) for c in retrieved_chunks]
                re_ranked_chunks, match_score = intent_rerank_chunks(user_query, cleaned_chunks)
                top_contexts = re_ranked_chunks[:3]

                best_answer = ""
                best_score = float("-inf")
                best_chunk = ""

                for context in top_contexts:
                    inputs = tokenizer(user_query, context, return_tensors="pt", max_length=512, truncation=True)
                    with torch.no_grad(): outputs = model(**inputs)
                    
                    sequence_ids = inputs.sequence_ids(0)
                    if sequence_ids is None: continue

                    context_token_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]
                    if not context_token_indices: continue

                    start_logits, end_logits = outputs.start_logits[0], outputs.end_logits[0]
                    mask = torch.ones_like(start_logits) * float('-inf')
                    mask[context_token_indices] = 0.0
                    
                    start_idx = torch.argmax(start_logits + mask).item()
                    end_idx = torch.argmax(end_logits + mask).item()
                    
                    if start_idx <= end_idx:
                        total_score = (start_logits[start_idx] + end_logits[end_idx]).item()
                        if total_score > best_score:
                            answer_tokens = inputs.input_ids[0][start_idx:end_idx + 1]
                            span_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                            if len(span_text) > 2 and span_text.lower() not in ["[cls]", "[sep]"]:
                                best_score = total_score
                                best_answer = span_text
                                best_chunk = context

                latency = round(time.time() - start_time, 2)

                # Guardrail for irrelevant questions
                if best_answer and match_score > 0.0:
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">🎯 Extracted Fact Answer (Confidence Score: {round(best_score, 2)} | Latency: {latency}s)</div>
                            <div class="result-body">{best_answer.capitalize()}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if best_chunk:
                        st.markdown("### 📜 Verified Document Context")
                        st.info(best_chunk)
                else:
                    st.markdown(f"""
                        <div class="result-card" style="border-left-color: #E53E3E;">
                            <div class="result-title" style="color: #FC8181;">⚠️ Out of Context / Irrelevant Query</div>
                            <div class="result-body">I'm sorry, I'm not designed for your irrelevant question.</div>
                        </div>
                    """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Vector Index Topology")
    st.caption("Inspecting top embedding chunks retrieved from memory store.")
    if st.session_state.processed:
        st.json({
            "file_count": st.session_state.file_count, 
            "total_chunks": st.session_state.chunk_count, 
            "embedding_model": "all-MiniLM-L6-v2"
        })
    else:
        st.write("No active session data available.")