import streamlit as st
from services.extractor import extract_text_from_pdf
from services.chunker import chunk_text
from services.embeddings import embed_texts
from services.vectorstore import create_collection, upsert_embeddings
from services.rag_pipeline import rag_answer
import time

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist aesthetic with enhanced UI - NIGHT MODE
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container - Dark Mode */
    .stApp {
        background: linear-gradient(to bottom, #0f172a, #1e293b);
    }
    
    /* Dark mode for all text */
    body, p, span, div, label {
        color: #e2e8f0 !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Title styling - Dark Mode */
    h1 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
        margin-bottom: 0.5rem !important;
        text-align: center;
    }
    
    /* Subtitle - Dark Mode */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    
    /* Upload section - Dark Mode */
    .upload-section {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    
    /* Chat messages - Dark Mode */
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    .user-message {
        background: #334155;
        border: 1px solid #475569;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #1e293b;
        border: 1px solid #3b82f6;
        margin-right: 2rem;
    }
    
    .message-label {
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        color: #94a3b8;
    }
    
    .message-content {
        color: #e2e8f0;
        line-height: 1.6;
    }
    
    /* Input styling - Dark Mode */
    .stTextInput input {
        border-radius: 24px !important;
        border: 2px solid #475569 !important;
        background: #1e293b !important;
        color: #e2e8f0 !important;
        padding: 0.75rem 1.25rem !important;
        font-size: 0.95rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Button styling - Dark Mode */
    .stButton button {
        border-radius: 24px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton button:hover {
        background: #2563eb !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 8px !important;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Status messages - Dark Mode */
    .status-message {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
        animation: slideDown 0.3s ease-in;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border-left: 4px solid #10b981;
    }
    
    .status-info {
        background: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border-left: 4px solid #3b82f6;
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fcd34d;
        border-left: 4px solid #f59e0b;
    }
    
    /* Loading spinner - Dark Mode */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #334155;
        border-top-color: #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .loading-text {
        margin-left: 1rem;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    
    /* Info tooltip - Dark Mode */
    .info-icon {
        display: inline-block;
        width: 18px;
        height: 18px;
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 18px;
        font-size: 12px;
        font-weight: bold;
        cursor: help;
        margin-left: 8px;
    }
    
    /* Typing indicator - Dark Mode */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        padding: 1rem 1.25rem;
        background: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 12px;
        margin-right: 2rem;
        margin-bottom: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #60a5fa;
        border-radius: 50%;
        margin: 0 3px;
        animation: pulse 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    /* Info panel - Dark Mode */
    .info-panel {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #93c5fd;
    }
    
    .info-panel-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        color: #60a5fa;
    }
    
    .info-panel ul {
        color: #cbd5e1;
    }
    
    /* Stats card - Dark Mode */
    .stats-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    
    .stat-value {
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "show_welcome_modal" not in st.session_state:
    st.session_state.show_welcome_modal = True
if "processing" not in st.session_state:
    st.session_state.processing = False

# Welcome Modal using Streamlit dialog
if st.session_state.show_welcome_modal:
    @st.dialog("👋 Welcome to RAG Chatbot")
    def show_welcome():
        st.markdown("""
        **How it works:**
        
        1. Upload a PDF document
        2. We'll process and analyze the content
        3. Ask any questions about the document
        4. Get accurate, context-aware answers
        
        **💡 Tip:** The more specific your questions, the better the answers!
        """)
        
        if st.button("Got it!", type="primary", use_container_width=True):
            st.session_state.show_welcome_modal = False
            st.rerun()
    
    show_welcome()

# Header
st.markdown("# 💬 RAG Chatbot")
st.markdown('<p class="subtitle">Upload a PDF and ask questions about its content</p>', unsafe_allow_html=True)

# Info panel
with st.expander("ℹ️ How to use this chatbot", expanded=False):
    st.markdown("""
    <div class="info-panel">
        <div class="info-panel-title">📚 Getting Started</div>
        <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
            <li>Upload a PDF document using the file uploader below</li>
            <li>Wait for the document to be processed (this may take a moment)</li>
            <li>Start asking questions in the chat input</li>
        </ul>
        
        <div class="info-panel-title" style="margin-top: 1rem;">✨ Best Practices</div>
        <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
            <li>Ask specific questions for better answers</li>
            <li>Reference specific sections or topics from the document</li>
            <li>You can ask follow-up questions based on previous answers</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Upload section
with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    uploaded = st.file_uploader("📄 Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    
    if uploaded and not st.session_state.pdf_processed:
        st.session_state.processing = True
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Extract and chunk
            status_text.info("📄 Extracting text from PDF...")
            progress_bar.progress(25)
            time.sleep(0.5)
            text = extract_text_from_pdf(uploaded)
            
            status_text.info("✂️ Splitting into chunks...")
            progress_bar.progress(50)
            time.sleep(0.5)
            chunks = chunk_text(text)
            st.session_state.chunks = chunks
            
            # Embed
            status_text.info("🧠 Creating embeddings...")
            progress_bar.progress(75)
            time.sleep(0.5)
            embeddings = embed_texts(chunks)
            
            status_text.info("💾 Storing in vector database...")
            create_collection()
            upsert_embeddings(embeddings, chunks)
            progress_bar.progress(100)
            time.sleep(0.3)
            
            st.session_state.pdf_processed = True
            st.session_state.processing = False
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Success message with stats
            st.success(f"✓ Successfully processed your PDF!")
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'''
                <div class="stats-card">
                    <div>
                        <div class="stat-label">Chunks</div>
                        <div class="stat-value">{len(chunks)}</div>
                    </div>
                    <div style="font-size: 1.5rem;">📦</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="stats-card">
                    <div>
                        <div class="stat-label">Characters</div>
                        <div class="stat-value">{len(text):,}</div>
                    </div>
                    <div style="font-size: 1.5rem;">📝</div>
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                <div class="stats-card">
                    <div>
                        <div class="stat-label">Status</div>
                        <div class="stat-value" style="font-size: 1rem; color: #10b981;">Ready</div>
                    </div>
                    <div style="font-size: 1.5rem;">✅</div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.balloons()
            
        except Exception as e:
            st.session_state.processing = False
            progress_bar.empty()
            status_text.empty()
            st.error(f"⚠️ Error processing PDF: {str(e)}")
    
    elif st.session_state.pdf_processed:
        st.markdown(f'''
        <div class="status-message status-info">
            📄 PDF loaded: {len(st.session_state.chunks)} chunks ready for questions
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat interface
if st.session_state.pdf_processed:
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''
            <div class="chat-message user-message">
                <div class="message-label">You</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="chat-message assistant-message">
                <div class="message-label">Assistant</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Show typing indicator if processing
    typing_placeholder = st.empty()
    if st.session_state.processing:
        typing_placeholder.markdown('''
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Chat input
    query = st.chat_input("Ask a question about your PDF...")
    
    if query and not st.session_state.processing:
        # Set processing flag
        st.session_state.processing = True
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Show typing indicator
        typing_placeholder.markdown('''
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Generate response
        answer = rag_answer(query)
        
        # Clear typing indicator
        typing_placeholder.empty()
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Reset processing flag
        st.session_state.processing = False
        
        # Rerun to show the new messages
        st.rerun()

else:
    # Empty state - Dark Mode
    st.markdown('''
    <div style="text-align: center; padding: 3rem 1rem; color: #64748b;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
        <div style="font-size: 1.1rem; margin-bottom: 0.5rem; font-weight: 500; color: #94a3b8;">No PDF uploaded yet</div>
        <div style="font-size: 0.9rem; color: #64748b;">Upload a PDF above to start chatting with your documents</div>
    </div>
    ''', unsafe_allow_html=True)