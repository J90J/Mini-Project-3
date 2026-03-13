import streamlit as st
import time

# Styling
st.set_page_config(page_title="Financial Agent Assistant", page_icon="💹", layout="wide")

st.markdown("""
<style>
    /* Dark Theme with Neon Green Accents */
    .stApp {
        background-color: #0d0d0d;
        color: #ffffff;
    }
    
    /* Neon Green Header Accent */
    h1, h2, h3 {
        color: #39FF14 !important;
        font-weight: 700;
        text-shadow: 0 0 5px rgba(57, 255, 20, 0.4);
    }
    
    /* Custom Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        border-right: 1px solid #333;
    }
    
    /* Neon Green Button Accent */
    .stButton > button {
        background-color: transparent !important;
        border: 1px solid #39FF14 !important;
        color: #39FF14 !important;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #39FF14 !important;
        color: #000000 !important;
        box-shadow: 0 0 10px #39FF14;
    }
    
    /* Chat bubbles */
    .stChatMessage {
        background-color: #1a1a1a;
        border-left: 3px solid #39FF14;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    [data-testid="chatAvatarIcon-user"] {
        background-color: #ffffff;
        color: #000;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #39FF14;
        color: #000;
    }
    
    /* Inputs */
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #262626 !important;
        color: #fff !important;
        border: 1px solid #404040 !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > textarea:focus {
        border-color: #39FF14 !important;
        box-shadow: 0 0 5px rgba(57, 255, 20, 0.5) !important;
    }
    
    /* Metadata formatting */
    .agent-meta {
        font-size: 0.8rem;
        color: #888;
        margin-top: 5px;
        border-top: 1px solid #333;
        padding-top: 5px;
    }
    .neon-text {
        color: #39FF14;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Imports for functional logic
import mp3_assignment

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Settings")
    
    agent_type = st.radio(
        "Agent Architecture",
        ["Baseline Worker", "Single Agent", "Multi-Agent"],
        index=1,
        help="Select the architecture pattern used to resolve queries."
    )
    
    model_selection = st.selectbox(
        "Model Version",
        ["gpt-4o-mini", "gpt-4o"],
        index=0
    )
    
    # Update the global ACTIVE_MODEL reference in mp3_assignment
    mp3_assignment.ACTIVE_MODEL = model_selection
    
    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("💹 Financial Intelligence Assistant")
st.markdown("Ask questions about stocks, market conditions, and financial fundamentals.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            st.markdown(f'<div class="agent-meta">Architecture: <span class="neon-text">{msg["meta"]["arch"]}</span> | Model: <span class="neon-text">{msg["meta"]["model"]}</span> | Time: {msg["meta"]["time"]:.2f}s</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("E.g., What is Apple's P/E ratio?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Analyzing request..."):
            start_time = time.time()
            
            try:
                if agent_type == "Baseline Worker":
                    res = mp3_assignment.run_baseline(prompt, verbose=False)
                    final_answer = res.answer
                elif agent_type == "Single Agent":
                    res = mp3_assignment.run_single_agent(prompt, verbose=False)
                    final_answer = res.answer
                else:
                    res = mp3_assignment.run_multi_agent(prompt, verbose=False)
                    final_answer = res.get("final_answer", "Error resolving multi-agent results.")
            except Exception as e:
                final_answer = f"Error processing request: {str(e)}"
                
            elapsed = time.time() - start_time
            
            message_placeholder.markdown(final_answer)
            
            meta_info = {
                "arch": agent_type,
                "model": model_selection,
                "time": elapsed
            }
            st.markdown(f'<div class="agent-meta">Architecture: <span class="neon-text">{meta_info["arch"]}</span> | Model: <span class="neon-text">{meta_info["model"]}</span> | Time: {meta_info["time"]:.2f}s</div>', unsafe_allow_html=True)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "meta": meta_info
            })
