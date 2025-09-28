import streamlit as st
from chatbot import DialogueFSM, load_exercises, get_pinecone_index, llm  # Import from your chatbot.py
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Load data (once on app start)
@st.cache_resource
def load_chatbot_resources():
    exercises = load_exercises()
    pinecone_index = get_pinecone_index()
    return exercises, pinecone_index

# Streamlit page config
st.set_page_config(page_title="Math AI Tutor Chatbot", page_icon="🤖", layout="wide")

# Initialize session state for chat history and FSM
if "fsm" not in st.session_state:
    exercises, pinecone_index = load_chatbot_resources()
    st.session_state.fsm = DialogueFSM(exercises, pinecone_index, llm)
    st.session_state.messages = []  # List of {"role": "user" or "assistant", "content": str}
    st.session_state.language = "en"  # Default language

# Sidebar for settings (e.g., language switch)
with st.sidebar:
    st.title("Settings")
    selected_lang = st.selectbox("Language", ["English", "Hebrew"])
    if selected_lang == "Hebrew":
        st.session_state.language = "he"
    else:
        st.session_state.language = "en"
    # Reset chat button
    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.fsm = DialogueFSM(load_chatbot_resources()[0], load_chatbot_resources()[1], llm)
        st.session_state.fsm.user_language = st.session_state.language
        st.rerun()

# Main chat interface
st.title("🤖 Math AI Tutor Chatbot")
st.markdown("Start chatting! I'll guide you through math exercises step by step.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Handle SVG paths if in content (display as image)
        if "[SVG:" in message["content"]:
            # Extract path, e.g., [SVG: svg_outputs/file.svg]
            import re
            svg_match = re.search(r'\[SVG:\s*(.*?)\]', message["content"])
            if svg_match:
                svg_path = svg_match.group(1)
                if os.path.exists(svg_path):
                    with open(svg_path, "r") as f:
                        st.components.v1.html(f.read(), height=300)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with your FSM
    st.session_state.fsm.user_language = st.session_state.language
    response_dict = st.session_state.fsm.transition(prompt)
    response_text = response_dict["text"]
    svg_path = response_dict.get("svg_file_path")

    # Add assistant response to UI
    with st.chat_message("assistant"):
        st.markdown(response_text)
        if svg_path and os.path.exists(svg_path):
            with open(svg_path, "r") as f:
                st.components.v1.html(f.read(), height=300)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()  # Refresh to update UI