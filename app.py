import streamlit as st
from chatbot import ChatState, process_user_input

st.title("🤖 מתרגל מתמטיקה - Math Tutor Bot")
st.write("אני עוזר לימודי מתמטיקה בעברית! בוא נתחיל.")

# Session state
if "chat_state" not in st.session_state:
    st.session_state.chat_state = ChatState()
if "messages" not in st.session_state:
    st.session_state.messages = []

state = st.session_state.chat_state

# Display chat history
for msg in state.user_history:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["bot"])

# Initial bot message if new
if state.state == "ask_class" and not state.user_history:
    initial_msg = "שלום! איזו כיתה אתה לומד? (למשל: כיתה ז)"
    st.chat_message("assistant").write(initial_msg)
    state.user_history.append({"user": None, "bot": initial_msg})

# User input
if prompt := st.chat_input("הקלד כאן..."):
    st.chat_message("user").write(prompt)
    response, updated_state = process_user_input(prompt, state)
    st.session_state.chat_state = updated_state
    st.chat_message("assistant").write(response)
    st.rerun()