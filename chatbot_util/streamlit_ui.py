import streamlit as st
import uuid


# Session state for chat history and active session
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}
if "active_session" not in st.session_state:
    st.session_state["active_session"] = None

# Function to display loading page
def show_loading():
    with st.spinner("Loading..."):
        pass  # Simulate some loading time

# Function to display chat history
def display_chat_history(session_id):
    if session_id in st.session_state["chat_history"]:
        for message in st.session_state["chat_history"][session_id]:
            st.write(message)

# Function to handle user input and update chat history
def process_user_input(session_id):
    user_input = st.text_input("Enter your message:")
    if user_input:
        st.session_state["chat_history"][session_id].append(f"User: {user_input}")
        # Simulate chatbot response (replace with your actual logic)
        response = f"Chatbot {session_id}: Thanks for your message!"
        st.session_state["chat_history"][session_id].append(response)
        st.echo(response)

# Main app logic
show_loading()  # Simulate some loading time (optional)

# Sidebar for selecting chat session
all_sessions = list(st.session_state["chat_history"].keys())

# Display session list with clickable links
session_options = []
for session_id in all_sessions:
    session_options.append(session_id)

selected_session = st.sidebar.selectbox(
    "Select Chat Session", session_options or ["No Sessions Available"]
)

# Create new session button
if st.sidebar.button("Create New Session"):
    new_session_id = str(uuid.uuid4()) # len(st.session_state["chat_history"]) + 1
    st.session_state["chat_history"][new_session_id] = []
    selected_session = new_session_id

# Extract session ID from selected option (handles both new and existing sessions)
if selected_session != "No Sessions Available":
    session_id = str(selected_session)
else:
    session_id = None


# Display chat history and user input for the selected session
st.header(f"Chat Session {session_id if session_id else ''}")
display_chat_history(session_id)
if session_id:  # Only show input field if a session is selected
    process_user_input(session_id)

# Update active session state
st.session_state["active_session"] = session_id
