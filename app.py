import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from agents.technical_expert import ResponseAgent
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Custom CSS for animations and styling
st.markdown("""
<style>
/* Custom container for chat messages */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    animation: fadeIn 0.5s ease-out;
}

.user-message {
    background-color: #2e3d49;
    border-left: 5px solid #4CAF50;
}

.bot-message {
    background-color: #1e2a35;
    border-left: 5px solid #2196F3;
}

/* Fade in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = ResponseAgent()

def display_chat_message(message: str, is_user: bool):
    """Display a chat message with animation."""
    message_type = "user" if is_user else "bot"
    st.markdown(
        f'<div class="chat-message {message_type}-message">{message}</div>',
        unsafe_allow_html=True
    )

def stream_callback(chunk: str):
    """Callback function for streaming responses."""
    # Get the last message container
    if 'current_response' not in st.session_state:
        st.session_state.current_response = ''
    
    st.session_state.current_response += chunk
    
    # Update the message in place
    if 'message_placeholder' in st.session_state:
        st.session_state.message_placeholder.markdown(
            f'<div class="chat-message bot-message">{st.session_state.current_response}</div>',
            unsafe_allow_html=True
        )

def main():
    st.title("AI Assistant")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message['content'], message['is_user'])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Display user message
        display_chat_message(user_input, True)
        st.session_state.messages.append({
            'content': user_input,
            'is_user': True
        })
        
        # Create placeholder for bot response
        st.session_state.message_placeholder = st.empty()
        
        # Reset current response
        st.session_state.current_response = ''
        
        # Generate response
        response = st.session_state.agent.generate_response(
            user_input,
            stream=True,
            stream_callback=stream_callback
        )
        
        # Add response to chat history
        st.session_state.messages.append({
            'content': response,
            'is_user': False
        })

if __name__ == "__main__":
    main() 