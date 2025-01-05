import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from agents.base_agent import BaseAgent
from agents.specialist import SpecialistAgent

# Load environment variables
load_dotenv()

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit page configuration
st.set_page_config(
    page_title="Multi-Agent Gemini Chatbot",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agents" not in st.session_state:
        st.session_state.agents = {
            "base": BaseAgent(),
            "specialist": SpecialistAgent()
        }

def display_chat_interface():
    """Display the chat interface."""
    st.title("🤖 Multi-Agent Gemini Chatbot")
    st.markdown("""
    This chatbot uses multiple AI agents powered by Google's Gemini to provide enhanced reasoning and responses.
    """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to discuss?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from agents
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # First, get base agent's response
                base_response = st.session_state.agents["base"].process(prompt)
                # Then, get specialist agent's enhanced response
                final_response = st.session_state.agents["specialist"].enhance_response(prompt, base_response)
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})

def main():
    initialize_session_state()
    display_chat_interface()

if __name__ == "__main__":
    main() 