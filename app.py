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
    if "show_thoughts" not in st.session_state:
        st.session_state.show_thoughts = False

def display_chat_interface():
    """Display the chat interface."""
    st.title("🤖 Multi-Agent Gemini Chatbot")
    
    # Add toggle for showing thought process
    st.session_state.show_thoughts = st.sidebar.checkbox(
        "Show AI's Thought Process",
        value=st.session_state.show_thoughts,
        help="Toggle to show/hide the AI's reasoning process"
    )
    
    st.markdown("""
    This chatbot uses multiple AI agents powered by Google's Gemini 2.0 Flash Thinking model to provide enhanced reasoning and responses.
    """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and st.session_state.show_thoughts:
                with st.expander("🤔 Thought Process", expanded=True):
                    st.markdown(message["thoughts"])
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to discuss?"):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from agents
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # First, get base agent's response
                base_thoughts, base_response = st.session_state.agents["base"].process(prompt)
                
                # Then, get specialist agent's enhanced response
                final_thoughts, final_response = st.session_state.agents["specialist"].enhance_response(
                    prompt,
                    (base_thoughts, base_response)
                )
                
                # Display thought process if enabled
                if st.session_state.show_thoughts:
                    with st.expander("🤔 Thought Process", expanded=True):
                        st.markdown(final_thoughts)
                
                # Display final response
                st.markdown(final_response)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "thoughts": final_thoughts
                })

def main():
    initialize_session_state()
    display_chat_interface()

if __name__ == "__main__":
    main() 