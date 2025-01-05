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

def count_tokens(text: str) -> int:
    """Count the number of tokens in the text using Gemini's tokenizer."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        return model.count_tokens(text).total_tokens
    except Exception:
        return 0

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agents" not in st.session_state:
        st.session_state.agents = {
            "base": BaseAgent(),
            "specialist": SpecialistAgent()
        }
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "stream_responses" not in st.session_state:
        st.session_state.stream_responses = True

def update_streaming_response(response_chunk: str, progress: float):
    """Callback function to update streaming responses in the UI."""
    if "current_response" not in st.session_state:
        st.session_state.current_response = ""
    
    st.session_state.current_response += response_chunk
    
    # Update progress bar
    if "progress_bar" in st.session_state:
        st.session_state.progress_bar.progress(progress)

def display_chat_interface():
    """Display the chat interface."""
    st.title("🤖 Multi-Agent Gemini Chatbot")
    
    # Sidebar controls
    with st.sidebar:
        st.session_state.stream_responses = st.checkbox(
            "Stream Responses",
            value=st.session_state.stream_responses,
            help="Toggle to enable/disable response streaming"
        )
        
        # Display token usage
        st.metric("Total Tokens Used", st.session_state.token_count)
    
    st.markdown("""
    This chatbot uses multiple AI agents powered by Google's Gemini Flash model to provide enhanced responses.
    The base agent generates initial responses, while the specialist agent enhances them for clarity and completeness.
    """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
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
        
        # Update token count
        st.session_state.token_count += count_tokens(prompt)
        
        # Get response from agents
        with st.chat_message("assistant"):
            if st.session_state.stream_responses:
                # Initialize progress bar and response container
                st.session_state.progress_bar = st.progress(0)
                response_container = st.empty()
                
                # Stream base response
                st.session_state.current_response = ""
                base_response = st.session_state.agents["base"].stream_process(
                    prompt, update_streaming_response
                )
                
                # Enhance the response
                final_response = st.session_state.agents["specialist"].enhance_response(
                    prompt, base_response
                )
                
                # Display final response
                response_container.markdown(final_response)
                
                # Clean up progress bar
                del st.session_state.progress_bar
            else:
                with st.spinner("Thinking..."):
                    # Get base agent's response
                    base_response = st.session_state.agents["base"].process(prompt)
                    
                    # Get specialist's enhanced response
                    final_response = st.session_state.agents["specialist"].enhance_response(
                        prompt, base_response
                    )
                    
                    # Display final response
                    st.markdown(final_response)
            
            # Update token count
            st.session_state.token_count += count_tokens(final_response)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response
            })

def main():
    initialize_session_state()
    display_chat_interface()

if __name__ == "__main__":
    main() 