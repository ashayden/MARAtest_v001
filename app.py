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

# Safety settings
safety_settings = {
    "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
    "SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
    "DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
}

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
    if "show_thoughts" not in st.session_state:
        st.session_state.show_thoughts = False
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "stream_responses" not in st.session_state:
        st.session_state.stream_responses = True

def update_streaming_response(thought_chunk: str, response_chunk: str, progress: float):
    """Callback function to update streaming responses in the UI."""
    if "current_response" not in st.session_state:
        st.session_state.current_response = {"thoughts": "", "response": ""}
    
    if thought_chunk:
        st.session_state.current_response["thoughts"] += thought_chunk
    if response_chunk:
        st.session_state.current_response["response"] += response_chunk
    
    # Update progress bar
    if "progress_bar" in st.session_state:
        st.session_state.progress_bar.progress(progress)

def display_chat_interface():
    """Display the chat interface."""
    st.title("🤖 Multi-Agent Gemini Chatbot")
    
    # Sidebar controls
    with st.sidebar:
        st.session_state.show_thoughts = st.checkbox(
            "Show AI's Thought Process",
            value=st.session_state.show_thoughts,
            help="Toggle to show/hide the AI's reasoning process"
        )
        st.session_state.stream_responses = st.checkbox(
            "Stream Responses",
            value=st.session_state.stream_responses,
            help="Toggle to enable/disable response streaming"
        )
        
        # Display token usage
        st.metric("Total Tokens Used", st.session_state.token_count)
    
    st.markdown("""
    This chatbot uses multiple AI agents powered by Google's Gemini models to provide enhanced reasoning and responses.
    The base agent uses the Flash Thinking model for detailed reasoning, while the specialist agent uses the Flash model for quick enhancements.
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
        
        # Update token count
        st.session_state.token_count += count_tokens(prompt)
        
        # Get response from agents
        with st.chat_message("assistant"):
            if st.session_state.stream_responses:
                # Initialize progress bar and response containers
                st.session_state.progress_bar = st.progress(0)
                thoughts_container = st.empty()
                response_container = st.empty()
                
                # Stream responses
                st.session_state.current_response = {"thoughts": "", "response": ""}
                
                for thought_chunk, response_chunk in st.session_state.agents["base"].stream_process(
                    prompt, update_streaming_response
                ):
                    if st.session_state.show_thoughts:
                        thoughts_container.markdown(st.session_state.current_response["thoughts"])
                    response_container.markdown(st.session_state.current_response["response"])
                
                # Clean up progress bar
                del st.session_state.progress_bar
                
                final_thoughts = st.session_state.current_response["thoughts"]
                final_response = st.session_state.current_response["response"]
            else:
                with st.spinner("Thinking..."):
                    # Get base agent's response
                    final_thoughts, final_response = st.session_state.agents["base"].process(prompt)
                    
                    # Display thought process if enabled
                    if st.session_state.show_thoughts:
                        with st.expander("🤔 Thought Process", expanded=True):
                            st.markdown(final_thoughts)
                    
                    # Display final response
                    st.markdown(final_response)
            
            # Update token count
            st.session_state.token_count += count_tokens(final_thoughts) + count_tokens(final_response)
            
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