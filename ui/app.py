import streamlit as st
from typing import Optional
import uuid

from ui.components.messages import MessageDisplay
from agents.orchestrator import AgentOrchestrator
from utils.state import StateManager

class ChatInterface:
    def __init__(self):
        """Initialize chat interface components."""
        self.message_display = MessageDisplay()
        self.orchestrator = AgentOrchestrator()
        self.state_manager = StateManager()
    
    def render(self):
        """Render the main chat interface."""
        # Initialize session state
        self.state_manager.initialize_session_state()
        
        # Set up the layout
        st.title("AI Assistant")
        st.caption("Powered by Google's Gemini API")
        
        # Add settings and documentation
        self._render_settings()
        
        # Show processing status
        status_container = st.empty()
        
        # Render existing messages
        for message_id in self.state_manager.get_all_messages():
            self.message_display.display_message(message_id)
        
        # Handle new input
        if prompt := st.chat_input("Message"):
            message_id = str(uuid.uuid4())
            
            # Show processing status
            with status_container:
                with st.status("Processing...", expanded=True) as status:
                    success = self.orchestrator.process_with_streaming(
                        prompt=prompt,
                        message_id=message_id,
                        timeout=30  # Configurable timeout
                    )
                    
                    if success:
                        status.update(label="Complete!", state="complete")
                    else:
                        status.update(label="Error occurred", state="error")
        
        # Handle suggestion clicks
        if 'next_prompt' in st.session_state:
            prompt = st.session_state.next_prompt
            del st.session_state.next_prompt
            st.rerun()
    
    def _render_settings(self):
        """Render settings and documentation panels."""
        col1, col2 = st.columns(2)
        
        # Model Settings
        with col1:
            with st.expander("⚙️ Model Settings", expanded=False):
                st.subheader("🔬 Domain Specialist Settings")
                st.info("""
                Adjust these settings to control how domain specialists analyze and respond to queries.
                Other agents use fixed settings for consistency.
                """)
                
                # Temperature slider
                st.session_state.settings['temperature'] = st.slider(
                    "Creativity Level",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.settings['temperature'],
                    step=0.1,
                    help="Higher values make specialist responses more creative but less focused"
                )
                
                # Top P slider
                st.session_state.settings['top_p'] = st.slider(
                    "Response Diversity",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.settings['top_p'],
                    step=0.05,
                    help="Controls how diverse specialist responses can be"
                )
                
                # Top K slider
                st.session_state.settings['top_k'] = st.slider(
                    "Choice Range",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.settings['top_k'],
                    step=1,
                    help="Controls how many options specialists consider"
                )
                
                # Max Output Tokens slider
                st.session_state.settings['max_tokens'] = st.slider(
                    "Maximum Response Length",
                    min_value=256,
                    max_value=4096,
                    value=st.session_state.settings['max_tokens'],
                    step=256,
                    help="Maximum length of specialist responses"
                )
        
        # Documentation
        with col2:
            with st.expander("ℹ️ How it Works", expanded=False):
                st.markdown("""
                This AI Assistant uses the Gemini 2.0 Flash Experimental model to provide comprehensive analysis through a coordinated multi-agent system.
                
                ### System Configuration
                - **Model**: Gemini 2.0 Flash Experimental
                - **Rate Limits**: 3 requests per minute
                - **Request Interval**: 1.5 seconds minimum
                
                ### Process Flow
                1. **Initial Analysis**
                   - Analyzes input topic
                   - Identifies required expertise
                   - Creates analysis framework
                
                2. **Domain Specialists**
                   - Created dynamically based on topic
                   - Maximum 3 specialists per query
                   - Each provides domain-specific insights
                
                3. **Final Synthesis**
                   - Integrates all specialist inputs
                   - Creates cohesive final report
                   - Maintains academic structure
                
                ### Features
                - Real-time streaming responses
                - Dynamic specialist creation
                - Persistent chat history
                - Downloadable reports
                - Follow-up suggestions
                """)

def main():
    """Main application entry point."""
    # Configure page
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and render chat interface
    chat = ChatInterface()
    chat.render()

if __name__ == "__main__":
    main() 