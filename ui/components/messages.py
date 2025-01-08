import streamlit as st
from typing import Optional, Dict
from utils.state import StateManager, MessageState

class MessageDisplay:
    def __init__(self):
        """Initialize message display component."""
        self.state_manager = StateManager()
    
    def display_message(self, message_id: str):
        """Display message with smart update detection."""
        message = self.state_manager.get_message(message_id)
        if not message:
            return
        
        # Create cache keys
        placeholder_key = f"placeholder_{message_id}"
        content_key = f"content_{message_id}"
        
        # Initialize placeholder if needed
        if placeholder_key not in st.session_state:
            st.session_state[placeholder_key] = st.empty()
        
        # Only update if content changed
        if (content_key not in st.session_state or 
            st.session_state[content_key] != message.content):
            
            with st.session_state[placeholder_key]:
                # Display avatar and content
                col1, col2 = st.columns([1, 20])
                with col1:
                    st.markdown(message.avatar or "🤖")
                with col2:
                    if message.domain:
                        st.markdown(f"**{message.domain}**")
                    st.markdown(message.content)
                    
                    if message.is_complete:
                        self._add_message_actions(message)
            
            # Cache the displayed content
            st.session_state[content_key] = message.content
    
    def _add_message_actions(self, message: MessageState):
        """Add interactive actions to complete messages."""
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("📋 Copy", key=f"copy_{hash(message.content)}"):
                self._copy_to_clipboard(message.content)
        
        if message.suggestions:
            st.markdown("### Follow-up Questions")
            for idx, suggestion in enumerate(message.suggestions):
                if st.button(
                    f"💡 {suggestion.get('headline', 'Follow up')}",
                    key=f"suggest_{idx}_{hash(str(suggestion))}"
                ):
                    st.session_state.next_prompt = suggestion.get('question', '')
                    st.rerun()
    
    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard using JavaScript."""
        js = f"""
            <script>
            function copyToClipboard() {{
                const el = document.createElement('textarea');
                el.value = {repr(text)};
                el.setAttribute('readonly', '');
                el.style.position = 'absolute';
                el.style.left = '-9999px';
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
                window.parent.postMessage({{
                    type: 'streamlit:showToast',
                    data: {{ message: 'Copied to clipboard!', kind: 'info' }}
                }}, '*');
            }}
            copyToClipboard();
            </script>
        """
        st.components.v1.html(js, height=0)
    
    def display_error(self, error_message: str):
        """Display error message."""
        with st.error(error_message):
            st.button("Retry", key=f"retry_{hash(error_message)}")
    
    def display_status(self, status: str, state: str = "info"):
        """Display status message."""
        if state == "error":
            st.error(status)
        elif state == "success":
            st.success(status)
        else:
            st.info(status) 