from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import streamlit as st

@dataclass
class MessageState:
    content: str
    is_complete: bool
    message_type: str
    domain: Optional[str] = None
    avatar: Optional[str] = None
    suggestions: List[Dict[str, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class StateManager:
    @staticmethod
    def initialize_session_state():
        """Initialize all required session state variables."""
        defaults = {
            'messages': {},
            'current_message_id': None,
            'settings': {
                'temperature': 0.7,
                'top_p': 0.95,
                'top_k': 40,
                'max_tokens': 2048
            }
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def update_message(message_id: str, content: str, is_complete: bool = False, 
                      message_type: str = "default", domain: Optional[str] = None,
                      avatar: Optional[str] = None, suggestions: List[Dict[str, str]] = None):
        """Thread-safe message update using Streamlit's session state."""
        if message_id not in st.session_state.messages:
            st.session_state.messages[message_id] = MessageState(
                content="",
                is_complete=False,
                message_type=message_type,
                domain=domain,
                avatar=avatar,
                suggestions=suggestions or []
            )
        
        current = st.session_state.messages[message_id]
        current.content += content
        current.is_complete = is_complete
        
        if suggestions:
            current.suggestions = suggestions
            
        return current

    @staticmethod
    def get_message(message_id: str) -> Optional[MessageState]:
        """Retrieve a message from the session state."""
        return st.session_state.messages.get(message_id)

    @staticmethod
    def get_all_messages() -> Dict[str, MessageState]:
        """Get all messages in chronological order."""
        return dict(sorted(
            st.session_state.messages.items(),
            key=lambda x: x[1].timestamp
        ))

    @staticmethod
    def clear_messages():
        """Clear all messages from the session state."""
        st.session_state.messages = {} 