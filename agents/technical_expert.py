"""Technical Expert Agent for handling user queries."""

class ResponseAgent:
    """A simple wrapper class for maintaining conversation state."""
    
    def __init__(self):
        """Initialize the ResponseAgent."""
        self.conversation_history = []
    
    def add_to_history(self, message: str, is_user: bool = True):
        """Add a message to the conversation history."""
        self.conversation_history.append({
            'content': message,
            'is_user': is_user
        })
    
    def get_history(self):
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = [] 