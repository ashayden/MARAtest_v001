"""Technical Expert Agent for handling user queries."""
from typing import Optional, Generator
import google.generativeai as genai

class AgentConfig:
    """Configuration for the agent."""
    def __init__(self, temperature: float = 0.7, top_p: float = 0.95,
                 top_k: int = 40, max_output_tokens: int = 2048):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens

class TechnicalExpert:
    """Technical expert agent specializing in detailed technical analysis."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the TechnicalExpert agent."""
        self.config = config
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.conversation_history = []
        
        # Expert prompt engineering
        self.system_prompt = """You are a technical expert AI assistant with deep knowledge across multiple domains.
        Your responses should:
        1. Begin with a clear, concise summary of the key points
        2. Provide detailed technical analysis when relevant
        3. Include specific examples and references
        4. Highlight potential implications or considerations
        5. Suggest practical applications or next steps
        
        When analyzing technical content:
        - Break down complex concepts into understandable parts
        - Identify underlying principles and patterns
        - Consider edge cases and limitations
        - Provide context for technical decisions
        - Reference relevant best practices
        
        Format your responses with:
        - Clear section headings when appropriate
        - Bullet points for key details
        - Code snippets when relevant
        - Markdown formatting for readability
        """
    
    def add_to_history(self, message: dict):
        """Add a message to the conversation history."""
        self.conversation_history.append(message)
    
    def get_history(self):
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def prepare_prompt(self, user_input: list) -> list:
        """Prepare the complete prompt with system context and history."""
        # Start with system prompt
        prompt = [{'text': self.system_prompt}]
        
        # Add conversation history
        for msg in self.conversation_history:
            prompt.append(msg)
        
        # Add current user input
        prompt.extend(user_input)
        
        return prompt
    
    def generate_response(self, user_input: list, stream: bool = True) -> Generator[str, None, None]:
        """Generate a response to the user's input."""
        try:
            # Prepare complete prompt
            prompt = self.prepare_prompt(user_input)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p,
                    'top_k': self.config.top_k,
                    'max_output_tokens': self.config.max_output_tokens,
                },
                stream=stream
            )
            
            if stream:
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            else:
                yield response.text
            
            # Add to history
            self.add_to_history({
                'role': 'assistant',
                'content': response.text if not stream else ''.join([chunk.text for chunk in response if chunk.text])
            })
            
        except Exception as e:
            yield f"Error generating response: {str(e)}" 