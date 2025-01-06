"""Base template for collaborative agents."""
from typing import Optional, Generator, List
import google.generativeai as genai

class AgentConfig:
    """Configuration for agents."""
    def __init__(self, temperature: float = 0.7, top_p: float = 0.95,
                 top_k: int = 40, max_output_tokens: int = 2048):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens

class BaseAgent:
    """Base agent with collaborative capabilities."""
    
    def __init__(self, config: AgentConfig, role: str = "base"):
        """Initialize the base agent."""
        self.config = config
        self.role = role
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.conversation_history = []
        
        # Base system prompt
        self.system_prompt = """You are a collaborative AI agent working as part of a multi-agent system.
        Your role is to:
        1. Analyze inputs thoroughly
        2. Apply your specialized expertise
        3. Consider how your insights complement other agents
        4. Provide clear reasoning for your contributions
        5. Format responses for easy integration
        
        Remember:
        - You are part of a team of specialized agents
        - Each contribution should build on previous insights
        - Be explicit about your reasoning process
        - Highlight areas where other agents should focus
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
    
    def prepare_prompt(self, user_input: list, previous_responses: List[str] = None) -> list:
        """Prepare the complete prompt with system context, history, and previous agent responses."""
        # Start with system prompt
        prompt = [{'text': self.system_prompt}]
        
        # Add conversation history
        for msg in self.conversation_history:
            prompt.append(msg)
        
        # Add previous agent responses if any
        if previous_responses:
            prompt.append({
                'text': "\nPrevious agent insights:\n" + "\n".join(previous_responses)
            })
        
        # Normalize user input
        if isinstance(user_input, str):
            normalized_input = [{'text': user_input}]
        elif isinstance(user_input, dict):
            normalized_input = [user_input]
        elif isinstance(user_input, list):
            if not user_input:
                normalized_input = [{'text': ''}]
            elif isinstance(user_input[0], str):
                normalized_input = [{'text': user_input[0]}]
            elif isinstance(user_input[0], dict) and 'text' in user_input[0]:
                normalized_input = user_input
            else:
                normalized_input = [{'text': str(user_input[0])}]
        else:
            normalized_input = [{'text': str(user_input)}]
        
        # Add normalized user input
        prompt.extend(normalized_input)
        
        return prompt
    
    def generate_response(self, user_input: list, previous_responses: List[str] = None, stream: bool = True) -> Generator[str, None, None]:
        """Generate a response to the input, considering previous agent responses."""
        try:
            # Prepare complete prompt
            prompt = self.prepare_prompt(user_input, previous_responses)
            
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
                full_response = ""
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        yield chunk.text
                
                # Add to history with role context
                self.add_to_history({
                    'role': 'assistant',
                    'agent': self.role,
                    'content': full_response
                })
            else:
                # Add to history with role context
                self.add_to_history({
                    'role': 'assistant',
                    'agent': self.role,
                    'content': response.text
                })
                yield response.text
            
        except Exception as e:
            yield f"Error generating response: {str(e)}" 