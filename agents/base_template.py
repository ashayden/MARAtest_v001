"""Base template for collaborative agents."""
from typing import Optional, Generator, List
import google.generativeai as genai
import time
from threading import Lock

class RateLimiter:
    """Rate limiter for API requests."""
    _instance = None
    _lock = Lock()
    
    def __init__(self):
        self.last_request_time = 0
        self.requests_this_minute = 0
        self.requests_today = 0
        self.MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests
        self.MAX_RPM = 30  # Requests per minute
        self.MAX_RPD = 1000  # Requests per day
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def wait_if_needed(self):
        with self._lock:
            current_time = time.time()
            
            # Enforce minimum interval between requests
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.MIN_REQUEST_INTERVAL:
                time.sleep(self.MIN_REQUEST_INTERVAL - time_since_last)
            
            # Reset counters if a minute has passed
            if time_since_last > 60:
                self.requests_this_minute = 0
            
            # Check rate limits
            if self.requests_this_minute >= self.MAX_RPM:
                sleep_time = 60 - (current_time - self.last_request_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.requests_this_minute = 0
            
            # Update counters
            self.requests_this_minute += 1
            self.requests_today += 1
            self.last_request_time = time.time()

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
        self.rate_limiter = RateLimiter.get_instance()
        
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
            # Ensure all responses are strings and filter out None values
            formatted_responses = []
            for resp in previous_responses:
                if resp is not None:
                    # Convert any response to string, stripping any potential formatting
                    formatted_responses.append(str(resp).strip())
            
            if formatted_responses:
                prompt.append({
                    'text': "\nPrevious agent insights:\n" + "\n---\n".join(formatted_responses)
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
            # Wait for rate limiter
            self.rate_limiter.wait_if_needed()
            
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