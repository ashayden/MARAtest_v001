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
        self.request_times = []  # Track request timestamps
        self.MIN_REQUEST_INTERVAL = 1.5  # 1.5 seconds between requests
        self.MAX_RPM = 3  # Free tier limit of 3 requests per minute
        self.WINDOW_SIZE = 60  # 60 seconds window for rate limiting
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def wait_if_needed(self, timeout: Optional[int] = None):
        """Check rate limits and wait if necessary."""
        with self._lock:
            current_time = time.time()
            
            # Clean up old request times
            self.request_times = [t for t in self.request_times if current_time - t < self.WINDOW_SIZE]
            
            # Check if we need to wait for the minimum interval
            if self.last_request_time > 0:
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.MIN_REQUEST_INTERVAL:
                    wait_time = self.MIN_REQUEST_INTERVAL - time_since_last
                    if timeout and wait_time > timeout:
                        raise Exception("Rate limit timeout exceeded")
                    time.sleep(wait_time)
            
            # Check if we've hit the per-minute limit
            if len(self.request_times) >= self.MAX_RPM:
                # Calculate time until oldest request expires
                wait_time = self.WINDOW_SIZE - (current_time - self.request_times[0])
                if timeout and wait_time > timeout:
                    raise Exception("Rate limit timeout exceeded")
                if wait_time > 0:
                    time.sleep(wait_time)
                # Clean up expired requests again after waiting
                current_time = time.time()
                self.request_times = [t for t in self.request_times if current_time - t < self.WINDOW_SIZE]
            
            # Record the request
            self.request_times.append(current_time)
            self.last_request_time = current_time

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
        
        System Configuration:
        - Model: Gemini 2.0 Flash Experimental
        - Rate Limits: 3 requests per minute
        - Request Interval: 1.5 seconds minimum between requests
        
        Your role is to:
        1. Analyze inputs thoroughly within rate limits
        2. Apply your specialized expertise
        3. Consider how your insights complement other agents
        4. Provide clear reasoning for your contributions
        5. Format responses for easy integration
        
        Rate Limit Guidelines:
        - Maximum 3 requests per minute
        - Wait 1.5 seconds between requests
        - Manual retry required if limits exceeded
        
        Remember:
        - You are part of a team of specialized agents
        - Each contribution should build on previous insights
        - Be explicit about your reasoning process
        - Highlight areas where other agents should focus
        - Respect rate limits and processing time
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
        
        # Add previous agent responses if any (limit to most recent and relevant)
        if previous_responses:
            # Take only the last response from each agent to reduce context size
            formatted_responses = []
            seen_agents = set()
            
            for resp in reversed(previous_responses):
                if isinstance(resp, dict) and 'agent' in resp:
                    agent = resp['agent']
                    if agent not in seen_agents:
                        formatted_responses.append(str(resp.get('text', '')).strip())
                        seen_agents.add(agent)
                else:
                    formatted_responses.append(str(resp).strip())
            
            if formatted_responses:
                prompt.append({
                    'text': "\nPrevious insights:\n" + "\n".join(formatted_responses[-3:])  # Limit to last 3 responses
                })
        
        # Add user input (ensure it's properly formatted and not too long)
        if isinstance(user_input, str):
            prompt.append({'text': user_input[:4000]})  # Limit length
        elif isinstance(user_input, dict):
            prompt.append({k: v[:4000] if isinstance(v, str) else v for k, v in user_input.items()})
        elif isinstance(user_input, list):
            for item in user_input[:3]:  # Limit number of items
                if isinstance(item, str):
                    prompt.append({'text': item[:4000]})
                elif isinstance(item, dict):
                    prompt.append({k: v[:4000] if isinstance(v, str) else v for k, v in item.items()})
        
        return prompt
    
    def generate_response(self, user_input: list, previous_responses: List[str] = None, stream: bool = True) -> Generator[str, None, None]:
        """Generate a response to the input, considering previous agent responses."""
        try:
            # Wait for rate limiter with timeout
            self.rate_limiter.wait_if_needed(timeout=10)  # 10 second timeout
            
            # Prepare complete prompt
            prompt = self.prepare_prompt(user_input, previous_responses)
            
            # Adjust output tokens based on role
            if self.role == "initializer":
                self.config.max_output_tokens = min(self.config.max_output_tokens, 1024)
            elif self.role.endswith("_specialist"):
                self.config.max_output_tokens = min(self.config.max_output_tokens, 1536)
            
            # Generate response with optimized settings
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p,
                    'top_k': self.config.top_k,
                    'max_output_tokens': self.config.max_output_tokens,
                    'stop_sequences': ["\n\n\n", "###", "```"]  # Prevent unnecessary output
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
            error_str = str(e)
            if "RATE_LIMIT_EXCEEDED" in error_str or "timeout exceeded" in error_str.lower():
                yield "Rate limit reached. Waiting to retry..."
                try:
                    # Wait longer and retry once with reduced tokens
                    time.sleep(5)
                    self.rate_limiter.wait_if_needed(timeout=15)
                    self.config.max_output_tokens = min(self.config.max_output_tokens, 512)
                    
                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': self.config.temperature,
                            'top_p': self.config.top_p,
                            'top_k': self.config.top_k,
                            'max_output_tokens': self.config.max_output_tokens,
                            'stop_sequences': ["\n\n\n", "###", "```"]
                        },
                        stream=stream
                    )
                    
                    if stream:
                        for chunk in response:
                            if chunk.text:
                                yield chunk.text
                    else:
                        yield response.text
                        
                except Exception:
                    yield "Could not complete request after retry. Please try again later."
            else:
                yield f"Error generating response: {error_str}" 