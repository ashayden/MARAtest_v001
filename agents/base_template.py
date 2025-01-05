import google.generativeai as genai
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

class AgentTemplate(ABC):
    """Abstract base template for creating specialized AI agents."""
    
    def __init__(
        self,
        model_name: str = 'gemini-2.0-flash-exp',
        temperature: float = 0.7,
        persona: Optional[Dict[str, Any]] = None,
        custom_instructions: Optional[str] = None
    ):
        """Initialize the agent template."""
        self.model = genai.GenerativeModel(model_name)
        
        # Default configuration
        self.generation_config = {
            'temperature': temperature,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # Set agent personality
        self.persona = persona or {
            'tone': 'professional',
            'style': 'clear and direct',
            'expertise_level': 'expert',
            'communication_style': 'structured'
        }
        
        # Base system prompt template
        self.base_prompt = """
        You are a specialized AI agent with the following characteristics:
        - Tone: {tone}
        - Style: {style}
        - Expertise Level: {expertise_level}
        - Communication Style: {communication_style}
        
        Core Principles:
        1. Structure and Organization:
           - Clear, logical flow of information
           - Well-defined sections and hierarchies
           - Consistent formatting and style
        
        2. Content Quality:
           - Accurate and verified information
           - Relevant examples and context
           - Practical applications
           - Specific details and data points
        
        3. Communication Standards:
           - Professional and appropriate tone
           - Clear and accessible language
           - Engaging and natural flow
           - Appropriate level of detail
        
        4. Response Format:
           - Use markdown for structure
           - Include relevant sections
           - Maintain consistent style
           - Focus on readability
        
        {custom_instructions}
        """
        
        # Update with custom instructions
        self.system_prompt = self.base_prompt.format(
            **self.persona,
            custom_instructions=custom_instructions or ""
        )
    
    @abstractmethod
    def preprocess_input(self, prompt: str) -> str:
        """Preprocess and validate the input prompt."""
        pass
    
    @abstractmethod
    def postprocess_response(self, response: str) -> str:
        """Postprocess and format the model's response."""
        pass
    
    def generate_response(
        self, 
        prompt: str,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate a response using the configured model and prompts.
        
        Args:
            prompt: User input to process
            stream: Whether to stream the response
            stream_callback: Optional callback for streaming responses
            
        Returns:
            str: Generated response
        """
        try:
            # Preprocess input
            processed_prompt = self.preprocess_input(prompt)
            
            # Combine prompts
            full_prompt = f"""
            {self.system_prompt}
            
            Input: {processed_prompt}
            """
            
            if stream and stream_callback:
                # Stream response with callback
                response_stream = self.model.generate_content(
                    full_prompt,
                    generation_config=self.generation_config,
                    stream=True
                )
                
                final_response = ""
                for chunk in response_stream:
                    if chunk.text:
                        final_response += chunk.text
                        stream_callback(chunk.text)
                
                return self.postprocess_response(final_response)
            else:
                # Generate complete response
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=self.generation_config
                )
                return self.postprocess_response(response.text)
            
        except Exception as e:
            error_msg = "We're unable to process your request at the moment. Please try again."
            if stream_callback:
                stream_callback(error_msg)
            return error_msg 