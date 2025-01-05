import google.generativeai as genai
from typing import Optional, Dict

class BaseAgent:
    """Base agent class for handling primary interactions with Gemini AI."""
    
    def __init__(self):
        """Initialize the base agent with Gemini model."""
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Set default parameters for generation
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
    
    def process(self, prompt: str) -> str:
        """
        Process the user input and generate a response.
        
        Args:
            prompt (str): User input prompt
            
        Returns:
            str: Generated response
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def stream_process(
        self, 
        prompt: str,
        stream_callback: Optional[Callable[[str, float], None]] = None
    ) -> str:
        """
        Process the user input and stream the response.
        
        Args:
            prompt (str): User input prompt
            stream_callback: Optional callback function for progress updates
                           Signature: callback(response_chunk, progress)
            
        Returns:
            str: Complete generated response
        """
        try:
            response_stream = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                stream=True
            )
            
            final_response = ""
            for chunk in response_stream:
                if chunk.text:
                    final_response += chunk.text
                    if stream_callback:
                        stream_callback(chunk.text, len(final_response) / 1000)  # Rough progress estimate
            return final_response
            
        except Exception as e:
            error_msg = f"An error occurred while streaming: {str(e)}"
            if stream_callback:
                stream_callback(error_msg, 1.0)
            return error_msg 