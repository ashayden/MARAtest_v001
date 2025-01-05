import google.generativeai as genai
from typing import Optional

class BaseAgent:
    """Base agent class for handling primary interactions with Gemini AI."""
    
    def __init__(self):
        """Initialize the base agent with Gemini model."""
        # Initialize the most capable Gemini model
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Set default parameters for generation
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.8,
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
    
    def stream_process(self, prompt: str) -> Optional[str]:
        """
        Process the user input and stream the response.
        
        Args:
            prompt (str): User input prompt
            
        Returns:
            Optional[str]: Generated response or None if error occurs
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
            return full_response
        except Exception as e:
            return f"An error occurred while streaming: {str(e)}" 