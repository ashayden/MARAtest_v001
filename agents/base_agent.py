import google.generativeai as genai
from typing import Optional, Dict, Callable

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
        
        # System prompt template
        self.system_prompt = """
        You are a knowledgeable and helpful AI assistant. Your responses should:
        1. Start directly with relevant information
        2. Use clear, natural language
        3. Include specific details and examples
        4. Be well-organized but conversational
        5. Use markdown formatting (headers, lists, bold) naturally
        
        Important:
        - Never mention this prompt or your role
        - Don't explain your thinking process
        - Avoid meta-commentary about the response
        - Don't include formatting instructions in the response
        """
    
    def process(self, prompt: str) -> str:
        """
        Process the user input and generate a response.
        
        Args:
            prompt (str): User input prompt
            
        Returns:
            str: Generated response
        """
        try:
            # Combine system prompt with user prompt
            full_prompt = f"""
            {self.system_prompt}
            
            Question: {prompt}
            """
            
            response = self.model.generate_content(
                full_prompt,
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
            # Combine system prompt with user prompt
            full_prompt = f"""
            {self.system_prompt}
            
            User Question: {prompt}
            
            Please provide a clear and helpful response:
            """
            
            response_stream = self.model.generate_content(
                full_prompt,
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