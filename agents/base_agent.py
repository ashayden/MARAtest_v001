import google.generativeai as genai
from typing import Optional, Tuple, Dict, Callable, Generator

class BaseAgent:
    """Base agent class for handling primary interactions with Gemini AI."""
    
    def __init__(self):
        """Initialize the base agent with Gemini model."""
        # Initialize with the flash thinking model for detailed reasoning
        self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-1219')
        
        # Set default parameters for generation - higher temperature for more creative thinking
        self.generation_config = {
            'temperature': 0.8,  # Increased for more detailed thinking
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
    
    def process(self, prompt: str) -> Tuple[str, str]:
        """
        Process the user input and generate both thought process and response.
        
        Args:
            prompt (str): User input prompt
            
        Returns:
            Tuple[str, str]: A tuple containing (thought_process, final_response)
        """
        try:
            # First call for detailed thought process
            thought_prompt = f"""Think through how you would answer this question in detail, 
            considering multiple perspectives and reasoning steps: {prompt}"""
            thought_response = self.model.generate_content(
                thought_prompt,
                generation_config=self.generation_config
            )
            
            # Second call for initial response based on thoughts
            response_prompt = f"""Based on the following thought process, provide an initial response:
            
            Thought Process: {thought_response.text}
            
            Question: {prompt}
            """
            response = self.model.generate_content(
                response_prompt,
                generation_config=self.generation_config
            )
            
            return thought_response.text, response.text
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            return error_msg, error_msg
    
    def stream_process(
        self, 
        prompt: str,
        stream_callback: Optional[Callable[[str, str, float], None]] = None
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Stream both thought process and response with progress updates.
        
        Args:
            prompt (str): User input prompt
            stream_callback: Optional callback function for progress updates
                           Signature: callback(thought_chunk, response_chunk, progress)
            
        Yields:
            Generator[Tuple[str, str], None, None]: Generates tuples of (thought_chunk, response_chunk)
        """
        try:
            # Stream thought process
            thought_prompt = f"""Think through how you would answer this question in detail, 
            considering multiple perspectives and reasoning steps: {prompt}"""
            
            thought_response = ""
            thought_stream = self.model.generate_content(
                thought_prompt,
                generation_config=self.generation_config,
                stream=True
            )
            
            # Process thought stream
            for chunk in thought_stream:
                if chunk.text:
                    thought_response += chunk.text
                    if stream_callback:
                        stream_callback(chunk.text, "", 0.25)  # 25% progress
                    yield chunk.text, ""
            
            # Stream response based on thoughts
            response_prompt = f"""Based on the following thought process, provide an initial response:
            
            Thought Process: {thought_response}
            
            Question: {prompt}
            """
            
            response_stream = self.model.generate_content(
                response_prompt,
                generation_config=self.generation_config,
                stream=True
            )
            
            # Process response stream
            final_response = ""
            for chunk in response_stream:
                if chunk.text:
                    final_response += chunk.text
                    if stream_callback:
                        stream_callback("", chunk.text, 0.75)  # 75% progress
                    yield "", chunk.text
            
            # Final yield with complete outputs
            if stream_callback:
                stream_callback(thought_response, final_response, 1.0)  # 100% progress
            yield thought_response, final_response
            
        except Exception as e:
            error_msg = f"An error occurred while streaming: {str(e)}"
            if stream_callback:
                stream_callback(error_msg, error_msg, 1.0)
            yield error_msg, error_msg 