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
        
        # Response generation uses lower temperature for more focused output
        self.response_config = {
            'temperature': 0.2,  # Even lower for more precise responses
            'top_p': 0.95,
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
            thought_prompt = f"""Think through how you would answer this question. Focus on:
            1. Key concepts and entities
            2. Important perspectives to consider
            3. Main arguments and evidence
            4. Potential challenges or limitations
            
            Question: {prompt}"""
            
            thought_response = self.model.generate_content(
                thought_prompt,
                generation_config=self.generation_config
            )
            
            # Second call for concise, focused response
            response_prompt = f"""Using the analysis below, write a clear and direct response.
            Do not mention the analysis process or include phrases like "based on the analysis" or "the user wants."
            Focus only on presenting the information itself.
            
            Analysis: {thought_response.text}
            
            Question: {prompt}
            
            Requirements:
            1. Start immediately with the relevant information
            2. Use clear, direct statements
            3. Organize information logically
            4. Be concise but complete
            5. Do not repeat information
            6. Do not include meta-commentary about the response itself
            """
            
            response = self.model.generate_content(
                response_prompt,
                generation_config=self.response_config
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
            thought_prompt = f"""Think through how you would answer this question. Focus on:
            1. Key concepts and entities
            2. Important perspectives to consider
            3. Main arguments and evidence
            4. Potential challenges or limitations
            
            Question: {prompt}"""
            
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
            response_prompt = f"""Using the analysis below, write a clear and direct response.
            Do not mention the analysis process or include phrases like "based on the analysis" or "the user wants."
            Focus only on presenting the information itself.
            
            Analysis: {thought_response}
            
            Question: {prompt}
            
            Requirements:
            1. Start immediately with the relevant information
            2. Use clear, direct statements
            3. Organize information logically
            4. Be concise but complete
            5. Do not repeat information
            6. Do not include meta-commentary about the response itself
            """
            
            response_stream = self.model.generate_content(
                response_prompt,
                generation_config=self.response_config,
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