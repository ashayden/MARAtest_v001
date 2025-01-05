import google.generativeai as genai
from typing import Optional, Dict, Callable

class BaseAgent:
    """Base agent class for handling primary interactions with Gemini AI."""
    
    def __init__(self):
        """Initialize the base agent with Gemini model."""
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Set default parameters for generation
        self.generation_config = {
            'temperature': 0.7,  # Balanced between creativity and focus
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # System prompt template
        self.system_prompt = """
        You are a knowledgeable AI assistant focused on providing well-structured initial responses.
        
        Format your responses following this structure:
        1. Start with a clear, one-sentence overview
        2. Break down the main topics into logical sections
        3. Use bullet points for key details within sections
        4. Include specific examples or data points
        5. End with practical implications or applications
        
        Use markdown formatting:
        - # for main title
        - ## for section headers
        - **bold** for emphasis
        - Lists for multiple points
        
        Important guidelines:
        - Be concise but informative
        - Use natural, clear language
        - Focus on accuracy and relevance
        - Provide concrete details
        - Stay focused on the core question
        - Never include meta-commentary
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
            # Create a focused prompt that encourages structured response
            full_prompt = f"""
            {self.system_prompt}
            
            Provide a well-structured response to: {prompt}
            
            Remember to:
            1. Start with a clear overview
            2. Use logical sections
            3. Include specific details
            4. Maintain natural flow
            """
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return "We're unable to process your request at the moment. Please try again."
    
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
            
            Question: {prompt}
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
                        progress = min(len(final_response) / 1000, 0.99)  # Cap progress at 99%
                        stream_callback(chunk.text, progress)
            
            # Final progress update
            if stream_callback:
                stream_callback("", 1.0)
            
            return final_response
            
        except Exception as e:
            error_msg = "We're unable to process your request at the moment. Please try again."
            if stream_callback:
                stream_callback(error_msg, 1.0)
            return error_msg 