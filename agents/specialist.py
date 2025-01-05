import google.generativeai as genai
from typing import Optional

class SpecialistAgent:
    """Specialist agent class for enhancing responses."""
    
    def __init__(self):
        """Initialize the specialist agent with Gemini model."""
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Set parameters for focused enhancement
        self.generation_config = {
            'temperature': 0.3,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # Enhancement prompt template
        self.enhancement_prompt = """
        You are a specialist AI agent focused on enhancing responses through:
        1. Adding essential context and details
        2. Improving clarity and structure
        3. Identifying key implications
        4. Ensuring accuracy and completeness
        
        Enhance the following response while maintaining conciseness and clarity.
        """
    
    def enhance_response(self, original_prompt: str, base_response: str) -> str:
        """
        Enhance the base response with additional context and clarity.
        
        Args:
            original_prompt (str): Original user prompt
            base_response (str): Response from the base agent
            
        Returns:
            str: Enhanced response
        """
        try:
            # Create enhancement prompt
            prompt = f"""
            Original Question: {original_prompt}
            
            Initial Response: {base_response}
            
            {self.enhancement_prompt}
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return base_response  # Fall back to base response if enhancement fails 