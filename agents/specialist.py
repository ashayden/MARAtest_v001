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
        Enhance this response by:
        1. Adding essential missing information
        2. Making the content more engaging and natural
        3. Improving organization without being too formal
        4. Using markdown naturally for readability
        
        Important rules:
        - Start directly with the enhanced content
        - Don't mention that you're enhancing anything
        - Don't explain your changes or process
        - Keep the natural flow of information
        - Maintain the original tone and style
        - Never include meta-commentary
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
            Question: {original_prompt}
            
            Response to enhance: {base_response}
            
            {self.enhancement_prompt}
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return base_response  # Fall back to base response if enhancement fails 