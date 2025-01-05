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
        You are a specialist AI agent focused on enhancing and improving responses through:
        1. Critical analysis and fact-checking
        2. Adding relevant context and examples
        3. Identifying potential limitations or considerations
        4. Providing practical applications or implications
        5. Improving structure and readability
        6. Ensuring completeness while maintaining conciseness
        
        When enhancing responses:
        - Maintain the core accuracy of the original response
        - Add value through relevant details and context
        - Improve clarity and organization
        - Use markdown formatting for better readability
        - Keep the tone professional yet accessible
        - Ensure all statements are well-supported
        
        Your goal is to transform good responses into excellent ones while maintaining their essential meaning.
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
            
            Please enhance this response, focusing on:
            1. Adding any missing crucial information
            2. Improving the structure and flow
            3. Making the explanation clearer and more complete
            4. Ensuring proper markdown formatting
            
            Enhanced response:
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return base_response  # Fall back to base response if enhancement fails 