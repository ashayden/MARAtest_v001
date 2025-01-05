import google.generativeai as genai
from typing import Optional

class SpecialistAgent:
    """Specialist agent class for enhancing responses."""
    
    def __init__(self):
        """Initialize the specialist agent with Gemini model."""
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Set parameters for focused enhancement
        self.generation_config = {
            'temperature': 0.3,  # Lower temperature for more focused refinements
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # Enhancement prompt template
        self.enhancement_prompt = """
        You are an expert at improving and enriching content while maintaining its core structure.
        
        Enhance this response by:
        1. Enriching key points with:
           - Relevant examples
           - Supporting data
           - Practical applications
           - Real-world context
        
        2. Improving readability through:
           - Better transitions between sections
           - More engaging language
           - Clearer explanations
           - Natural flow of information
        
        3. Adding value with:
           - Industry best practices
           - Common pitfalls to avoid
           - Success factors
           - Implementation tips
        
        Important rules:
        - Preserve the original structure
        - Keep the professional tone
        - Maintain factual accuracy
        - Focus on practical value
        - Never add meta-commentary
        - Start directly with content
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
        # Check if base_response is an error message
        if "unable to process" in base_response.lower():
            return base_response
            
        try:
            # Create enhancement prompt
            prompt = f"""
            Original question: {original_prompt}
            
            Base content to enhance: {base_response}
            
            {self.enhancement_prompt}
            
            Focus areas for this topic:
            1. Practical implementation details
            2. Real-world applications
            3. Best practices and guidelines
            4. Common challenges and solutions
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception:
            return base_response 