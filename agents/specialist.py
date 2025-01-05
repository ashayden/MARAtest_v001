import google.generativeai as genai
from typing import Optional

class SpecialistAgent:
    """Specialist agent class for enhancing responses with additional reasoning."""
    
    def __init__(self):
        """Initialize the specialist agent with Gemini model."""
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Set parameters for more focused and analytical responses
        self.generation_config = {
            'temperature': 0.3,  # Lower temperature for more focused outputs
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # System prompt for enhanced reasoning
        self.system_prompt = """
        You are a specialist AI agent focused on enhancing and improving responses through:
        1. Critical analysis and fact-checking
        2. Adding relevant context and examples
        3. Identifying potential limitations or considerations
        4. Providing practical applications or implications
        
        Analyze the given response and enhance it while maintaining clarity and conciseness.
        """
    
    def enhance_response(self, original_prompt: str, base_response: str) -> str:
        """
        Enhance the base response with additional reasoning and context.
        
        Args:
            original_prompt (str): Original user prompt
            base_response (str): Response from the base agent
            
        Returns:
            str: Enhanced response
        """
        enhancement_prompt = f"""
        Original User Question: {original_prompt}
        
        Base Response: {base_response}
        
        {self.system_prompt}
        
        Please provide an enhanced response that builds upon and improves the base response.
        """
        
        try:
            response = self.model.generate_content(
                enhancement_prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return base_response  # Fall back to base response if enhancement fails
    
    def get_confidence_score(self, response: str) -> float:
        """
        Evaluate the confidence level of a response.
        
        Args:
            response (str): Response to evaluate
            
        Returns:
            float: Confidence score between 0 and 1
        """
        confidence_prompt = f"""
        Analyze the following response and provide a confidence score between 0 and 1,
        where 1 indicates highest confidence and 0 indicates lowest confidence.
        Consider factors like accuracy, completeness, and clarity.
        
        Response to evaluate: {response}
        
        Provide only the numerical score without any explanation.
        """
        
        try:
            score_response = self.model.generate_content(
                confidence_prompt,
                generation_config={'temperature': 0.1}
            )
            return float(score_response.text.strip())
        except:
            return 0.5  # Default middle confidence if evaluation fails 