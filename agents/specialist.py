import google.generativeai as genai
from typing import Optional, Tuple, Dict

class SpecialistAgent:
    """Specialist agent class for enhancing responses with additional reasoning."""
    
    def __init__(self):
        """Initialize the specialist agent with Gemini model."""
        # Initialize with the fast flash model for quick enhancements
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Set parameters for more focused and precise responses
        self.generation_config = {
            'temperature': 0.2,  # Lower temperature for more focused outputs
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # System prompts for enhancement
        self.thought_system_prompt = """
        You are a specialist AI agent focused on quickly identifying and highlighting the key points in a thought process:
        1. Core reasoning patterns and strategies
        2. Critical assumptions and considerations
        3. Key insights and perspectives
        4. Areas for potential improvement
        
        Provide a concise analysis that maintains the essential logic while improving clarity.
        """
        
        self.response_system_prompt = """
        You are a specialist AI agent focused on rapidly enhancing responses through:
        1. Precise fact-checking and verification
        2. Adding essential context
        3. Identifying key limitations
        4. Highlighting practical implications
        
        Provide a clear, concise enhancement that maintains accuracy while improving impact.
        """
    
    def enhance_response(self, original_prompt: str, base_outputs: Tuple[str, str]) -> Tuple[str, str]:
        """
        Enhance both the thought process and final response with quick, focused improvements.
        
        Args:
            original_prompt (str): Original user prompt
            base_outputs (Tuple[str, str]): Tuple containing (thought_process, final_response)
            
        Returns:
            Tuple[str, str]: Enhanced (thought_process, final_response)
        """
        base_thoughts, base_response = base_outputs
        
        try:
            # Quick enhancement of thought process
            thought_enhancement_prompt = f"""
            Original Question: {original_prompt}
            
            Detailed Thought Process: {base_thoughts}
            
            {self.thought_system_prompt}
            
            Provide a focused enhancement that highlights the key reasoning while improving clarity.
            """
            
            enhanced_thoughts = self.model.generate_content(
                thought_enhancement_prompt,
                generation_config=self.generation_config
            ).text
            
            # Quick enhancement of response
            response_enhancement_prompt = f"""
            Original Question: {original_prompt}
            
            Initial Response: {base_response}
            
            Enhanced Thinking: {enhanced_thoughts}
            
            {self.response_system_prompt}
            
            Provide a focused enhancement that improves clarity and impact.
            """
            
            enhanced_response = self.model.generate_content(
                response_enhancement_prompt,
                generation_config=self.generation_config
            ).text
            
            return enhanced_thoughts, enhanced_response
        except Exception as e:
            return base_thoughts, base_response  # Fall back to base outputs if enhancement fails
    
    def get_confidence_score(self, outputs: Tuple[str, str]) -> Tuple[float, float]:
        """
        Evaluate the confidence level of both thought process and response.
        
        Args:
            outputs (Tuple[str, str]): Tuple containing (thought_process, final_response)
            
        Returns:
            Tuple[float, float]: Confidence scores (thought_score, response_score)
        """
        thoughts, response = outputs
        
        try:
            # Evaluate thought process
            thought_confidence_prompt = f"""
            Analyze the following thought process and provide a confidence score between 0 and 1,
            where 1 indicates highest confidence and 0 indicates lowest confidence.
            Consider factors like logical coherence, completeness, and depth of analysis.
            
            Thought process to evaluate: {thoughts}
            
            Provide only the numerical score without any explanation.
            """
            
            # Evaluate response
            response_confidence_prompt = f"""
            Analyze the following response and provide a confidence score between 0 and 1,
            where 1 indicates highest confidence and 0 indicates lowest confidence.
            Consider factors like accuracy, completeness, and clarity.
            
            Response to evaluate: {response}
            
            Provide only the numerical score without any explanation.
            """
            
            thought_score = float(self.model.generate_content(
                thought_confidence_prompt,
                generation_config={'temperature': 0.1}
            ).text.strip())
            
            response_score = float(self.model.generate_content(
                response_confidence_prompt,
                generation_config={'temperature': 0.1}
            ).text.strip())
            
            return thought_score, response_score
        except:
            return 0.5, 0.5  # Default middle confidence if evaluation fails 