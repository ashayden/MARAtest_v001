import google.generativeai as genai
from typing import Optional, Tuple, Dict

class SpecialistAgent:
    """Specialist agent class for enhancing responses with additional reasoning."""
    
    def __init__(self):
        """Initialize the specialist agent with Gemini model."""
        self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-1219')
        
        # Set parameters for more focused and analytical responses
        self.generation_config = {
            'temperature': 0.3,  # Lower temperature for more focused outputs
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # System prompts for enhanced reasoning
        self.thought_system_prompt = """
        You are a specialist AI agent focused on analyzing and improving thought processes through:
        1. Identifying key reasoning patterns and strategies
        2. Highlighting important considerations and assumptions
        3. Suggesting alternative perspectives or approaches
        4. Evaluating the completeness and logic of the thinking process
        
        Analyze the given thought process and enhance it while maintaining clarity and structure.
        """
        
        self.response_system_prompt = """
        You are a specialist AI agent focused on enhancing and improving responses through:
        1. Critical analysis and fact-checking
        2. Adding relevant context and examples
        3. Identifying potential limitations or considerations
        4. Providing practical applications or implications
        
        Analyze the given response and enhance it while maintaining clarity and conciseness.
        """
    
    def enhance_response(self, original_prompt: str, base_outputs: Tuple[str, str]) -> Tuple[str, str]:
        """
        Enhance both the thought process and final response.
        
        Args:
            original_prompt (str): Original user prompt
            base_outputs (Tuple[str, str]): Tuple containing (thought_process, final_response)
            
        Returns:
            Tuple[str, str]: Enhanced (thought_process, final_response)
        """
        base_thoughts, base_response = base_outputs
        
        try:
            # Enhance thought process
            thought_enhancement_prompt = f"""
            Original User Question: {original_prompt}
            
            Original Thought Process: {base_thoughts}
            
            {self.thought_system_prompt}
            
            Please provide an enhanced thought process that builds upon and improves the original thinking.
            """
            
            enhanced_thoughts = self.model.generate_content(
                thought_enhancement_prompt,
                generation_config=self.generation_config
            ).text
            
            # Enhance final response
            response_enhancement_prompt = f"""
            Original User Question: {original_prompt}
            
            Original Response: {base_response}
            
            Enhanced Thought Process: {enhanced_thoughts}
            
            {self.response_system_prompt}
            
            Please provide an enhanced response that incorporates insights from the improved thought process.
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